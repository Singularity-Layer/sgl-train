"""Job spec parsing. Torch-free by design — this module must import on a laptop.

The runner re-applies every clamp the backend applies. Two independent checks of
the same bound is deliberate: the backend is the policy, this is the fail-safe.
"""
from __future__ import annotations

import ipaddress
import os
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

KINDS = {"finetune", "decision_finetune", "scratch", "autoresearch"}
QUANTIZE = {"Q4_K_M", "Q5_K_M", "Q8_0"}

_RUN_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")

CLAMPS: dict[str, tuple[float, float]] = {
    "lora_rank": (4, 64),
    "lora_alpha": (8, 128),
    "lr": (1e-5, 1e-3),
    "epochs": (1, 10),
    "max_seq_len": (128, 4096),
    "batch_size": (1, 16),
    "grad_accum": (1, 32),
    "eval_split": (0.05, 0.3),
}
INTS = {"lora_rank", "lora_alpha", "epochs", "max_seq_len", "batch_size", "grad_accum"}


class SpecError(ValueError):
    """The spec is unusable. Always fatal — never retried."""


@dataclass(frozen=True)
class Recipe:
    lora_rank: int
    lora_alpha: int
    lr: float
    epochs: int
    max_seq_len: int
    batch_size: int
    grad_accum: int
    eval_split: float
    seed: int
    load_in_4bit: bool


@dataclass(frozen=True)
class Trial:
    """One auto-tune trial: the recipe fields a sweep varies. Everything else
    (seq len, batch, eval split, seed) is held constant so trials compare."""
    lr: float
    lora_rank: int
    lora_alpha: int
    epochs: int


@dataclass(frozen=True)
class Autotune:
    trials: tuple[Trial, ...]
    budget_seconds: int          # the prepaid block, as seen by the box
    trial_estimate_seconds: int  # server-side estimate for trial 1 (no history yet)
    reserve_seconds: int         # kept free for test eval + GGUF + upload
    experiment_url: str


@dataclass(frozen=True)
class RuntimeBudget:
    budget_seconds: int
    finalization_reserve_seconds: int
    estimated_p95_seconds: int
    profile_version: str


@dataclass(frozen=True)
class ResumeCheckpoint:
    source_run_id: str
    archive_url: str
    archive_sha256: str
    archive_bytes: int
    step: int
    epoch: int
    planned_steps: int
    slot: str
    model_id: str
    dataset_sha256: str
    recipe_sha256: str


MAX_TRIALS = 6


@dataclass(frozen=True)
class JobSpec:
    run_id: str
    kind: str
    hf_repo: str
    hf_revision: str
    decision_repo: str | None
    decision_revision: str | None
    unsloth_repo: str | None
    unsloth_revision: str | None
    unsloth_4bit_repo: str | None
    unsloth_4bit_revision: str | None
    dataset_url: str
    quantize: str | None
    progress_url: str
    result_url: str
    token: str
    artifact_urls: dict[str, str]
    recipe: Recipe
    autotune: Autotune | None = None
    runtime: RuntimeBudget | None = None
    checkpoint_urls: dict[str, str] | None = None
    resume: ResumeCheckpoint | None = None


def _req(d: dict, key: str, where: str) -> object:
    if key not in d or d[key] in (None, ""):
        raise SpecError(f"{where}: missing required field '{key}'")
    return d[key]


def _optional_pair(d: dict, key_a: str, key_b: str, where: str) -> tuple[str | None, str | None]:
    """Both-or-neither optional string pair, e.g. unsloth_repo/unsloth_revision.

    ABSENT means the key is missing entirely, or explicitly null. Anything else
    is PRESENT and must be an actual str with a non-empty .strip() — a present
    but malformed value (0, [], {}, True, "   ") raises rather than being read
    as absent, because silently downgrading a malformed pin to "no pin" would
    turn a backend bug into a quiet engine downgrade instead of a loud failure.

    Exactly one of the pair present is also fatal: a repo with no pinned
    revision (or vice versa) is not reproducible.

    Stored values are stripped.
    """
    def classify(key: str) -> str | None:
        if key not in d or d[key] is None:
            return None
        v = d[key]
        if not isinstance(v, str) or v.strip() == "":
            raise SpecError(
                f"{where}: '{key}' must be a non-empty string when present "
                f"(got {type(v).__name__})")
        return v.strip()

    a_val = classify(key_a)
    b_val = classify(key_b)
    if (a_val is None) != (b_val is None):
        raise SpecError(f"{where}: '{key_a}' and '{key_b}' must both be present or both be absent")
    return a_val, b_val


def _check_url(name: str, value: str) -> None:
    """https everywhere, and never a private/loopback/link-local target.

    The one escape hatch is SGL_ALLOW_LOCAL_URLS=1, which allows http and
    lifts the private/loopback/localhost block for local testing. Never set
    it in production: the backend is the only legitimate spec author, and it
    always hands out https URLs to public hosts.
    """
    allow_local = os.environ.get("SGL_ALLOW_LOCAL_URLS") == "1"
    parts = urlsplit(value)

    if parts.scheme != "https" and not (allow_local and parts.scheme == "http"):
        raise SpecError(f"spec: {name} must be https")

    host = parts.hostname
    if not host:
        raise SpecError(f"spec: {name} has no host")

    if allow_local:
        return

    if host.lower() == "localhost":
        raise SpecError(f"spec: {name} must not target a private or local address")

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        # host is a DNS name, not an IP literal — we don't do DNS resolution
        # here. The backend's URL allowlist (Plan 3) owns rejecting DNS names
        # that resolve to private/local addresses.
        return

    if (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    ):
        raise SpecError(f"spec: {name} must not target a private or local address")


def _parse_bool_flag(name: str, value: object) -> bool:
    """Strict boolean parse: only actual JSON true/false survive.

    isinstance(x, bool) — not truthiness — is deliberate: JSON's booleans decode
    to Python bool, so "true"/1/"yes" etc. are all rejected rather than silently
    coerced. A QLoRA flag is a training-shape decision (4-bit or not); guessing
    wrong from a loosely-typed value is worse than failing the spec outright.
    """
    if isinstance(value, bool):
        return value
    raise SpecError(f"recipe: '{name}' must be a boolean (true/false), got {value!r}")


def _clamp(name: str, value: object) -> float | int:
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise SpecError(f"recipe: '{name}' must be a number, got {value!r}") from None
    lo, hi = CLAMPS[name]
    num = max(lo, min(hi, num))
    return int(num) if name in INTS else num


def parse_spec(raw: dict) -> JobSpec:
    if not isinstance(raw, dict):
        raise SpecError("spec must be a JSON object")

    kind = str(_req(raw, "kind", "spec"))
    if kind not in KINDS:
        raise SpecError(f"spec: unknown kind '{kind}'")

    if kind == "decision_finetune":
        # Clef heads are not part of the standard llama.cpp GGUF contract.
        # A decision job must never look successful merely because a plain-LM
        # GGUF was produced without the trained head.
        if raw.get("quantize") not in (None, ""):
            raise SpecError("spec: decision_finetune does not support GGUF quantization")
        quantize = None
    else:
        quantize = str(_req(raw, "quantize", "spec"))
        if quantize not in QUANTIZE:
            raise SpecError(f"spec: unknown quantize '{quantize}'")

    base = raw.get("base_model") or {}
    if not isinstance(base, dict):
        raise SpecError("spec: base_model must be an object")
    hf_repo = str(_req(base, "hf_repo", "base_model"))
    # Fail-closed: an unpinned model means an unreproducible run, which means the
    # result can never be verified. Mirrors the sha256 pin on GGUF downloads.
    hf_revision = str(_req(base, "revision", "base_model"))
    # Optional, independently pinned source for the permissive Clef/PEFT lane.
    # It is deliberately separate from the legacy Unsloth mirror pins used by
    # ordinary SFT/QLoRA.
    decision_repo, decision_revision = _optional_pair(
        base, "decision_repo", "decision_revision", "base_model")
    # OPTIONAL: the unsloth engine's own mirror pin (e.g.
    # "unsloth/Qwen2.5-0.5B-Instruct" @ its own sha, NOT hf_repo/hf_revision's
    # sha). unsloth's FastLanguageModel.from_pretrained silently remaps a
    # canonical repo name to its mirror namespace but still applies whatever
    # revision it was given — that revision belongs to the canonical repo, not
    # the mirror, so the mirror lookup 404s (see runner/trainer_unsloth.py).
    # Handing unsloth its own mirror repo+revision directly makes that remap a
    # no-op. Without this pin the unsloth engine must never be attempted (see
    # runner/trainer.py: unsloth_pin_ok).
    unsloth_repo, unsloth_revision = _optional_pair(
        base, "unsloth_repo", "unsloth_revision", "base_model")
    # OPTIONAL: a SECOND, independent mirror pin for the 4-bit (QLoRA) path.
    # Live GPU smoke (2026-08-25) proved unsloth's remap goes one level
    # deeper when load_in_4bit=True: FastLanguageModel.from_pretrained
    # resolves to a THIRD repo — a dedicated pre-quantized "-unsloth-bnb-4bit"
    # mirror, distinct from both the canonical repo and the 16-bit unsloth
    # mirror — and again applies whatever revision it was handed, which
    # belongs to neither of those. unsloth_repo/unsloth_revision above pin the
    # 16-bit mirror; these two pin the 4-bit mirror. Same strict
    # both-or-neither validation, same "absent" semantics (missing key or
    # explicit null only). See runner/trainer_unsloth.py's load_unsloth and
    # runner/trainer.py's unsloth_pin_for.
    unsloth_4bit_repo, unsloth_4bit_revision = _optional_pair(
        base, "unsloth_4bit_repo", "unsloth_4bit_revision", "base_model")

    raw_recipe = raw.get("recipe") or {}
    if not isinstance(raw_recipe, dict):
        raise SpecError("recipe: must be an object")
    if "seed" not in raw_recipe:
        raise SpecError("recipe: missing required field 'seed'")
    try:
        seed = int(raw_recipe["seed"])
    except (TypeError, ValueError):
        raise SpecError("recipe: 'seed' must be an integer") from None

    recipe = Recipe(
        lora_rank=_clamp("lora_rank", raw_recipe.get("lora_rank", 16)),
        lora_alpha=_clamp("lora_alpha", raw_recipe.get("lora_alpha", 32)),
        lr=_clamp("lr", raw_recipe.get("lr", 2e-4)),
        epochs=_clamp("epochs", raw_recipe.get("epochs", 3)),
        max_seq_len=_clamp("max_seq_len", raw_recipe.get("max_seq_len", 1024)),
        batch_size=_clamp("batch_size", raw_recipe.get("batch_size", 4)),
        grad_accum=_clamp("grad_accum", raw_recipe.get("grad_accum", 4)),
        eval_split=_clamp("eval_split", raw_recipe.get("eval_split", 0.1)),
        seed=seed,
        load_in_4bit=_parse_bool_flag("load_in_4bit", raw_recipe.get("load_in_4bit", False)),
    )
    if kind == "decision_finetune":
        if not decision_repo or not decision_revision:
            raise SpecError("base_model: decision_finetune requires a pinned decision repo")

    artifact_urls = raw.get("artifact_urls") or {}
    if not isinstance(artifact_urls, dict):
        raise SpecError("spec: artifact_urls must be an object")
    required_artifacts = ("adapter", "report") if kind == "decision_finetune" else (
        "adapter", "gguf", "report")
    for key in required_artifacts:
        _req(artifact_urls, key, "artifact_urls")
    if kind == "decision_finetune" and "gguf" in artifact_urls:
        raise SpecError("artifact_urls: decision_finetune must not advertise a GGUF artifact")
    artifact_urls = {k: str(v) for k, v in artifact_urls.items()}
    for key, value in artifact_urls.items():
        _check_url(f"artifact_urls.{key}", value)

    run_id = str(_req(raw, "run_id", "spec"))
    # Validated here, not just in artifacts.paths(): parse_spec is the single gate
    # execute() cannot route a traversal-y run_id past — paths() is defense in depth.
    if not _RUN_ID_RE.fullmatch(run_id):
        raise SpecError("spec: run_id must be 1-64 chars of [A-Za-z0-9_-]")

    dataset_url = str(_req(raw, "dataset_url", "spec"))
    progress_url = str(_req(raw, "progress_url", "spec"))
    result_url = str(_req(raw, "result_url", "spec"))
    _check_url("dataset_url", dataset_url)
    _check_url("progress_url", progress_url)
    _check_url("result_url", result_url)

    autotune = _parse_autotune(raw.get("autotune"))
    runtime = _parse_runtime(raw.get("runtime"))
    checkpoint_urls = _parse_checkpoint_urls(raw.get("checkpoint_urls"))
    resume = _parse_resume(raw.get("resume"))
    if autotune is not None and (runtime is not None or checkpoint_urls is not None or resume is not None):
        raise SpecError("autotune cannot be combined with runtime, checkpoint_urls, or resume")
    if resume is not None and checkpoint_urls is None:
        raise SpecError("resume requires checkpoint_urls so progress can continue safely")
    if kind == "decision_finetune" and autotune is not None:
        raise SpecError("spec: decision_finetune does not support autotune yet")

    return JobSpec(
        run_id=run_id,
        kind=kind,
        hf_repo=hf_repo,
        hf_revision=hf_revision,
        decision_repo=decision_repo,
        decision_revision=decision_revision,
        unsloth_repo=unsloth_repo,
        unsloth_revision=unsloth_revision,
        unsloth_4bit_repo=unsloth_4bit_repo,
        unsloth_4bit_revision=unsloth_4bit_revision,
        dataset_url=dataset_url,
        quantize=quantize,
        progress_url=progress_url,
        result_url=result_url,
        token=str(_req(raw, "token", "spec")),
        artifact_urls=artifact_urls,
        recipe=recipe,
        autotune=autotune,
        runtime=runtime,
        checkpoint_urls=checkpoint_urls,
        resume=resume,
    )


def _pos_int(where: str, value: object, lo: int, hi: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise SpecError(f"autotune: '{where}' must be an integer") from None
    if n < lo or n > hi:
        raise SpecError(f"autotune: '{where}' must be between {lo} and {hi}")
    return n


def _parse_autotune(raw: object) -> Autotune | None:
    """Absent/null = a plain single run. Present = strict: a malformed sweep is
    a spec error (EXIT_SPEC, refunded), never silently downgraded to one trial,
    because the buyer paid for a sweep."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise SpecError("autotune: must be an object")
    trials_raw = raw.get("trials")
    if not isinstance(trials_raw, list) or not (1 <= len(trials_raw) <= MAX_TRIALS):
        raise SpecError(f"autotune: 'trials' must be a list of 1-{MAX_TRIALS} entries")
    trials = []
    for i, t in enumerate(trials_raw):
        if not isinstance(t, dict):
            raise SpecError(f"autotune: trial {i + 1} must be an object")
        trials.append(Trial(
            lr=_clamp("lr", t.get("lr", 2e-4)),
            lora_rank=_clamp("lora_rank", t.get("lora_rank", 16)),
            lora_alpha=_clamp("lora_alpha", t.get("lora_alpha", 32)),
            epochs=_clamp("epochs", t.get("epochs", 3)),
        ))
    experiment_url = str(_req(raw, "experiment_url", "autotune"))
    _check_url("autotune.experiment_url", experiment_url)
    return Autotune(
        trials=tuple(trials),
        budget_seconds=_pos_int("budget_seconds", raw.get("budget_seconds"), 600, 48 * 3600),
        trial_estimate_seconds=_pos_int("trial_estimate_seconds", raw.get("trial_estimate_seconds"), 1, 48 * 3600),
        reserve_seconds=_pos_int("reserve_seconds", raw.get("reserve_seconds"), 0, 48 * 3600),
        experiment_url=experiment_url,
    )


def _runtime_int(where: str, value: object, lo: int, hi: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise SpecError(f"runtime: '{where}' must be an integer") from None
    if n < lo or n > hi:
        raise SpecError(f"runtime: '{where}' must be between {lo} and {hi}")
    return n


def _parse_runtime(raw: object) -> RuntimeBudget | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise SpecError("runtime: must be an object")
    budget = _runtime_int("budget_seconds", raw.get("budget_seconds"), 600, 48 * 3600)
    reserve = _runtime_int(
        "finalization_reserve_seconds", raw.get("finalization_reserve_seconds"), 300, 12 * 3600)
    p95 = _runtime_int("estimated_p95_seconds", raw.get("estimated_p95_seconds"), 1, 7 * 24 * 3600)
    profile = str(raw.get("profile_version") or '')
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", profile):
        raise SpecError("runtime: 'profile_version' must be 1-64 safe characters")
    if reserve >= budget:
        raise SpecError("runtime: finalization reserve must be smaller than the budget")
    return RuntimeBudget(
        budget_seconds=budget,
        finalization_reserve_seconds=reserve,
        estimated_p95_seconds=p95,
        profile_version=profile,
    )


def _parse_checkpoint_urls(raw: object) -> dict[str, str] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise SpecError("checkpoint_urls: must be an object")
    result: dict[str, str] = {}
    for key in ("a", "b", "manifest"):
        value = str(_req(raw, key, "checkpoint_urls"))
        _check_url(f"checkpoint_urls.{key}", value)
        result[key] = value
    return result


def _parse_resume(raw: object) -> ResumeCheckpoint | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise SpecError("resume: must be an object")
    source = str(_req(raw, "source_run_id", "resume"))
    if not _RUN_ID_RE.fullmatch(source):
        raise SpecError("resume: source_run_id is invalid")
    archive_url = str(_req(raw, "archive_url", "resume"))
    _check_url("resume.archive_url", archive_url)
    digest = str(_req(raw, "archive_sha256", "resume"))
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise SpecError("resume: archive_sha256 must be lowercase sha256")
    dataset_digest = str(_req(raw, "dataset_sha256", "resume"))
    recipe_digest = str(_req(raw, "recipe_sha256", "resume"))
    if not re.fullmatch(r"[a-f0-9]{64}", dataset_digest):
        raise SpecError("resume: dataset_sha256 must be lowercase sha256")
    if not re.fullmatch(r"[a-f0-9]{64}", recipe_digest):
        raise SpecError("resume: recipe_sha256 must be lowercase sha256")
    model_id = str(_req(raw, "model_id", "resume"))
    if not model_id or len(model_id) > 512:
        raise SpecError("resume: model_id is invalid")
    slot = str(_req(raw, "slot", "resume"))
    if slot not in {"a", "b"}:
        raise SpecError("resume: slot must be a or b")
    step = _runtime_int("resume.step", raw.get("step"), 0, 10_000_000)
    epoch = _runtime_int("resume.epoch", raw.get("epoch"), 0, 10_000)
    planned_steps = _runtime_int("resume.planned_steps", raw.get("planned_steps"), 1, 10_000_000)
    if step > planned_steps:
        raise SpecError("resume: step cannot exceed planned_steps")
    return ResumeCheckpoint(
        source_run_id=source,
        archive_url=archive_url,
        archive_sha256=digest,
        archive_bytes=_runtime_int("resume.archive_bytes", raw.get("archive_bytes"), 1, 20 * 1024**3),
        step=step,
        epoch=epoch,
        planned_steps=planned_steps,
        slot=slot,
        model_id=model_id,
        dataset_sha256=dataset_digest,
        recipe_sha256=recipe_digest,
    )
