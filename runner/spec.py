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

KINDS = {"finetune", "scratch", "autoresearch"}
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
class JobSpec:
    run_id: str
    kind: str
    hf_repo: str
    hf_revision: str
    unsloth_repo: str | None
    unsloth_revision: str | None
    dataset_url: str
    quantize: str
    progress_url: str
    result_url: str
    token: str
    artifact_urls: dict[str, str]
    recipe: Recipe


def _req(d: dict, key: str, where: str) -> object:
    if key not in d or d[key] in (None, ""):
        raise SpecError(f"{where}: missing required field '{key}'")
    return d[key]


def _optional_pair(d: dict, key_a: str, key_b: str, where: str) -> tuple[str | None, str | None]:
    """Both-or-neither optional string pair, e.g. unsloth_repo/unsloth_revision.

    Presence requires each field, when present, to be an actual str whose
    .strip() is non-empty. Missing key, None, empty/whitespace string, or
    non-str type all count as absent. If exactly one of the pair is present
    the spec is unusable — a repo with no pinned revision (or vice versa)
    is not reproducible — so that raises SpecError rather than silently
    picking a default.

    Stored values are stripped of leading/trailing whitespace.
    """
    def is_valid_string(v: object) -> bool:
        return isinstance(v, str) and v.strip() != ""

    a_present = is_valid_string(d.get(key_a))
    b_present = is_valid_string(d.get(key_b))
    if a_present != b_present:
        raise SpecError(f"{where}: '{key_a}' and '{key_b}' must both be present or both be absent")
    if not a_present:
        return None, None
    # Both are present and valid strings; strip and return.
    a_val = d[key_a]  # type: ignore  # guarded by is_valid_string above
    b_val = d[key_b]  # type: ignore
    if not isinstance(a_val, str) or not isinstance(b_val, str):
        raise SpecError(f"{where}: '{key_a}' and '{key_b}' must be strings")
    return a_val.strip(), b_val.strip()


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

    artifact_urls = raw.get("artifact_urls") or {}
    if not isinstance(artifact_urls, dict):
        raise SpecError("spec: artifact_urls must be an object")
    for key in ("adapter", "gguf", "report"):
        _req(artifact_urls, key, "artifact_urls")
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

    return JobSpec(
        run_id=run_id,
        kind=kind,
        hf_repo=hf_repo,
        hf_revision=hf_revision,
        unsloth_repo=unsloth_repo,
        unsloth_revision=unsloth_revision,
        dataset_url=dataset_url,
        quantize=quantize,
        progress_url=progress_url,
        result_url=result_url,
        token=str(_req(raw, "token", "spec")),
        artifact_urls=artifact_urls,
        recipe=recipe,
    )
