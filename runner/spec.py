"""Job spec parsing. Torch-free by design — this module must import on a laptop.

The runner re-applies every clamp the backend applies. Two independent checks of
the same bound is deliberate: the backend is the policy, this is the fail-safe.
"""
from __future__ import annotations

from dataclasses import dataclass

KINDS = {"finetune", "scratch", "autoresearch"}
QUANTIZE = {"Q4_K_M", "Q5_K_M", "Q8_0"}

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


@dataclass(frozen=True)
class JobSpec:
    run_id: str
    kind: str
    hf_repo: str
    hf_revision: str
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
    )

    artifact_urls = raw.get("artifact_urls") or {}
    if not isinstance(artifact_urls, dict):
        raise SpecError("spec: artifact_urls must be an object")
    for key in ("adapter", "gguf", "report"):
        _req(artifact_urls, key, "artifact_urls")

    return JobSpec(
        run_id=str(_req(raw, "run_id", "spec")),
        kind=kind,
        hf_repo=hf_repo,
        hf_revision=hf_revision,
        dataset_url=str(_req(raw, "dataset_url", "spec")),
        quantize=quantize,
        progress_url=str(_req(raw, "progress_url", "spec")),
        result_url=str(_req(raw, "result_url", "spec")),
        token=str(_req(raw, "token", "spec")),
        artifact_urls={k: str(v) for k, v in artifact_urls.items()},
        recipe=recipe,
    )
