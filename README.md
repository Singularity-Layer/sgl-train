# sgl-train image

Turns (pinned base model + JSONL dataset) into (LoRA adapter + merged weights +
quantized GGUF + eval report).

## Tests (no GPU needed)

```bash
pip install -e ".[dev]"
python -m pytest -v
```

## Run it by hand

```bash
docker run --gpus all \
  -e SGL_JOB_SPEC="$(cat spec.json)" \
  ghcr.io/singularity-layer/sgl-train:v0.1.0
```

`spec.json` shape: see `runner/spec.py`. `base_model.revision` is required — an
unpinned model cannot be reproduced, and an unreproducible run cannot be verified.

## Engines

Two training engines share the same load → baseline_eval → wrap-with-LoRA →
train → eval → save pipeline, so baseline/final eval numbers are comparable
regardless of which one ran:

- **vanilla** — plain `transformers` + `peft`. Always available; no 4-bit.
- **unsloth** — faster kernels, and the only engine that supports QLoRA
  (`recipe.load_in_4bit`).

`SGL_ENGINE` selects which one runs (default `auto`):

| `SGL_ENGINE` | behavior |
|---|---|
| `auto` (default) | try loading + wrapping via unsloth; ANY failure (import, model load, LoRA wrap) falls back to the vanilla path automatically. The engine is locked in once training starts — no switching mid-run. |
| `vanilla` | skip the unsloth attempt entirely. |
| `unsloth` | force unsloth; a failure raises instead of falling back (testing mode). |

The report includes `engine` (`"unsloth"` or `"vanilla"`) and, when a fallback
happened, `engine_fallback_reason` (the exception class name only, or the
literal string `"NoUnslothPin"` — see below — never buyer-facing content). It
also includes `trained_repo`/`trained_revision`: the repo+revision that
*actually* trained, which for the unsloth engine is the pinned mirror below,
not `base_model`/`base_revision`.

### The unsloth mirror pin (`base_model.unsloth_repo` / `unsloth_revision`)

unsloth's `FastLanguageModel.from_pretrained` silently **remaps** a canonical
repo name (e.g. `Qwen/Qwen2.5-0.5B-Instruct`) to its own mirror namespace
(e.g. `unsloth/Qwen2.5-0.5B-Instruct`) — but it still applies whatever
`revision` it was given, and that revision is the *canonical* repo's sha, not
the mirror's. The mirror almost never has a commit at that exact sha, so the
load fails (`AutoConfig error: Unrecognized model in
unsloth/Qwen2.5-0.5B-Instruct`) — or, worse, could silently resolve to
*whatever the mirror's default branch happens to be*, which is not
reproducible and breaks this image's honesty guarantee.

To use the unsloth engine, a spec must pin the mirror explicitly with two
**optional, both-or-neither** `base_model` fields:

```json
"base_model": {
  "hf_repo": "Qwen/Qwen2.5-0.5B-Instruct",
  "revision": "<canonical repo sha>",
  "unsloth_repo": "unsloth/Qwen2.5-0.5B-Instruct",
  "unsloth_revision": "<the MIRROR repo's own sha>"
}
```

Supplying only one of the pair is a `SpecError`. If neither is supplied, the
unsloth engine is simply never attempted:

- `SGL_ENGINE=auto` with no mirror pin → skips straight to vanilla, no
  unsloth import at all, `engine_fallback_reason: "NoUnslothPin"`.
- `SGL_ENGINE=unsloth` (forced) with no mirror pin → `InfraError` → exit `6`
  (refundable: this is our capability gap, not the buyer's).

### QLoRA (`recipe.load_in_4bit`)

`recipe.load_in_4bit` (default `false`) requests 4-bit (QLoRA) training. It's
strict-parsed: only JSON `true`/`false` are accepted, anything else is a
`SpecError`. The vanilla engine cannot do 4-bit — if `load_in_4bit` is `true`
and the run resolves to vanilla (unsloth unavailable in `auto`, or `vanilla`
forced explicitly), the run raises `InfraError` and exits `6` (refundable:
this is a capability gap on our box, never the buyer's fault).

## Exit codes

| code | meaning | refundable? |
|---|---|---|
| 0 | completed | no |
| 2 | bad spec | yes, our fault |
| 3 | dataset unusable | no, but warn before charging |
| 4 | training failed (OOM etc.) | no, compute was delivered |
| 5 | artifact upload failed | yes, no artifact delivered |
| 6 | infrastructure failure (fetch/model download/no CUDA) | yes, our fault |

## Environment contracts

- `HF_TOKEN` — the backend may inject this env var for gated base models;
  `huggingface_hub` reads it automatically, so no extra plumbing is needed here.
- `SGL_ALLOW_CPU=1` — a local-testing override that lets `run_finetune` proceed
  without a GPU. Never set this in production; without it, a box with no CUDA
  raises `InfraError` (exit 6) instead of silently training on CPU.
- `SGL_ENGINE` — `auto` (default) / `vanilla` / `unsloth`. See "Engines" above.
