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
