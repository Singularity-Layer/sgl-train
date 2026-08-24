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
