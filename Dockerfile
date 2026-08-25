# Managed training image for Singularity Layer.
#
# Inference machines use ghcr.io/ggml-org/llama.cpp:server-cuda-* — that image
# CANNOT fine-tune. This is the training counterpart. Keep LLAMA_CPP_TAG in sync
# with SGLCompute_Backend/src/lib/ai-machine.ts and the orchestrator's machines.ts.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ARG LLAMA_CPP_TAG=b9867
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/sgl/hf

RUN apt-get update -y \
 && apt-get install -y --no-install-recommends git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      "transformers==4.46.3" \
      "peft==0.13.2" \
      "trl==0.12.1" \
      "datasets==3.1.0" \
      "accelerate==1.1.1" \
      "sentencepiece==0.2.0" \
      "protobuf==5.28.3" \
      "gguf==0.10.0"

# --- Unsloth engine (optional) -----------------------------------------------
# trainer.py TRIES this at runtime (SGL_ENGINE=auto, the default) and falls
# back to the vanilla stack above on ANY import/load/wrap failure, so this
# block is not fatal to the image's core purpose if the pin below goes stale.
#
# PROVISIONAL pin, verified against PyPI metadata (not yet build-tested in
# CI — that build is the real verifier; if `import unsloth` fails inside the
# built image, fix this block, do NOT bump torch to chase a newer unsloth):
#
#   unsloth==2025.3.18 is the newest release whose BASE (non-extra) deps still
#   declare `transformers!=4.47.0,>=4.46.1`, `peft!=0.11.0,>=0.7.1`,
#   `datasets>=2.16.0`, `trl>=0.7.9,<=0.15.2`, `accelerate>=0.34.1` — all
#   satisfied by the exact pins above. From unsloth==2025.5.1 onward the
#   requirement jumps to `transformers==4.51.3` (exact) and later
#   `datasets>=3.4.1`, which would force-upgrade the whole vanilla stack this
#   Dockerfile has already pinned and tested; staying on 2025.3.18 keeps the
#   vanilla engine's pins untouched, matching this file's existing "never let
#   the converter's deps clobber the main env" philosophy (see convert-venv
#   below) — same reasoning, applied to unsloth instead of llama.cpp's convert
#   script.
#
#   unsloth's own `cu124-torch251` extra (an official, versioned install
#   target for exactly this torch+CUDA combo) resolves xformers==0.0.29.post1,
#   which itself pins `torch==2.5.1` exactly — independent confirmation this
#   unsloth vintage targets our exact base image's torch build.
#
#   Installed with --no-deps: at this vintage, unsloth AND unsloth_zoo both
#   still declare `protobuf<4.0.0`, which conflicts with protobuf==5.28.3
#   pinned above for the main training stack (a real requirement, not just an
#   unsloth artifact — see convert.py's docstring). --no-deps skips that
#   resolver conflict; every other package unsloth/unsloth_zoo actually import
#   at runtime is installed explicitly below instead of left to pip's resolver.
RUN pip install --no-cache-dir --no-deps \
      "unsloth==2025.3.18" \
      "unsloth_zoo==2025.3.14" \
 && pip install --no-cache-dir \
      "xformers==0.0.29.post1" \
      "bitsandbytes==0.45.3" \
      "torchvision==0.20.1" \
      "tyro" \
      "hf_transfer" \
      "diffusers" \
      "cut-cross-entropy==25.1.1" \
      "pillow" \
      "regex" \
      "psutil"

# convert_hf_to_gguf.py is pure Python and lives in the llama.cpp tree; llama-quantize
# is a compiled binary we take from the prebuilt release rather than compiling here.
RUN git clone --depth 1 --branch ${LLAMA_CPP_TAG} \
      https://github.com/ggml-org/llama.cpp /opt/llama.cpp

# Converter deps live in their OWN venv: the requirements file pins a CPU-only
# torch + different transformers/protobuf and would CLOBBER the CUDA training
# stack if installed into the main environment (verified against tag b9867).
RUN python -m venv --without-pip /opt/sgl/convert-venv \
 && curl -fsSL https://bootstrap.pypa.io/get-pip.py | /opt/sgl/convert-venv/bin/python \
 && /opt/sgl/convert-venv/bin/pip install --no-cache-dir \
      -r /opt/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

RUN mkdir -p /opt/sgl/bin /opt/sgl/work \
 && curl -fsSL -o /tmp/llama.tgz \
      "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_CPP_TAG}/llama-${LLAMA_CPP_TAG}-bin-ubuntu-x64.tar.gz" \
 && tar xzf /tmp/llama.tgz -C /tmp \
 && QBIN="$(find /tmp -type f -name llama-quantize | head -1)" && test -n "$QBIN" \
 && cp -a "$(dirname "$QBIN")"/. /opt/sgl/bin/ \
 && chmod +x /opt/sgl/bin/llama-quantize \
 && rm -rf /tmp/llama.tgz

ENV LD_LIBRARY_PATH=/opt/sgl/bin

WORKDIR /opt/sgl/app
COPY runner ./runner
COPY pyproject.toml ./

# The spec arrives as SGL_JOB_SPEC. No secrets are baked into this image.
ENTRYPOINT ["python", "-m", "runner.main"]
