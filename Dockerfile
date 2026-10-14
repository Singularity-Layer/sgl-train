# Managed training image for Singularity Layer.
#
# Inference machines use ghcr.io/ggml-org/llama.cpp:server-cuda-* — that image
# CANNOT fine-tune. This is the training counterpart. Keep LLAMA_CPP_TAG in sync
# with SGLCompute_Backend/src/lib/ai-machine.ts and the orchestrator's machines.ts.
# Use RunPod's own pinned PyTorch 2.5.1 / CUDA 12.4.1 image. RunPod hosts commonly
# carry these official template layers already, so a cold pod downloads only
# our private runner/dependency layers instead of a monolithic 3.15 GB PyTorch
# layer from GHCR. The devel toolchain also preserves Triton/Unsloth JIT support.
FROM runpod/pytorch:0.7.0-cu1241-torch251-ubuntu2204@sha256:3bdc5341e5261bb4fbe009498eb437717ab53e92bdbd13e957c5927bb8fca910

ARG LLAMA_CPP_TAG=b9867
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/sgl/hf

RUN apt-get update -y \
 && apt-get install -y --no-install-recommends git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --no-cache-dir \
      "transformers==4.49.0" \
      "peft==0.14.0" \
      "trl==0.15.2" \
      "datasets==3.1.0" \
      "accelerate==1.4.0" \
      "sentencepiece==0.2.0" \
      "protobuf==5.28.3" \
      "gguf==0.10.0" \
      "runpod==1.12.0"

# --- Unsloth engine (optional) -----------------------------------------------
# trainer.py TRIES this at runtime (SGL_ENGINE=auto, the default) and falls
# back to the vanilla stack above on ANY import/load/wrap failure, so this
# block is not fatal to the image's core purpose if the pin below goes stale.
#
# PROVISIONAL pin, verified against PyPI metadata. CI does NOT `import unsloth`
# itself — unsloth GPU-gates at import time and CI runners have no GPU, so that
# import would fail even on a correctly-pinned image. What CI DOES verify at
# build time is the dependency closure below (import of everything unsloth
# actually needs, none of which GPU-gate); the real end-to-end gate is the
# forced SGL_ENGINE=unsloth smoke step run on real GPU hardware. If that smoke
# step fails, fix this block, do NOT bump torch to chase a newer unsloth):
#
#   LIVE-SMOKE CORRECTION (2026-08-26): unsloth's DECLARED metadata
#   (`transformers>=4.46.1`) understates what its code actually imports —
#   unsloth/models/vision.py does `from transformers import GenerationConfig,
#   CompileConfig, HybridCache`, and CompileConfig only exists from
#   transformers 4.48. On transformers==4.46.3 the import raised ImportError,
#   the runner fell back to vanilla, and the run still completed (fallback
#   working as designed). Main stack therefore moved to the 4.49/0.14/0.15.2/
#   1.4.0 set, contemporaneous with this unsloth vintage. The build now
#   asserts that exact import line, so this can never regress silently.
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
RUN python3.10 -m pip install --no-cache-dir --no-deps \
      "unsloth==2025.3.18" \
      "unsloth_zoo==2025.3.14" \
 && python3.10 -m pip install --no-cache-dir \
      "xformers==0.0.29.post1" \
      "bitsandbytes==0.45.3" \
      "torchvision==0.20.1" \
      "tyro==0.9.14" \
      "hf_transfer==0.1.9" \
      "diffusers==0.32.2" \
      "cut-cross-entropy==25.1.1" \
      "pillow==11.1.0" \
      "regex==2024.11.6" \
      "psutil==6.1.1"
# ^ tyro/hf_transfer/diffusers/pillow/regex/psutil: PROVISIONAL pins, same
#   reasoning and vintage as the unsloth pin above (versions current as of
#   unsloth==2025.3.18's release window) — pinned so a bare `pip install`
#   can't silently drift a transitive (e.g. a newer diffusers pulling a
#   newer huggingface_hub) out from under the pins this file already tested.

# CPU-safe closure check: proves every non-unsloth package unsloth/unsloth_zoo
# import at runtime is actually importable in THIS image, before the smoke
# step ever touches a GPU. Deliberately NOT `import unsloth` itself — unsloth
# GPU-gates at import time, so that would fail in CI regardless of whether the
# pins above are correct. The real runtime gate for unsloth itself is the
# forced SGL_ENGINE=unsloth smoke step (GPU hardware, not this build).
RUN python3.10 -c "import importlib.util as u; \
    assert u.find_spec('unsloth'), 'unsloth not installed'; \
    assert u.find_spec('unsloth_zoo'), 'unsloth_zoo not installed'; \
    import xformers, bitsandbytes, cut_cross_entropy, diffusers, tyro; \
    from transformers import GenerationConfig, CompileConfig, HybridCache"

# Fail the image build if the official base ever drifts behind its tag.
RUN python3.10 -c "import sys, torch; \
    assert sys.version_info >= (3, 10); \
    assert torch.__version__.split('+')[0] == '2.5.1', torch.__version__; \
    assert str(torch.version.cuda).startswith('12.4'), torch.version.cuda"

# convert_hf_to_gguf.py is pure Python and lives in the llama.cpp tree; llama-quantize
# is a compiled binary we take from the prebuilt release rather than compiling here.
RUN git clone --depth 1 --branch ${LLAMA_CPP_TAG} \
      https://github.com/ggml-org/llama.cpp /opt/llama.cpp

# Converter deps live in their OWN venv: the requirements file pins a CPU-only
# torch + different transformers/protobuf and would CLOBBER the CUDA training
# stack if installed into the main environment (verified against tag b9867).
RUN python3.10 -m venv --without-pip /opt/sgl/convert-venv \
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

# unsloth AUTO-UPDATES unsloth_zoo over the network on every `import unsloth`
# unless this is set — observed live: a pinned 2025.3.14 silently became
# 2026.8.15 inside a running pod. That makes the pin a fiction, adds a
# network dependency to every paid run, and is an unreviewed-code-at-runtime
# supply-chain hole. Must be set BEFORE the import; a Docker ENV guarantees it.
ENV UNSLOTH_DISABLE_AUTO_UPDATES=1

ENV LD_LIBRARY_PATH=/opt/sgl/bin

WORKDIR /opt/sgl/app
COPY runner ./runner
COPY pyproject.toml ./

# The spec arrives as SGL_JOB_SPEC. No secrets are baked into this image.
ENTRYPOINT ["python3.10", "-m", "runner.serverless"]
