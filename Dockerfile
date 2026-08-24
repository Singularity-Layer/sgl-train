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

# convert_hf_to_gguf.py is pure Python and lives in the llama.cpp tree; llama-quantize
# is a compiled binary we take from the prebuilt release rather than compiling here.
RUN git clone --depth 1 --branch ${LLAMA_CPP_TAG} \
      https://github.com/ggml-org/llama.cpp /opt/llama.cpp \
 && pip install --no-cache-dir -r /opt/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

RUN mkdir -p /opt/sgl/bin /opt/sgl/work \
 && curl -fsSL -o /tmp/llama.tgz \
      "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_CPP_TAG}/llama-${LLAMA_CPP_TAG}-bin-ubuntu-vulkan-x64.tar.gz" \
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
