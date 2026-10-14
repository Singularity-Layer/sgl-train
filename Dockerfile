# Tiny managed-training bootstrap. The GPU runtime is seeded once onto the
# configured Runpod network volume; paid Pods pull only this small private
# control image and execute against the immutable cached runtime.
FROM python:3.10-slim-bookworm@sha256:db6ee5170ebce0f4d165af242e8e77addc08bd9ffc37e8b1815752cf027aa4f8

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    SGL_RUNTIME_VERSION=v0.5.0 \
    SGL_RUNTIME_ROOT=/workspace/sgl-train/runtime/v0.5.0 \
    SGL_WORKDIR=/tmp/sgl/work \
    HF_HOME=/workspace/sgl-train/cache/huggingface \
    TORCH_HOME=/workspace/sgl-train/cache/torch \
    UNSLOTH_DISABLE_AUTO_UPDATES=1

RUN apt-get update -y \
 && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
 && ln -sf /usr/local/bin/python3.10 /usr/bin/python3.10 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/sgl/app
COPY runner ./runner
COPY pyproject.toml ./

# SGL_JOB_SPEC contains the per-run callback token and arrives only at runtime.
# The bootstrap validates the volume marker before importing any heavy package.
ENTRYPOINT ["python3.10", "-m", "runner.bootstrap"]
