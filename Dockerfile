# Tiny managed-training bootstrap. The GPU runtime is seeded once onto the
# configured Runpod network volume; paid Pods pull only this small private
# control image and execute against the immutable cached runtime.
FROM python:3.10-slim-bookworm@sha256:db6ee5170ebce0f4d165af242e8e77addc08bd9ffc37e8b1815752cf027aa4f8 AS libstdcpp

# bitsandbytes 0.45.3 was built with GCC 13 and requires GLIBCXX_3.4.32.
# Debian bookworm carries GCC 12, so extract the exact conda-forge runtime
# instead of upgrading the base OS or compiling unpinned native code.
RUN apt-get update -y \
 && apt-get install -y --no-install-recommends ca-certificates curl zstd \
 && curl -fsSL -o /tmp/libstdcxx.conda https://conda.anaconda.org/conda-forge/linux-64/libstdcxx-ng-13.2.0-hc0a3c3a_13.conda \
 && echo '143171c6084e526122cc2976fbbfadf7b9f50e5d13036adf20feb7ed9d036dd2  /tmp/libstdcxx.conda' | sha256sum -c - \
 && curl -fsSL -o /tmp/libgcc.conda https://conda.anaconda.org/conda-forge/linux-64/libgcc-ng-13.2.0-h77fa898_13.conda \
 && echo 'ffa0f472c8b37f864de855af2d3c057f1813162319f10ebd97332d73fc27ba60  /tmp/libgcc.conda' | sha256sum -c - \
 && python -c "import zipfile; zipfile.ZipFile('/tmp/libstdcxx.conda').extractall('/tmp/libstdcxx'); zipfile.ZipFile('/tmp/libgcc.conda').extractall('/tmp/libgcc')" \
 && mkdir -p /opt/sgl/conda \
 && tar --zstd -xf "$(find /tmp/libstdcxx -name 'pkg-*.tar.zst' -print -quit)" -C /opt/sgl/conda \
 && tar --zstd -xf "$(find /tmp/libgcc -name 'pkg-*.tar.zst' -print -quit)" -C /opt/sgl/conda \
 && grep -aq 'GLIBCXX_3.4.32' /opt/sgl/conda/lib/libstdc++.so.6

FROM python:3.10-slim-bookworm@sha256:db6ee5170ebce0f4d165af242e8e77addc08bd9ffc37e8b1815752cf027aa4f8

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    SGL_RUNTIME_VERSION=v0.6.0 \
    SGL_RUNTIME_ROOT=/workspace/sgl-train/runtime/v0.6.0 \
    SGL_WORKDIR=/tmp/sgl/work \
    HF_HOME=/workspace/sgl-train/cache/huggingface \
    TORCH_HOME=/workspace/sgl-train/cache/torch \
    UNSLOTH_DISABLE_AUTO_UPDATES=1 \
    LD_LIBRARY_PATH=/opt/sgl/conda/lib \
    CC=gcc \
    CXX=g++

COPY --from=libstdcpp /opt/sgl/conda /opt/sgl/conda

RUN apt-get update -y \
 && apt-get install -y --no-install-recommends ca-certificates gcc g++ libgomp1 \
 && ln -sf /usr/local/bin/python3.10 /usr/bin/python3.10 \
 && gcc --version \
 && g++ --version \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/sgl/app
COPY runner ./runner
COPY pyproject.toml ./

# SGL_JOB_SPEC contains the per-run callback token and arrives only at runtime.
# The bootstrap validates the volume marker before importing any heavy package.
ENTRYPOINT ["python3.10", "-m", "runner.bootstrap"]
