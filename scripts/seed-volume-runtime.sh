#!/usr/bin/env bash
set -euo pipefail

# Populate a versioned runtime atomically. READY.json is written last, so paid
# Pods reject a partial install even when this container is interrupted.
version="${SGL_RUNTIME_VERSION:-v0.6.0}"
dest="${SGL_RUNTIME_ROOT:-/workspace/sgl-train/runtime/${version}}"
stage="${dest}.staging.$$"
llama_tag="${LLAMA_CPP_TAG:-b9867}"

test -d /workspace || { echo "[seed] /workspace network volume is not mounted" >&2; exit 1; }
test ! -e "$dest" || { echo "[seed] ${dest} already exists; refusing to overwrite" >&2; exit 1; }
mkdir -p "$stage" /workspace/sgl-train/cache/huggingface /workspace/sgl-train/cache/torch
cleanup_failed_stage() {
  echo "[seed] failed before READY marker" >&2
  case "$stage" in
    "$dest".staging.*)
      chmod -R u+w "$stage" 2>/dev/null || true
      rm -rf -- "$stage"
      ;;
    *) echo "[seed] refusing unsafe cleanup path" >&2 ;;
  esac
}
trap cleanup_failed_stage ERR

echo "[seed] creating Python 3.10 runtime"
python3.10 -m venv "$stage/venv"
runtime_python="$stage/venv/bin/python"
runtime_pip="$stage/venv/bin/pip"
"$runtime_pip" install --upgrade "pip==25.2" "setuptools==80.9.0" "wheel==0.45.1"

echo "[seed] installing pinned PyTorch CUDA 12.4 stack"
"$runtime_pip" install --index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.5.1" "torchvision==0.20.1"

echo "[seed] installing pinned training dependencies"
"$runtime_pip" install \
  "transformers==4.49.0" \
  "peft==0.14.0" \
  "trl==0.15.2" \
  "datasets==3.1.0" \
  "accelerate==1.4.0" \
  "sentencepiece==0.2.0" \
  "protobuf==5.28.3" \
  "gguf==0.10.0"

echo "[seed] installing pinned LoRA and QLoRA acceleration stack"
"$runtime_pip" install --no-deps \
  "unsloth==2025.3.18" \
  "unsloth_zoo==2025.3.14"
"$runtime_pip" install \
  "xformers==0.0.29.post1" \
  "bitsandbytes==0.45.3" \
  "tyro==0.9.14" \
  "hf_transfer==0.1.9" \
  "diffusers==0.32.2" \
  "cut-cross-entropy==25.1.1" \
  "pillow==11.1.0" \
  "regex==2024.11.6" \
  "psutil==6.1.1"

echo "[seed] verifying dependency closure"
"$runtime_python" - <<'PY'
import importlib.util as u
import torch, torchvision, transformers, peft, trl, datasets, accelerate
import xformers, bitsandbytes, cut_cross_entropy, diffusers, tyro
from transformers import GenerationConfig, CompileConfig, HybridCache
assert torch.__version__.split('+')[0] == '2.5.1', torch.__version__
assert str(torch.version.cuda).startswith('12.4'), torch.version.cuda
assert u.find_spec('unsloth'), 'unsloth not installed'
assert u.find_spec('unsloth_zoo'), 'unsloth_zoo not installed'
PY

echo "[seed] installing pinned GGUF conversion tools"
git clone --depth 1 --branch "$llama_tag" https://github.com/ggml-org/llama.cpp "$stage/llama.cpp"
python3.10 -m venv "$stage/convert-venv"
"$stage/convert-venv/bin/pip" install --upgrade "pip==25.2"
"$stage/convert-venv/bin/pip" install \
  -r "$stage/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt"

mkdir -p "$stage/bin"
curl -fsSL -o /tmp/llama.tgz \
  "https://github.com/ggml-org/llama.cpp/releases/download/${llama_tag}/llama-${llama_tag}-bin-ubuntu-x64.tar.gz"
tar xzf /tmp/llama.tgz -C /tmp
quantize_bin="$(find /tmp -type f -name llama-quantize | head -1)"
test -n "$quantize_bin"
cp -a "$(dirname "$quantize_bin")"/. "$stage/bin"/
chmod +x "$stage/bin/llama-quantize"

# Existing bootstrap expects this stable path. The symlink points at the pinned
# venv inside the same immutable runtime directory.
ln -s "venv/lib/python3.10/site-packages" "$stage/site-packages"
printf '{"schema":1,"version":"%s"}\n' "$version" > "$stage/READY.json"
chmod -R a-w "$stage"
mv "$stage" "$dest"
chmod 1777 /workspace/sgl-train/cache/huggingface /workspace/sgl-train/cache/torch
trap - ERR
echo "[seed] complete"
du -sh "$dest"
