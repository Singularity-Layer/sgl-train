#!/usr/bin/env bash
set -euo pipefail

# Run once inside the full v0.4.0 image with the 50 GB network volume mounted
# at /workspace. It copies only immutable runtime files; buyer data is never
# written to the shared volume.
version="${SGL_RUNTIME_VERSION:-v0.5.0}"
dest="${SGL_RUNTIME_ROOT:-/workspace/sgl-train/runtime/${version}}"
stage="${dest}.staging.$$"
python_bin="${PYTHON_BIN:-python3.10}"

test "$(id -u)" = "0" || { echo "seed must run as root" >&2; exit 1; }
test -d /workspace || { echo "/workspace volume is not mounted" >&2; exit 1; }
test ! -e "$dest" || { echo "$dest already exists; refusing to overwrite" >&2; exit 1; }

purelib="$($python_bin -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
platlib="$($python_bin -c 'import sysconfig; print(sysconfig.get_paths()["platlib"])')"
mkdir -p "$stage/site-packages" "$stage/bin"
cp -a "$purelib"/. "$stage/site-packages"/
if [[ "$platlib" != "$purelib" ]]; then cp -a "$platlib"/. "$stage/site-packages"/; fi
cp -a /opt/llama.cpp "$stage/llama.cpp"
cp -a /opt/sgl/convert-venv "$stage/convert-venv"
cp -a /opt/sgl/bin/. "$stage/bin"/

cat > "$stage/READY.json" <<EOF
{"schema":1,"version":"${version}"}
EOF
chmod -R a-w "$stage"
mv "$stage" "$dest"
mkdir -p /workspace/sgl-train/cache/huggingface /workspace/sgl-train/cache/torch
chmod 1777 /workspace/sgl-train/cache/huggingface /workspace/sgl-train/cache/torch
du -sh "$dest"
echo "seeded ${dest}"
