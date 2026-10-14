"""HF safetensors → GGUF → quantized GGUF, via llama.cpp tooling.

Two steps on purpose: convert_hf_to_gguf.py only emits F16/F32, and llama-quantize
only consumes a GGUF. The F16 intermediate is large and is deleted by the caller.

Failure here is NOT fatal to the run: the buyer still gets the adapter and merged
weights. It only disables deploy-to-grid, which needs GGUF.
"""
from __future__ import annotations

import os
import subprocess

RUNTIME_ROOT = os.environ.get("SGL_RUNTIME_ROOT", "/opt/sgl")
LLAMA_ROOT = os.environ.get("SGL_LLAMA_ROOT", "/opt/llama.cpp")
CONVERT_ROOT = os.environ.get("SGL_CONVERT_ROOT", "/opt/sgl/convert-venv")
BIN_ROOT = os.environ.get("SGL_BIN_ROOT", "/opt/sgl/bin")
CONVERT_SCRIPT = f"{LLAMA_ROOT}/convert_hf_to_gguf.py"
QUANTIZE_BIN = f"{BIN_ROOT}/llama-quantize"
# The converter's requirements pin a different torch/transformers/protobuf than
# the training stack, so it lives in its own venv (see Dockerfile) rather than
# the main interpreter this process runs under.
CONVERT_PY = f"{CONVERT_ROOT}/bin/python"


class ConvertError(RuntimeError):
    pass


def convert_commands(merged_dir: str, out_gguf: str, quantize: str) -> list[list[str]]:
    f16 = out_gguf.replace(".gguf", ".f16.gguf")
    return [
        [CONVERT_PY, CONVERT_SCRIPT, merged_dir, "--outfile", f16, "--outtype", "f16"],
        [QUANTIZE_BIN, f16, out_gguf, quantize],
    ]


def _subprocess_run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return p.returncode, (p.stderr or "")[-2000:]


def to_gguf(merged_dir: str, out_gguf: str, quantize: str, run=None) -> None:
    runner = run or _subprocess_run
    cmds = convert_commands(merged_dir, out_gguf, quantize)
    for cmd in cmds:
        res = runner(cmd)
        code, err = res if isinstance(res, tuple) else (res, "")
        if code != 0:
            step = "convert" if "convert_hf_to_gguf.py" in cmd[1] else "quantize"
            raise ConvertError(f"{step} failed with exit code {code}: {err}".rstrip(': '))
    f16 = cmds[1][1]
    if os.path.exists(f16):
        os.remove(f16)
