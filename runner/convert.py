"""HF safetensors → GGUF → quantized GGUF, via llama.cpp tooling.

Two steps on purpose: convert_hf_to_gguf.py only emits F16/F32, and llama-quantize
only consumes a GGUF. The F16 intermediate is large and is deleted by the caller.

Failure here is NOT fatal to the run: the buyer still gets the adapter and merged
weights. It only disables deploy-to-grid, which needs GGUF.
"""
from __future__ import annotations

import os
import subprocess

CONVERT_SCRIPT = "/opt/llama.cpp/convert_hf_to_gguf.py"
QUANTIZE_BIN = "/opt/sgl/bin/llama-quantize"


class ConvertError(RuntimeError):
    pass


def convert_commands(merged_dir: str, out_gguf: str, quantize: str) -> list[list[str]]:
    f16 = out_gguf.replace(".gguf", ".f16.gguf")
    return [
        ["python", CONVERT_SCRIPT, merged_dir, "--outfile", f16, "--outtype", "f16"],
        [QUANTIZE_BIN, f16, out_gguf, quantize],
    ]


def _subprocess_run(cmd: list[str]) -> int:
    return subprocess.run(cmd, check=False).returncode


def to_gguf(merged_dir: str, out_gguf: str, quantize: str, run=None) -> None:
    runner = run or _subprocess_run
    cmds = convert_commands(merged_dir, out_gguf, quantize)
    for cmd in cmds:
        code = runner(cmd)
        if code != 0:
            step = "convert" if "convert_hf_to_gguf.py" in cmd[1] else "quantize"
            raise ConvertError(f"{step} failed with exit code {code}")
    f16 = cmds[1][1]
    if os.path.exists(f16):
        os.remove(f16)
