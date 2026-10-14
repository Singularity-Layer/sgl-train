"""Fail-closed entrypoint for the volume-backed training runtime."""
from __future__ import annotations

import json
import os
import site
import sys
from pathlib import Path
from typing import Callable

RUNTIME_SCHEMA = 1


class RuntimeNotReady(RuntimeError):
    pass


def prepare_runtime(env: dict[str, str] | None = None) -> Path:
    values = os.environ if env is None else env
    expected = values.get("SGL_RUNTIME_VERSION", "").strip()
    root_text = values.get("SGL_RUNTIME_ROOT", "").strip()
    if not expected or not root_text:
        raise RuntimeNotReady("runtime version/root is not configured")

    root = Path(root_text)
    marker_path = root / "READY.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeNotReady("runtime marker is missing or invalid") from exc
    if marker != {"schema": RUNTIME_SCHEMA, "version": expected}:
        raise RuntimeNotReady("runtime marker does not match this bootstrap")

    site_packages = root / "site-packages"
    if not site_packages.is_dir():
        raise RuntimeNotReady("runtime site-packages directory is missing")
    sys.path.insert(0, str(site_packages))
    site.addsitedir(str(site_packages))

    values["SGL_LLAMA_ROOT"] = str(root / "llama.cpp")
    values["SGL_CONVERT_ROOT"] = str(root / "convert-venv")
    values["SGL_BIN_ROOT"] = str(root / "bin")
    values["PATH"] = f"{root / 'bin'}:{values.get('PATH', '')}"
    values["LD_LIBRARY_PATH"] = f"{root / 'bin'}:{values.get('LD_LIBRARY_PATH', '')}"
    Path(values.get("SGL_WORKDIR", "/tmp/sgl/work")).mkdir(parents=True, exist_ok=True)
    Path(values.get("HF_HOME", "/tmp/huggingface")).mkdir(parents=True, exist_ok=True)
    return root


def run(entrypoint: Callable[[], int] | None = None) -> int:
    try:
        prepare_runtime()
    except RuntimeNotReady as exc:
        print(f"[sgl] cached runtime unavailable: {exc}", file=sys.stderr)
        return 6

    if entrypoint is not None:
        return entrypoint()

    # Import after sys.path is wired so torch/transformers resolve exclusively
    # from the immutable volume runtime rather than this tiny image.
    from .main import _default_deps, execute

    try:
        raw = json.loads(os.environ.get("SGL_JOB_SPEC", ""))
    except (TypeError, ValueError):
        print("[sgl] SGL_JOB_SPEC is invalid JSON", file=sys.stderr)
        return 2
    if not isinstance(raw, dict):
        print("[sgl] SGL_JOB_SPEC must be an object", file=sys.stderr)
        return 2
    return execute(raw, _default_deps())


if __name__ == "__main__":
    raise SystemExit(run())
