"""Runpod Serverless queue handler for managed training.

The existing runner remains the single training implementation. This module
only unwraps the Serverless job envelope and returns a small terminal summary;
progress, experiments, results, and artifacts still use the authenticated
callback URLs in the job spec.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .main import _default_deps, execute


def _run(spec: dict[str, Any]) -> int:
    return execute(spec, _default_deps())


def handle_job(job: object, run: Callable[[dict[str, Any]], int] = _run) -> dict[str, Any]:
    if not isinstance(job, dict):
        return {"ok": False, "exit_code": 2, "error": "job must be an object"}
    payload = job.get("input")
    if not isinstance(payload, dict):
        return {"ok": False, "exit_code": 2, "error": "job.input must be an object"}
    spec = payload.get("spec")
    if not isinstance(spec, dict):
        return {"ok": False, "exit_code": 2, "error": "job.input.spec must be an object"}

    code = int(run(spec))
    return {
        "ok": code == 0,
        "exit_code": code,
        "run_id": str(spec.get("run_id", ""))[:64],
    }


def cli() -> None:
    import runpod

    runpod.serverless.start({"handler": handle_job})


if __name__ == "__main__":
    cli()
