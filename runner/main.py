"""Orchestration and the failure taxonomy.

Every dependency is injected so the whole control flow is testable with no GPU,
no network and no filesystem. The real wiring lives in `_default_deps()`.

Exit codes are the contract with the backend: they distinguish "we broke" from
"the buyer's data was unusable" from "the GPU ran out of memory", which is what
lets the money path decide whether to refund.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from dataclasses import dataclass
from typing import Callable

from .artifacts import paths, put_file, tar_dir
from .dataset import load_jsonl, split
from .evalmath import improved
from .report import Reporter
from .spec import SpecError, parse_spec

EXIT_OK = 0
EXIT_SPEC = 2
EXIT_DATA = 3
EXIT_TRAIN = 4
EXIT_UPLOAD = 5


@dataclass(frozen=True)
class Deps:
    fetch: Callable
    train: Callable
    convert: Callable
    upload: Callable
    reporter_factory: Callable
    write_report: Callable
    tar: Callable


def execute(raw_spec: dict, deps: Deps) -> int:
    try:
        spec = parse_spec(raw_spec)
    except SpecError as e:
        # No reporter yet — we may not even have a result_url. Log and bail.
        print(f"[sgl] spec error: {e}", file=sys.stderr)
        return EXIT_SPEC

    rep = deps.reporter_factory(spec)
    dirs = paths(spec.run_id)

    try:
        rep.progress("fetching_dataset", 0.02)
        rows, stats = load_jsonl(deps.fetch(spec.dataset_url))
        if len(rows) < 2:
            rep.result("failed", {
                "error": f"dataset unusable: {stats.kept} valid rows of {stats.total}",
                "dataset_stats": stats.__dict__})
            return EXIT_DATA
        train_rows, eval_rows = split(rows, spec.recipe.eval_split, spec.recipe.seed)
    except Exception as e:
        rep.result("failed", {"error": f"dataset error: {e}"})
        return EXIT_DATA

    try:
        outcome = deps.train(spec, train_rows, eval_rows, dirs, rep.progress)
    except Exception as e:
        rep.result("failed", {"error": f"training failed: {e}"})
        return EXIT_TRAIN

    # GGUF is a nice-to-have. Losing it costs deploy-to-grid, not the run.
    gguf_ok, gguf_error = True, None
    try:
        rep.progress("converting", 0.94)
        deps.convert(dirs["merged_dir"], dirs["gguf"], spec.quantize)
    except Exception as e:
        gguf_ok, gguf_error = False, str(e)

    report = {
        "run_id": spec.run_id,
        "base_model": spec.hf_repo,
        "base_revision": spec.hf_revision,
        "seed": outcome["seed"],
        "steps": outcome["steps"],
        "baseline": outcome["baseline"],
        "final": outcome["final"],
        "improved": improved(outcome["baseline"]["eval_loss"],
                             outcome["final"]["eval_loss"]),
        "dataset_stats": stats.__dict__,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "gguf_ok": gguf_ok,
        "gguf_error": gguf_error,
    }

    rep.progress("uploading", 0.97)
    deps.tar(dirs["adapter_dir"], dirs["adapter_tar"])
    if not deps.upload(spec.artifact_urls["adapter"], dirs["adapter_tar"]):
        rep.result("failed", {"error": "adapter upload failed", **report})
        return EXIT_UPLOAD
    if gguf_ok and not deps.upload(spec.artifact_urls["gguf"], dirs["gguf"]):
        gguf_ok, gguf_error = False, "gguf upload failed"
        report["gguf_ok"] = False
        report["gguf_error"] = gguf_error

    # report.json is written and uploaded LAST so it can never contradict the
    # result callback — every fact it states is final by this point.
    deps.write_report(dirs["report"], report)
    deps.upload(spec.artifact_urls["report"], dirs["report"])

    rep.result("completed", report)
    return EXIT_OK


def _http_get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=300) as res:
        return res.read().decode()


def _write_report(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def _default_deps() -> Deps:
    from .convert import to_gguf
    from .trainer import run_finetune
    return Deps(
        fetch=_http_get,
        train=run_finetune,
        convert=to_gguf,
        upload=put_file,
        reporter_factory=lambda s: Reporter(s.progress_url, s.result_url, s.token, s.run_id),
        write_report=_write_report,
        tar=tar_dir,
    )


def cli() -> int:
    raw = os.environ.get("SGL_JOB_SPEC")
    if not raw:
        print("[sgl] SGL_JOB_SPEC is not set", file=sys.stderr)
        return EXIT_SPEC
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[sgl] SGL_JOB_SPEC is not valid JSON: {e}", file=sys.stderr)
        return EXIT_SPEC
    return execute(parsed, _default_deps())


if __name__ == "__main__":
    sys.exit(cli())
