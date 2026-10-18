"""Orchestration and the failure taxonomy.

Every dependency is injected so the whole control flow is testable with no GPU,
no network and no filesystem. The real wiring lives in `_default_deps()`.

Exit codes are the contract with the backend: they distinguish "we broke" from
"the buyer's data was unusable" from "the GPU ran out of memory", which is what
lets the money path decide whether to refund.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
from dataclasses import dataclass, replace
from typing import Callable

from .artifacts import paths, put_file, tar_dir
from .dataset import load_jsonl, split
from .errors import InfraError
from .evalmath import improved
from .report import Reporter
from .spec import SpecError, parse_spec

EXIT_OK = 0
EXIT_SPEC = 2
EXIT_DATA = 3
EXIT_TRAIN = 4
EXIT_UPLOAD = 5
EXIT_INFRA = 6

FETCH_ATTEMPTS = 3


@dataclass(frozen=True)
class Deps:
    fetch: Callable
    train: Callable
    convert: Callable
    upload: Callable
    reporter_factory: Callable
    write_report: Callable
    tar: Callable
    sleep: Callable = time.sleep
    # Auto-tune only. Defaults keep every existing single-run caller unchanged.
    evaluate: Callable | None = None      # (model_ref, revision, rows, recipe) -> {eval_loss,...}
    clock: Callable = time.monotonic
    rmtree: Callable = lambda p: shutil.rmtree(p, ignore_errors=True)
    sha256: Callable | None = None        # (path) -> (hex, bytes) | None
    prepare_resume: Callable | None = None  # (resume_spec, archive_path, target_dir) -> path


def execute(raw_spec: dict, deps: Deps) -> int:
    try:
        spec = parse_spec(raw_spec)
    except SpecError as e:
        # No reporter yet — we may not even have a result_url. Log and bail.
        print(f"[sgl] spec error: {e}", file=sys.stderr)
        return EXIT_SPEC

    rep = deps.reporter_factory(spec)
    dirs = paths(spec.run_id)
    from .budget import RuntimeClock
    from .checkpoint import canonical_sha256
    runtime = RuntimeClock(spec.runtime, clock=deps.clock)

    if spec.resume is not None:
        if (spec.resume.model_id != f"{spec.hf_repo}@{spec.hf_revision}"
                or spec.resume.recipe_sha256 != canonical_sha256(spec.recipe)):
            rep.result("failed", {"error": "resume checkpoint provenance does not match this model or Training Recipe",
                                  "exit_code": EXIT_INFRA})
            return EXIT_INFRA
        if deps.prepare_resume is None:
            rep.result("failed", {"error": "resume checkpoint support is unavailable in this image",
                                  "exit_code": EXIT_INFRA})
            return EXIT_INFRA
        try:
            rep.progress("resuming_checkpoint", 0.01)
            phase_started = runtime.phase_started()
            deps.prepare_resume(spec.resume, dirs["resume_archive"], dirs["resume_dir"])
            runtime.record_phase("resume_download", phase_started)
        except Exception as e:
            rep.result("failed", {"error": f"resume checkpoint failed: {e}",
                                  "exit_code": EXIT_INFRA})
            return EXIT_INFRA
        if runtime.should_finalize():
            rep.result("failed", {"error": "runtime budget entered finalization reserve during resume download",
                                  "exit_code": EXIT_INFRA})
            return EXIT_INFRA

    if spec.kind != "finetune":
        rep.result("failed", {"error": f"kind '{spec.kind}' is not supported by this image"})
        return EXIT_SPEC

    # First liveness signal BEFORE the fetch loop — a run stuck in fetch retries
    # (up to 3 slow timeouts) must not look dead to the watchdog.
    rep.progress("fetching_dataset", 0.02)
    data_started = runtime.phase_started()
    raw_dataset = None
    fetch_error: Exception | None = None
    for attempt in range(1, FETCH_ATTEMPTS + 1):
        try:
            raw_dataset = deps.fetch(spec.dataset_url)
            fetch_error = None
            break
        except Exception as e:
            fetch_error = e
            if attempt < FETCH_ATTEMPTS:
                deps.sleep(2 * attempt)
    if fetch_error is not None:
        rep.result("failed", {"error": f"dataset fetch failed: {fetch_error}"})
        return EXIT_INFRA

    try:
        # Decoding happens INSIDE this try, not in _http_get: a non-UTF-8 buyer
        # file is a data fault (EXIT_DATA, no refund), not an infra fault. str is
        # accepted too so injected test fakes that already hand back text keep working.
        text = raw_dataset.decode("utf-8") if isinstance(raw_dataset, (bytes, bytearray)) else raw_dataset
        rows, stats = load_jsonl(text)
        if len(rows) < 2:
            rep.result("failed", {
                "error": f"dataset unusable: {stats.kept} valid rows of {stats.total}",
                "dataset_stats": stats.__dict__})
            return EXIT_DATA
        if spec.autotune is None:
            train_rows, eval_rows = split(rows, spec.recipe.eval_split, spec.recipe.seed)
    except Exception as e:
        rep.result("failed", {"error": f"dataset error: {e}"})
        return EXIT_DATA
    runtime.record_phase("dataset_fetch_and_validation", data_started)
    if runtime.should_finalize():
        rep.result("failed", {"error": "runtime budget entered finalization reserve before training",
                              "exit_code": EXIT_INFRA})
        return EXIT_INFRA

    if spec.autotune is not None:
        return _execute_autotune(spec, rows, stats, deps, rep, dirs)

    try:
        if spec.runtime is None:
            outcome = deps.train(spec, train_rows, eval_rows, dirs, rep.progress)
        else:
            outcome = deps.train(spec, train_rows, eval_rows, dirs, rep.progress, runtime)
    except InfraError as e:
        payload = {"error": str(e), "exit_code": EXIT_INFRA}
        # Merge any attached context (engine, trained_repo, trained_revision).
        ctx = getattr(e, 'sgl_context', {})
        payload.update(ctx)
        rep.result("failed", payload)
        return EXIT_INFRA
    except Exception as e:
        payload = {"error": f"training failed: {e}"}
        # Merge any attached context (engine, trained_repo, trained_revision).
        ctx = getattr(e, 'sgl_context', {})
        payload.update(ctx)
        rep.result("failed", payload)
        return EXIT_TRAIN

    # GGUF is a nice-to-have. Losing it costs deploy-to-grid, not the run.
    gguf_ok, gguf_error = True, None
    if outcome.get("early_finalized"):
        gguf_ok = False
        gguf_error = "skipped to deliver the adapter within the prepaid runtime"
    else:
        conversion_started = runtime.phase_started()
        try:
            rep.progress("converting", 0.94)
            deps.convert(dirs["merged_dir"], dirs["gguf"], spec.quantize)
        except Exception as e:
            gguf_ok, gguf_error = False, str(e)
        finally:
            runtime.record_phase("gguf_conversion", conversion_started)

    report = {
        "run_id": spec.run_id,
        "base_model": spec.hf_repo,
        "base_revision": spec.hf_revision,
        "seed": outcome["seed"],
        "steps": outcome["steps"],
        "planned_steps": outcome.get("planned_steps", outcome["steps"]),
        "partial_completion": bool(outcome.get("early_finalized")),
        "finalization_reason": outcome.get("finalization_reason", "training_complete"),
        "runtime_phase_seconds": outcome.get("runtime_phase_seconds", runtime.phase_seconds),
        "baseline": outcome["baseline"],
        "final": outcome["final"],
        "improved": improved(outcome["baseline"]["eval_loss"],
                             outcome["final"]["eval_loss"]),
        "dataset_stats": stats.__dict__,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "gguf_ok": gguf_ok,
        "gguf_error": gguf_error,
        "device": outcome.get("device"),
        "dtype": outcome.get("dtype"),
        "engine": outcome.get("engine"),
        "engine_fallback_reason": outcome.get("engine_fallback_reason"),
        # The repo+revision that ACTUALLY trained — with unsloth this is the
        # pinned MIRROR, not base_model/base_revision above. See
        # runner/trainer.py's run_finetune return value.
        "trained_repo": outcome.get("trained_repo"),
        "trained_revision": outcome.get("trained_revision"),
    }

    # The ADAPTER is what the buyer pays for. Only a failure to deliver it is
    # a failed (refundable, exit 5) run. Everything after it is optional and
    # can never turn a delivered run into a failure (_deliver_extras).
    try:
        delivery_started = runtime.phase_started()
        rep.progress("uploading", 0.97)
        deps.tar(dirs["adapter_dir"], dirs["adapter_tar"])
        adapter_ok = deps.upload(spec.artifact_urls["adapter"], dirs["adapter_tar"])
    except Exception as e:  # noqa: BLE001
        rep.result("failed", {"error": f"artifact stage failed: {e}", **report})
        return EXIT_UPLOAD
    if not adapter_ok:
        rep.result("failed", {"error": "adapter upload failed", **report})
        return EXIT_UPLOAD
    runtime.record_phase("adapter_pack_and_upload", delivery_started)
    _deliver_extras(deps, spec, dirs, dirs["report"], report)

    if not rep.result("completed", report):
        print(f"[sgl] WARNING: result callback delivery failed for run {spec.run_id}", file=sys.stderr)
    return EXIT_OK


def _deliver_extras(deps: Deps, spec, art_dirs: dict, report_path: str, report: dict) -> None:
    """Everything after the adapter is delivered: GGUF, checksums, report.json.
    All optional, all best-effort. Mutates `report` so the result callback
    tells the truth (e.g. gguf_ok false if its upload failed). Never raises:
    a delivered adapter must never become a refundable failure because of an
    extra (Codex r3)."""
    if report.get("gguf_ok"):
        try:
            ok = deps.upload(spec.artifact_urls["gguf"], art_dirs["gguf"])
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"[sgl] gguf upload raised: {e}", file=sys.stderr)
        if not ok:
            report["gguf_ok"], report["gguf_error"] = False, "gguf upload failed"
    report.update(_artifact_hashes(deps, art_dirs, report["gguf_ok"]))
    # report.json is written and uploaded LAST so it can never contradict the
    # result callback. Its full content also rides that callback, so failing to
    # store it is logged, never fatal.
    try:
        deps.write_report(report_path, report)
        deps.upload(spec.artifact_urls["report"], report_path)
    except Exception as e:  # noqa: BLE001
        print(f"[sgl] report artifact failed (non-fatal): {e}", file=sys.stderr)


def _artifact_hashes(deps: Deps, dirs: dict, gguf_ok: bool) -> dict:
    """sha256 + size of what was uploaded, so a later deploy can verify the
    exact bytes it downloads. Best-effort: a hashing failure never fails a run
    that already delivered its artifacts (deploy then refuses, honestly)."""
    out: dict = {}
    if deps.sha256 is None:
        return out
    artifacts = [("adapter", dirs["adapter_tar"])]
    if gguf_ok:
        artifacts.append(("gguf", dirs["gguf"]))
    for name, path in artifacts:
        try:
            metadata = deps.sha256(path)
        except Exception as e:  # noqa: BLE001 — informational only
            # Hash artifacts independently. A missing/corrupt optional GGUF
            # must not erase the adapter proof (and vice versa).
            print(f"[sgl] {name} hashing failed: {e}", file=sys.stderr)
            continue
        if metadata:
            out[f"{name}_sha256"], out[f"{name}_bytes"] = metadata
    return out


def _sha256_file(path: str) -> tuple[str, int] | None:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def _execute_autotune(spec, rows, stats, deps: Deps, rep, dirs) -> int:
    """N trials on this one box, inside the one prepaid block.

    Money/honesty rules (Codex design review, 2026-09-24):
      * THREE-way split with the recipe seed: trials train on TRAIN and are
        SELECTED on VALIDATION; the report's baseline/final are measured on
        the untouched TEST split, so the headline number is not inflated by
        the selection itself.
      * A failed trial (OOM, a bad lr) is recorded and the sweep continues.
        The run fails only if EVERY trial failed.
      * Best-so-far is uploaded the moment a trial improves on it, so a
        deadline kill mid-sweep still leaves the buyer the best adapter.
      * Before trial k>1 starts: elapsed + estimate + reserve must fit the
        budget, else the remaining trials are skipped (reported, not hidden).
    """
    at = spec.autotune
    r = spec.recipe
    t0 = deps.clock()

    try:
        rest, test_rows = split(rows, r.eval_split, r.seed)
        if len(rest) < 2:
            raise ValueError("too few rows left after the test split")
        train_rows, val_rows = split(rest, r.eval_split, r.seed + 1)
    except Exception as e:
        rep.result("failed", {"error": f"dataset error: {e}", "exit_code": EXIT_DATA})
        return EXIT_DATA

    n = len(at.trials)
    # Trial 1 must fit too. The backend refuses such a block before charging,
    # so reaching this means the backend authored an impossible budget: our
    # fault, refunded (exit 2), rather than starting work the deadline kills
    # before any model exists.
    if at.trial_estimate_seconds + at.reserve_seconds > at.budget_seconds:
        rep.result("failed", {"error": "spec: auto-tune budget cannot fit a single trial",
                              "exit_code": EXIT_SPEC, "autotune": True})
        return EXIT_SPEC
    trials: list[dict] = []
    delivered_idx: int | None = None   # trial whose adapter is already at the adapter URL
    best: dict | None = None
    best_dirs: dict | None = None
    last_wall: float | None = None
    skipped = 0
    infra_failures = 0

    for idx, t in enumerate(at.trials, start=1):
        elapsed = deps.clock() - t0
        est = (last_wall * 1.2) if last_wall else at.trial_estimate_seconds
        if idx > 1 and elapsed + est + at.reserve_seconds > at.budget_seconds:
            skipped = n - idx + 1
            break

        trial_recipe = replace(r, lr=t.lr, lora_rank=t.lora_rank, lora_alpha=t.lora_alpha, epochs=t.epochs)
        trial_spec = replace(spec, recipe=trial_recipe)
        tdirs = paths(f"{spec.run_id}-t{idx}")
        base_pct, span = 0.05 + 0.8 * (idx - 1) / n, 0.8 / n

        def prog(phase, pct, _b=base_pct, _s=span, _i=idx, **extra):
            return rep.progress(f"trial {_i}/{n}: {phase}", _b + _s * max(0.0, min(1.0, float(pct))),
                                **extra)

        config = {"lr": t.lr, "lora_rank": t.lora_rank, "lora_alpha": t.lora_alpha, "epochs": t.epochs}
        started = deps.clock()
        try:
            outcome = deps.train(trial_spec, train_rows, val_rows, tdirs, prog)
        except Exception as e:  # noqa: BLE001 — isolate the trial, keep sweeping
            wall = deps.clock() - started
            last_wall = wall
            if isinstance(e, InfraError):
                infra_failures += 1
            rec = {"idx": idx, "config": config, "seed": r.seed, "score": None,
                   "wall_seconds": round(wall, 1), "error": str(e)[:300]}
            trials.append(rec)
            rep.experiment(at.experiment_url, rec)
            deps.rmtree(tdirs["root"])
            continue

        wall = deps.clock() - started
        last_wall = wall
        score = float(outcome["final"]["eval_loss"])
        rec = {"idx": idx, "config": config, "seed": r.seed, "score": score,
               "baseline_score": float(outcome["baseline"]["eval_loss"]),
               "wall_seconds": round(wall, 1), "engine": outcome.get("engine")}
        trials.append(rec)
        rep.experiment(at.experiment_url, rec)

        if best is None or score < best["score"]:   # ties keep the earlier (default-first) trial
            if best_dirs is not None:
                deps.rmtree(best_dirs["root"])
            best, best_dirs = {**rec, "outcome": outcome}, tdirs
            # Insurance: the best adapter so far is already delivered.
            try:
                deps.tar(tdirs["adapter_dir"], tdirs["adapter_tar"])
                if deps.upload(spec.artifact_urls["adapter"], tdirs["adapter_tar"]):
                    delivered_idx = idx
                else:
                    print(f"[sgl] best-so-far adapter upload failed (trial {idx})", file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                print(f"[sgl] best-so-far adapter stage failed (trial {idx}): {e}", file=sys.stderr)
        else:
            deps.rmtree(tdirs["root"])

    if best is None:
        all_infra = infra_failures == len(trials) and len(trials) > 0
        code = EXIT_INFRA if all_infra else EXIT_TRAIN
        rep.result("failed", {
            "error": ("every auto-tune trial failed on our side" if all_infra
                      else "training failed: every auto-tune trial failed"),
            "exit_code": code, "autotune": True, "trials": trials, "trials_skipped": skipped})
        return code

    # Honest headline: base vs winner on the TEST split (never selected on).
    test_error = None
    baseline_test = final_test = None
    if deps.evaluate is not None:
        try:
            rep.progress("test_eval", 0.9)
            baseline_test = deps.evaluate(spec.hf_repo, spec.hf_revision, test_rows, r)
            final_test = deps.evaluate(best_dirs["merged_dir"], None, test_rows, r)
        except Exception as e:  # noqa: BLE001 — the model is still delivered
            test_error = str(e)[:300]
            baseline_test = final_test = None

    # GGUF only if the budget still allows; the adapter is already delivered.
    gguf_ok, gguf_error = True, None
    if deps.clock() - t0 > at.budget_seconds - at.reserve_seconds * 0.5:
        gguf_ok, gguf_error = False, "skipped: not enough time left in the block"
    else:
        try:
            rep.progress("converting", 0.94)
            deps.convert(best_dirs["merged_dir"], best_dirs["gguf"], spec.quantize)
        except Exception as e:  # noqa: BLE001
            gguf_ok, gguf_error = False, str(e)

    bo = best["outcome"]
    report = {
        "run_id": spec.run_id,
        "base_model": spec.hf_repo,
        "base_revision": spec.hf_revision,
        "seed": r.seed,
        "steps": bo["steps"],
        "autotune": True,
        "selection_split": "validation",
        "eval_split_name": "test" if final_test else "validation",
        "baseline": baseline_test or bo["baseline"],
        "final": final_test or bo["final"],
        "improved": improved((baseline_test or bo["baseline"])["eval_loss"],
                             (final_test or bo["final"])["eval_loss"]),
        "test_eval_error": test_error,
        "trials": trials,
        "best_idx": best["idx"],
        "best_config": best["config"],
        "trials_skipped": skipped,
        "dataset_stats": stats.__dict__,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "eval_rows": len(test_rows),
        "gguf_ok": gguf_ok,
        "gguf_error": gguf_error,
        "device": bo.get("device"),
        "dtype": bo.get("dtype"),
        "engine": bo.get("engine"),
        "engine_fallback_reason": bo.get("engine_fallback_reason"),
        "trained_repo": bo.get("trained_repo"),
        "trained_revision": bo.get("trained_revision"),
    }

    rep.progress("uploading", 0.97)
    if delivered_idx != best["idx"]:
        # The winner is NOT yet at the adapter URL (its best-so-far upload
        # failed). Re-tar from scratch (a tar that failed midway can leave a
        # partial file; "exists" never means "complete"), then upload.
        try:
            deps.tar(best_dirs["adapter_dir"], best_dirs["adapter_tar"])
            adapter_ok = deps.upload(spec.artifact_urls["adapter"], best_dirs["adapter_tar"])
        except Exception as e:  # noqa: BLE001
            rep.result("failed", {"error": f"artifact stage failed: {e}", "exit_code": EXIT_UPLOAD, **report})
            return EXIT_UPLOAD
        if not adapter_ok:
            rep.result("failed", {"error": "adapter upload failed", "exit_code": EXIT_UPLOAD, **report})
            return EXIT_UPLOAD
    # else: the winner's adapter was already uploaded, from exactly the file
    # still on disk (the upload only ran after its tar succeeded). Touching it
    # again could only fail a delivered run or make the hash describe bytes
    # that were never uploaded.
    _deliver_extras(deps, spec, best_dirs, dirs["report"], report)

    if not rep.result("completed", report):
        print(f"[sgl] WARNING: result callback delivery failed for run {spec.run_id}", file=sys.stderr)
    return EXIT_OK


def _http_get(url: str) -> bytes:
    # Return raw bytes — decoding is the dataset-validation step's job, so a
    # non-UTF-8 buyer file fails as EXIT_DATA, not as a retryable fetch error.
    with urllib.request.urlopen(url, timeout=300) as res:
        return res.read()


def _write_report(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def _lazy_evaluate(model_ref, revision, rows, recipe):
    from .trainer import evaluate_rows
    return evaluate_rows(model_ref, revision, rows, recipe)


def _default_deps() -> Deps:
    from .convert import to_gguf
    from .trainer import run_finetune
    from .checkpoint import prepare_resume_checkpoint

    def train_with_checkpoints(spec, train_rows, eval_rows, out_dirs, on_progress, runtime=None):
        if not spec.checkpoint_urls:
            return run_finetune(
                spec, train_rows, eval_rows, out_dirs, on_progress, runtime=runtime)
        from .checkpoint import CheckpointPublisher, checkpoint_provenance
        publish = CheckpointPublisher(
            urls=spec.checkpoint_urls,
            out_dirs=out_dirs,
            provenance=checkpoint_provenance(spec, train_rows, eval_rows),
            upload=put_file,
            active_slot=spec.resume.slot if spec.resume is not None else None,
        )

        return run_finetune(
            spec, train_rows, eval_rows, out_dirs, on_progress,
            on_checkpoint=publish, runtime=runtime,
        )

    return Deps(
        fetch=_http_get,
        train=train_with_checkpoints,
        convert=to_gguf,
        upload=put_file,
        reporter_factory=lambda s: Reporter(s.progress_url, s.result_url, s.token, s.run_id),
        write_report=_write_report,
        tar=tar_dir,
        evaluate=_lazy_evaluate,
        sha256=_sha256_file,
        prepare_resume=prepare_resume_checkpoint,
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
