"""Auto-tune sweep: N trials on one box, one prepaid block (no GPU needed)."""
import copy

import pytest

from runner.errors import InfraError
from runner.main import execute, Deps, EXIT_OK, EXIT_SPEC, EXIT_TRAIN, EXIT_INFRA, EXIT_UPLOAD
from runner.spec import SpecError, parse_spec

ROWS = "\n".join(
    '{"messages":[{"role":"user","content":"%d"},{"role":"assistant","content":"y"}]}' % i
    for i in range(40))

TRIALS = [
    {"lr": 2e-4, "lora_rank": 16, "lora_alpha": 32, "epochs": 3},
    {"lr": 1e-4, "lora_rank": 16, "lora_alpha": 32, "epochs": 3},
    {"lr": 5e-4, "lora_rank": 16, "lora_alpha": 32, "epochs": 3},
]

BASE = {
    "run_id": "r1", "kind": "finetune",
    "base_model": {"hf_repo": "Q/M", "revision": "sha"},
    "dataset_url": "https://d.invalid/x.jsonl", "quantize": "Q4_K_M",
    "progress_url": "https://p.invalid", "result_url": "https://res.invalid",
    "token": "t",
    "artifact_urls": {"adapter": "https://a.invalid", "gguf": "https://g.invalid",
                      "report": "https://r.invalid"},
    "recipe": {"seed": 7, "epochs": 1, "eval_split": 0.2, "batch_size": 1},
    "autotune": {"trials": TRIALS, "budget_seconds": 3600, "trial_estimate_seconds": 300,
                 "reserve_seconds": 600, "experiment_url": "https://x.invalid/experiment"},
}


def spec(**over):
    s = copy.deepcopy(BASE)
    s["autotune"].update(over)
    return s


class Rep:
    def __init__(self):
        self.results, self.experiments, self.progress_calls = [], [], []

    def progress(self, phase, pct, **extra):
        self.progress_calls.append((phase, pct))
        return True

    def experiment(self, url, payload):
        self.experiments.append(payload)
        return True

    def result(self, status, payload):
        self.results.append((status, payload))
        return True


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def outcome(val_loss, base=2.0):
    return {"baseline": {"eval_loss": base, "perplexity": 7.4, "eval_tokens": 10},
            "final": {"eval_loss": val_loss, "perplexity": 2.7, "eval_tokens": 10},
            "steps": 5, "seed": 7, "engine": "vanilla", "trained_repo": "Q/M", "trained_revision": "sha"}


def make(rep, losses, clock=None, trial_seconds=60, **over):
    clock = clock or Clock()
    calls = {"train": [], "upload": [], "rm": [], "eval": []}

    def train(s, tr, ev, d, cb):
        calls["train"].append((s.recipe.lr, len(tr), len(ev), d["root"]))
        cb("training", 0.5)
        clock.t += trial_seconds
        v = losses[len(calls["train"]) - 1]
        if isinstance(v, Exception):
            raise v
        return outcome(v)

    def evaluate(ref, rev, rows, recipe):
        calls["eval"].append((ref, rev, len(rows)))
        return {"eval_loss": 1.8 if rev else 0.9, "perplexity": 1.0, "eval_tokens": len(rows)}

    base = dict(
        fetch=lambda url: ROWS, train=train, convert=lambda m, g, q: None,
        upload=lambda url, path: calls["upload"].append((url, path)) or True,
        reporter_factory=lambda s: rep, write_report=lambda p, o: None,
        tar=lambda src, dest: dest, evaluate=evaluate, clock=clock,
        rmtree=lambda p: calls["rm"].append(p), sha256=lambda p: ("ab" * 32, 123),
    )
    base.update(over)
    return Deps(**base), calls


def test_spec_parses_and_clamps_trials():
    s = parse_spec(spec(trials=[{"lr": 9, "lora_rank": 999, "epochs": 0}]))
    t = s.autotune.trials[0]
    assert t.lr == 1e-3 and t.lora_rank == 64 and t.epochs == 1


@pytest.mark.parametrize("bad", [
    {"trials": []}, {"trials": [{}] * 7}, {"trials": "x"},
    {"budget_seconds": 10}, {"experiment_url": "http://x.invalid"},
])
def test_malformed_sweep_is_a_spec_error_not_a_silent_single_run(bad):
    with pytest.raises(SpecError):
        parse_spec(spec(**bad))


def test_absent_autotune_is_a_plain_run():
    s = copy.deepcopy(BASE)
    del s["autotune"]
    assert parse_spec(s).autotune is None


def test_picks_lowest_validation_loss_and_reports_test_split():
    rep = Rep()
    deps, calls = make(rep, [1.5, 1.2, 1.4])
    assert execute(spec(), deps) == EXIT_OK
    status, payload = rep.results[-1]
    assert status == "completed"
    assert payload["best_idx"] == 2
    assert payload["best_config"]["lr"] == 1e-4
    assert len(rep.experiments) == 3 and [e["idx"] for e in rep.experiments] == [1, 2, 3]
    # headline = TEST split, measured by evaluate(), not the validation score used to select
    assert payload["eval_split_name"] == "test"
    assert payload["final"]["eval_loss"] == 0.9 and payload["baseline"]["eval_loss"] == 1.8
    assert payload["selection_split"] == "validation"
    # three disjoint splits: train + val + test == all rows
    _, n_train, n_val, _ = calls["train"][0]
    assert n_train + n_val + payload["eval_rows"] == 40
    assert payload["gguf_sha256"] == "ab" * 32
    # the losing trials' dirs are cleaned up, the winner is kept
    assert any(p.endswith("r1-t1") for p in calls["rm"])
    assert any(p.endswith("r1-t3") for p in calls["rm"])
    assert not any(p.endswith("r1-t2") for p in calls["rm"])


def test_ties_keep_the_earlier_default_trial():
    rep = Rep()
    deps, _ = make(rep, [1.2, 1.2, 1.3])
    assert execute(spec(), deps) == EXIT_OK
    assert rep.results[-1][1]["best_idx"] == 1


def test_best_so_far_adapter_uploaded_on_each_improvement():
    rep = Rep()
    deps, calls = make(rep, [1.5, 1.2, 1.4])
    execute(spec(), deps)
    adapter_uploads = [p for (u, p) in calls["upload"] if u == "https://a.invalid"]
    # trial1 (first best), trial2 (improved), final re-upload of trial2
    assert [p.split("/")[-2] for p in adapter_uploads] == ["r1-t1", "r1-t2", "r1-t2"]


def test_a_failed_trial_does_not_fail_the_run():
    rep = Rep()
    deps, _ = make(rep, [1.5, RuntimeError("CUDA out of memory"), 1.4])
    assert execute(spec(), deps) == EXIT_OK
    payload = rep.results[-1][1]
    assert payload["best_idx"] == 3
    failed = [t for t in payload["trials"] if t["score"] is None]
    assert len(failed) == 1 and "out of memory" in failed[0]["error"]


def test_every_trial_failed_training_is_exit_train_no_refund():
    rep = Rep()
    deps, _ = make(rep, [RuntimeError("nan"), RuntimeError("nan"), RuntimeError("nan")])
    assert execute(spec(), deps) == EXIT_TRAIN
    assert rep.results[-1][1]["exit_code"] == EXIT_TRAIN


def test_every_trial_failed_on_our_side_is_exit_infra_refund():
    rep = Rep()
    deps, _ = make(rep, [InfraError("download"), InfraError("download"), InfraError("download")])
    assert execute(spec(), deps) == EXIT_INFRA
    assert rep.results[-1][1]["exit_code"] == EXIT_INFRA


def test_budget_gate_skips_trials_that_cannot_finish():
    rep = Rep()
    # 1000s trials, 3600s block, 600s reserve: trial 2 needs 1000+1200+600 <= 3600 ok,
    # trial 3 needs 2000+1200+600 > 3600 -> skipped.
    deps, calls = make(rep, [1.5, 1.2, 1.1], trial_seconds=1000)
    assert execute(spec(), deps) == EXIT_OK
    payload = rep.results[-1][1]
    assert len(calls["train"]) == 2
    assert payload["trials_skipped"] == 1
    assert payload["best_idx"] == 2


def test_gguf_skipped_when_block_nearly_spent_but_adapter_still_delivered():
    rep = Rep()
    deps, calls = make(rep, [1.5], trial_seconds=3500)
    assert execute(spec(trials=[TRIALS[0]]), deps) == EXIT_OK
    payload = rep.results[-1][1]
    assert payload["gguf_ok"] is False and "skipped" in payload["gguf_error"]
    assert any(u == "https://a.invalid" for (u, _) in calls["upload"])


def test_test_eval_failure_falls_back_to_validation_and_says_so():
    rep = Rep()

    def bad_eval(*a):
        raise RuntimeError("oom")

    deps, _ = make(rep, [1.5, 1.2, 1.4], evaluate=bad_eval)
    assert execute(spec(), deps) == EXIT_OK
    payload = rep.results[-1][1]
    assert payload["eval_split_name"] == "validation"
    assert payload["test_eval_error"] == "oom"
    assert payload["final"]["eval_loss"] == 1.2


def test_final_adapter_upload_failure_is_exit_upload():
    rep = Rep()
    deps, _ = make(rep, [1.5], upload=lambda url, path: url != "https://a.invalid")
    assert execute(spec(trials=[TRIALS[0]]), deps) == EXIT_UPLOAD


def test_bad_spec_autotune_never_touches_the_gpu():
    rep = Rep()
    deps, calls = make(rep, [1.0])
    assert execute(spec(trials=[]), deps) == EXIT_SPEC
    assert calls["train"] == []


def test_trial_one_that_cannot_fit_is_refused_as_our_fault_before_training():
    rep = Rep()
    deps, calls = make(rep, [1.0])
    code = execute(spec(budget_seconds=900, trial_estimate_seconds=600, reserve_seconds=600), deps)
    assert code == EXIT_SPEC
    assert calls["train"] == []
    assert rep.results[-1][1]["exit_code"] == EXIT_SPEC


def test_final_stage_always_retars_never_trusts_a_partial_file():
    rep = Rep()
    tars = []
    deps, _ = make(rep, [1.5], tar=lambda src, dest: tars.append(dest) or dest)
    assert execute(spec(trials=[TRIALS[0]]), deps) == EXIT_OK
    assert len(tars) == 2   # best-so-far + mandatory final re-tar


def test_final_reupload_failure_is_success_when_that_best_was_already_delivered():
    rep = Rep()
    n = {"adapter": 0}

    def upload(url, path):
        if url == "https://a.invalid":
            n["adapter"] += 1
            return n["adapter"] == 1   # best-so-far ok, final re-upload fails
        return True

    deps, _ = make(rep, [1.5], upload=upload)
    assert execute(spec(trials=[TRIALS[0]]), deps) == EXIT_OK
    assert rep.results[-1][0] == "completed"


def test_final_upload_failure_when_best_never_delivered_is_exit_upload():
    rep = Rep()
    # trial 1 delivered, trial 2 improves but its best-so-far AND final uploads fail
    n = {"adapter": 0}

    def upload(url, path):
        if url == "https://a.invalid":
            n["adapter"] += 1
            return n["adapter"] == 1
        return True

    deps, _ = make(rep, [1.5, 1.2], upload=upload)
    assert execute(spec(trials=TRIALS[:2]), deps) == EXIT_UPLOAD
