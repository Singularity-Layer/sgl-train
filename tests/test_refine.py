"""Guided refinement: rounds through a grid with an early stop (no GPU needed)."""
import copy

import pytest

from runner.main import execute, Deps, EXIT_OK, EXIT_SPEC
from runner.spec import SpecError, parse_spec

ROWS = "\n".join(
    '{"messages":[{"role":"user","content":"%d"},{"role":"assistant","content":"y"}]}' % i
    for i in range(40))

GRID = [{"lr": 2e-4 / (i + 1), "lora_rank": 16, "lora_alpha": 32, "epochs": 1} for i in range(16)]

BASE = {
    "run_id": "r1", "kind": "finetune",
    "base_model": {"hf_repo": "Q/M", "revision": "sha"},
    "dataset_url": "https://d.invalid/x.jsonl", "quantize": "Q4_K_M",
    "progress_url": "https://p.invalid", "result_url": "https://res.invalid",
    "token": "t",
    "artifact_urls": {"adapter": "https://a.invalid", "gguf": "https://g.invalid",
                      "report": "https://r.invalid"},
    "recipe": {"seed": 7, "epochs": 1, "eval_split": 0.2, "batch_size": 1},
    "refine": {"max_rounds": 8, "patience": 2, "grid": GRID[:8], "budget_seconds": 7200,
               "trial_estimate_seconds": 300, "reserve_seconds": 600,
               "experiment_url": "https://x.invalid/experiment"},
}


def spec(**over):
    s = copy.deepcopy(BASE)
    s["refine"].update(over)
    return s


class Rep:
    def __init__(self):
        self.results, self.experiments, self.progress_calls = [], [], []

    def progress(self, phase, pct, **extra):
        self.progress_calls.append((phase, pct))
        return True

    def experiment(self, url, payload):
        self.experiments.append((url, payload))
        return True

    def result(self, status, payload):
        self.results.append((status, payload))
        return True


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class Killed(BaseException):
    """Stands in for the deadline kill: not an Exception, so nothing catches it."""


def outcome(val_loss, base=2.0):
    return {"baseline": {"eval_loss": base, "perplexity": 7.4, "eval_tokens": 10},
            "final": {"eval_loss": val_loss, "perplexity": 2.7, "eval_tokens": 10},
            "steps": 5, "seed": 7, "engine": "vanilla", "trained_repo": "Q/M", "trained_revision": "sha"}


def make(rep, losses, clock=None, trial_seconds=60, **over):
    clock = clock or Clock()
    calls = {"train": [], "upload": [], "rm": []}

    def train(s, tr, ev, d, cb):
        calls["train"].append((s.recipe.lr, d["root"]))
        cb("training", 0.5)
        clock.t += trial_seconds
        v = losses[len(calls["train"]) - 1]
        if isinstance(v, BaseException):
            raise v
        return outcome(v)

    base = dict(
        fetch=lambda url: ROWS, train=train, convert=lambda m, g, q: None,
        upload=lambda url, path: calls["upload"].append((url, path)) or True,
        reporter_factory=lambda s: rep, write_report=lambda p, o: None,
        tar=lambda src, dest: dest,
        evaluate=lambda ref, rev, rows, recipe: {"eval_loss": 1.8 if rev else 0.9,
                                                 "perplexity": 1.0, "eval_tokens": len(rows)},
        clock=clock, rmtree=lambda p: calls["rm"].append(p),
    )
    base.update(over)
    return Deps(**base), calls


def test_spec_parses_refine_and_defaults_patience_to_three():
    raw = spec()
    del raw["refine"]["patience"]
    s = parse_spec(raw)
    assert s.refine.max_rounds == 8 and s.refine.patience == 3 and len(s.refine.grid) == 8
    assert s.autotune is None


def test_grid_values_are_clamped_like_autotune():
    grid = [{"lr": 9, "lora_rank": 999, "lora_alpha": 1, "epochs": 0}] + GRID[1:2]
    s = parse_spec(spec(max_rounds=2, grid=grid))
    t = s.refine.grid[0]
    assert (t.lr, t.lora_rank, t.lora_alpha, t.epochs) == (1e-3, 64, 8, 1)


def test_grid_longer_than_the_cap_only_uses_max_rounds_entries():
    s = parse_spec(spec(max_rounds=3, grid=GRID))
    assert len(s.refine.grid) == 3


@pytest.mark.parametrize("bad", [
    {"max_rounds": 1}, {"max_rounds": 17}, {"patience": 0}, {"patience": 6},
    {"grid": GRID[:7]}, {"grid": "x"}, {"grid": [1] * 8}, {"budget_seconds": 10},
    {"experiment_url": "http://x.invalid"},
])
def test_malformed_refine_is_a_spec_error(bad):
    with pytest.raises(SpecError):
        parse_spec(spec(**bad))


def test_refine_and_autotune_together_fail_fast_without_training():
    raw = spec()
    raw["autotune"] = {"trials": GRID[:2], "budget_seconds": 3600, "trial_estimate_seconds": 300,
                       "reserve_seconds": 600, "experiment_url": "https://x.invalid/experiment"}
    with pytest.raises(SpecError, match="mutually exclusive"):
        parse_spec(raw)
    rep = Rep()
    deps, calls = make(rep, [1.0])
    assert execute(raw, deps) == EXIT_SPEC
    assert calls["train"] == [] and rep.results == []


def test_refine_cannot_share_a_runtime_budget():
    raw = spec()
    raw["runtime"] = {"budget_seconds": 3600, "finalization_reserve_seconds": 300,
                      "estimated_p95_seconds": 1200, "profile_version": "v1"}
    with pytest.raises(SpecError, match="refine cannot"):
        parse_spec(raw)


def test_patience_stops_before_the_round_cap_when_scores_stop_improving():
    rep = Rep()
    # round 2 improves; rounds 3 and 4 are within 1 percent -> patience 2 exhausted.
    deps, calls = make(rep, [1.5, 1.2, 1.195, 1.19, 0.5, 0.5, 0.5, 0.5])
    assert execute(spec(), deps) == EXIT_OK
    status, payload = rep.results[-1]
    assert status == "completed"
    assert len(calls["train"]) == 4
    assert payload["refine"] == {"rounds_run": 4, "max_rounds": 8, "stopped_reason": "patience"}
    # a sub-1-percent gain still becomes the kept best; it just does not reset patience
    assert payload["best_idx"] == 4
    assert payload["trials_skipped"] == 0
    assert payload["eval_split_name"] == "test" and payload["final"]["eval_loss"] == 0.9
    # experiment callbacks keep the autotune body shape, 1-based idx
    assert [p["idx"] for (_, p) in rep.experiments] == [1, 2, 3, 4]
    assert all(u == "https://x.invalid/experiment" for (u, _) in rep.experiments)
    assert set(rep.experiments[0][1]) >= {"idx", "config", "seed", "score", "baseline_score", "wall_seconds"}
    # heartbeats keep "trial i/N" with N = max_rounds
    assert ("trial 1/8: training" in [ph for ph, _ in rep.progress_calls])


def test_failed_rounds_count_toward_patience_once_a_best_exists():
    rep = Rep()
    deps, calls = make(rep, [1.5, RuntimeError("oom"), RuntimeError("oom"), 0.5])
    assert execute(spec(), deps) == EXIT_OK
    assert len(calls["train"]) == 3
    assert rep.results[-1][1]["refine"]["stopped_reason"] == "patience"


def test_round_cap_stop_when_every_round_keeps_improving():
    rep = Rep()
    deps, calls = make(rep, [1.6, 1.4, 1.2, 1.0])
    assert execute(spec(max_rounds=4, grid=GRID[:4]), deps) == EXIT_OK
    payload = rep.results[-1][1]
    assert len(calls["train"]) == 4
    assert payload["refine"] == {"rounds_run": 4, "max_rounds": 4, "stopped_reason": "max_rounds"}
    assert payload["best_idx"] == 4
    assert [ph for ph, _ in rep.progress_calls if ph.startswith("trial 4/")] == ["trial 4/4: training"]


def test_budget_gate_stops_refine():
    rep = Rep()
    # 2000s rounds, 7200s block, 600s reserve: round 3 needs 4000+2400+600 <= 7200 ok,
    # round 4 needs 6000+2400+600 > 7200 -> stop.
    deps, calls = make(rep, [1.6, 1.4, 1.2, 1.0, 0.9, 0.8, 0.7, 0.6], trial_seconds=2000)
    assert execute(spec(), deps) == EXIT_OK
    payload = rep.results[-1][1]
    assert len(calls["train"]) == 3
    assert payload["refine"] == {"rounds_run": 3, "max_rounds": 8, "stopped_reason": "budget"}
    assert payload["trials_skipped"] == 5


def test_refine_budget_that_cannot_fit_round_one_is_refused_before_training():
    rep = Rep()
    deps, calls = make(rep, [1.0])
    assert execute(spec(budget_seconds=900, trial_estimate_seconds=600), deps) == EXIT_SPEC
    assert calls["train"] == []


def test_best_adapter_is_delivered_before_a_kill_mid_loop():
    rep = Rep()
    deps, calls = make(rep, [1.5, 1.2, Killed()])
    with pytest.raises(Killed):
        execute(spec(), deps)
    adapter_uploads = [p for (u, p) in calls["upload"] if u == "https://a.invalid"]
    # the best so far (round 2) is already at the adapter URL when the box dies
    assert [p.split("/")[-2] for p in adapter_uploads] == ["r1-t1", "r1-t2"]
    assert rep.results == []


def test_refine_report_keeps_every_autotune_field():
    rep = Rep()
    deps, _ = make(rep, [1.6, 1.4])
    assert execute(spec(max_rounds=2, grid=GRID[:2]), deps) == EXIT_OK
    payload = rep.results[-1][1]
    for key in ("baseline", "final", "improved", "trials", "best_idx", "best_config",
                "trials_skipped", "selection_split", "eval_split_name", "autotune", "gguf_ok"):
        assert key in payload
    assert "samples_ok" not in payload
