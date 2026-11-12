"""Before/after sample answers for plain, autotune and refine runs (no GPU)."""
import copy
import hashlib
import json

import pytest

from runner import samples
from runner.dataset import Example
from runner.main import execute, Deps, EXIT_OK

SAMPLES_URL = "https://s.invalid/samples"


def rows_text(n, answer="y"):
    return "\n".join(
        json.dumps({"messages": [{"role": "user", "content": "q%d" % i},
                                 {"role": "assistant", "content": answer}]})
        for i in range(n))


BASE = {
    "run_id": "r1", "kind": "finetune",
    "base_model": {"hf_repo": "Q/M", "revision": "sha"},
    "dataset_url": "https://d.invalid/x.jsonl", "quantize": "Q4_K_M",
    "progress_url": "https://p.invalid", "result_url": "https://res.invalid",
    "token": "t",
    "artifact_urls": {"adapter": "https://a.invalid", "gguf": "https://g.invalid",
                      "report": "https://r.invalid", "samples": SAMPLES_URL},
    "recipe": {"seed": 7, "epochs": 1, "eval_split": 0.3, "batch_size": 1},
}
SWEEP = {"budget_seconds": 7200, "trial_estimate_seconds": 300, "reserve_seconds": 600,
         "experiment_url": "https://x.invalid/experiment"}
TRIALS = [{"lr": 2e-4, "lora_rank": 16, "lora_alpha": 32, "epochs": 1},
          {"lr": 1e-4, "lora_rank": 16, "lora_alpha": 32, "epochs": 1}]


def raw_spec(mode="plain", samples_url=True):
    s = copy.deepcopy(BASE)
    if not samples_url:
        del s["artifact_urls"]["samples"]
    if mode == "autotune":
        s["autotune"] = {"trials": TRIALS, **SWEEP}
    elif mode == "refine":
        s["refine"] = {"max_rounds": 2, "patience": 3, "grid": TRIALS, **SWEEP}
    return s


class Rep:
    def __init__(self):
        self.results, self.progress_calls = [], []

    def progress(self, phase, pct, **extra):
        self.progress_calls.append(phase)
        return True

    def experiment(self, url, payload):
        return True

    def result(self, status, payload):
        self.results.append((status, payload))
        return True


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def outcome(val_loss):
    return {"baseline": {"eval_loss": 2.0, "perplexity": 7.4, "eval_tokens": 10},
            "final": {"eval_loss": val_loss, "perplexity": 2.7, "eval_tokens": 10},
            "steps": 5, "seed": 7, "engine": "vanilla", "trained_repo": "Q/M", "trained_revision": "sha"}


def make(rep, data, trial_seconds=60, **over):
    clock = Clock()
    calls = {"upload": [], "gen": [], "written": {}, "eval_rows": []}
    losses = iter([1.5, 1.2, 1.1, 1.0])

    def train(s, tr, ev, d, cb, *rest):
        clock.t += trial_seconds
        calls["eval_rows"].append([r.messages[0]["content"] for r in ev])
        return outcome(next(losses))

    def generate(ref, rev, prompts, recipe, max_new_tokens):
        calls["gen"].append((ref, rev, len(prompts), max_new_tokens))
        tag = "base" if rev else "tuned"
        return ["Y " if (tag == "tuned" and i % 2 == 0) else f"{tag} answer {i}"
                for i in range(len(prompts))]

    base = dict(
        fetch=lambda url: data, train=train, convert=lambda m, g, q: None,
        upload=lambda url, path: calls["upload"].append((url, path)) or True,
        reporter_factory=lambda s: rep, write_report=lambda p, o: None,
        tar=lambda src, dest: dest,
        evaluate=lambda ref, rev, rows, recipe: {"eval_loss": 1.8 if rev else 0.9,
                                                 "perplexity": 1.0, "eval_tokens": len(rows)},
        clock=clock, rmtree=lambda p: None, generate=generate,
        write_bytes=lambda path, b: calls["written"].__setitem__(path, b),
    )
    base.update(over)
    return Deps(**base), calls


@pytest.mark.parametrize("mode", ["plain", "autotune", "refine"])
def test_samples_written_with_exact_line_shape_and_uploaded(mode):
    rep = Rep()
    deps, calls = make(rep, rows_text(60))
    assert execute(raw_spec(mode), deps) == EXIT_OK
    status, payload = rep.results[-1]
    assert status == "completed"
    (path, data), = calls["written"].items()
    assert path.endswith("samples.jsonl")
    assert (SAMPLES_URL, path) in calls["upload"]
    lines = [json.loads(x) for x in data.decode().splitlines()]
    assert all(set(x) == {"prompt", "reference", "before", "after"} for x in lines)
    assert all(isinstance(v, str) for x in lines for v in x.values())
    assert lines[0]["reference"] == "y" and lines[0]["before"] == "base answer 0"
    assert payload["samples_ok"] is True
    assert payload["samples_count"] == len(lines)
    assert payload["samples_bytes"] == len(data)
    assert payload["samples_sha256"] == hashlib.sha256(data).hexdigest()
    # base model is the pinned base, tuned is the best merged dir; 256 new tokens, greedy
    (b_ref, b_rev, _, b_max), (t_ref, t_rev, _, _) = calls["gen"]
    assert (b_ref, b_rev, b_max) == ("Q/M", "sha", 256)
    assert t_rev is None and t_ref.endswith("merged")
    # prompts come from the held-out split, never from rows any trial selected on
    prompts = {x["prompt"] for x in lines}
    if mode == "plain":
        assert prompts == set(calls["eval_rows"][0])
    else:
        assert not prompts & {p for ev in calls["eval_rows"] for p in ev}


def test_at_most_forty_samples_and_trimmed_to_2000_chars():
    long = "z" * 5000
    data = "\n".join(json.dumps({"messages": [{"role": "user", "content": long},
                                              {"role": "assistant", "content": long}]})
                     for _ in range(300))
    rep = Rep()

    def generate(ref, rev, prompts, recipe, max_new_tokens):
        return [long] * len(prompts)

    deps, calls = make(rep, data, generate=generate)
    assert execute(raw_spec("autotune"), deps) == EXIT_OK
    (_, written), = calls["written"].items()
    lines = [json.loads(x) for x in written.decode().splitlines()]
    assert len(lines) == 40
    assert all(len(v) == 2000 for x in lines for v in x.values())
    # long free-text references: no exact-match number at all
    assert "quality" not in rep.results[-1][1]


def test_exact_match_for_short_label_references_case_and_space_insensitive():
    rep = Rep()
    deps, _ = make(rep, rows_text(60, answer=" y "))
    assert execute(raw_spec("refine"), deps) == EXIT_OK
    q = rep.results[-1][1]["quality"]
    n = q["n"]
    assert n == rep.results[-1][1]["samples_count"]
    assert q["exact_match_before"] == 0.0
    # the fake tuned model answers "Y " on every even prompt
    assert q["exact_match_after"] == pytest.approx(((n + 1) // 2) / n)


def test_exact_match_helper_only_for_short_references():
    assert samples.exact_match(["a" * 33], ["x"], ["x"]) is None
    assert samples.exact_match([], [], []) is None
    assert samples.exact_match([" Yes "], ["no"], ["yes"]) == {
        "exact_match_before": 0.0, "exact_match_after": 1.0, "n": 1}


def test_pick_uses_messages_before_the_last_assistant_turn():
    rows = [Example(messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                              {"role": "assistant", "content": "a1"}, {"role": "user", "content": "u2"},
                              {"role": "assistant", "content": "a2"}]),
            Example(messages=[{"role": "assistant", "content": "only"}])]
    picked = samples.pick(rows)
    assert len(picked) == 1
    msgs, ref = picked[0]
    assert ref == "a2" and [m["content"] for m in msgs] == ["s", "u", "a1", "u2"]
    assert samples.render_prompt(msgs).startswith("system: s")


@pytest.mark.parametrize("mode", ["plain", "autotune", "refine"])
def test_samples_failure_never_fails_the_run(mode):
    rep = Rep()

    def boom(*a):
        raise RuntimeError("CUDA out of memory")

    deps, calls = make(rep, rows_text(60), generate=boom)
    assert execute(raw_spec(mode), deps) == EXIT_OK
    status, payload = rep.results[-1]
    assert status == "completed"
    assert payload["samples_ok"] is False and "out of memory" in payload["samples_error"]
    assert payload["gguf_ok"] is True
    assert any(u == "https://a.invalid" for (u, _) in calls["upload"])


def test_samples_upload_failure_is_samples_ok_false_not_a_failed_run():
    rep = Rep()
    deps, _ = make(rep, rows_text(60), upload=lambda url, path: url != SAMPLES_URL)
    assert execute(raw_spec("plain"), deps) == EXIT_OK
    assert rep.results[-1][1]["samples_ok"] is False


@pytest.mark.parametrize("mode", ["plain", "autotune", "refine"])
def test_no_samples_url_means_no_samples_work(mode):
    rep = Rep()
    deps, calls = make(rep, rows_text(60))
    assert execute(raw_spec(mode, samples_url=False), deps) == EXIT_OK
    payload = rep.results[-1][1]
    assert calls["gen"] == [] and calls["written"] == {}
    assert not any(k.startswith("samples") for k in payload) and "quality" not in payload
    assert "generating_samples" not in rep.progress_calls


def test_samples_skipped_when_the_block_has_no_room_after_the_reserve():
    rep = Rep()
    # one 6400s round: 6400 + 300 (samples) + 600 (reserve) > 7200
    deps, calls = make(rep, rows_text(60), trial_seconds=6400)
    raw = raw_spec("autotune")
    raw["autotune"]["trials"] = TRIALS[:1]
    assert execute(raw, deps) == EXIT_OK
    payload = rep.results[-1][1]
    assert calls["gen"] == []
    assert payload["samples_ok"] is False and "skipped" in payload["samples_error"]


def test_plain_run_skips_samples_when_runtime_reserve_is_reached():
    rep = Rep()
    raw = raw_spec("plain")
    raw["runtime"] = {"budget_seconds": 3600, "finalization_reserve_seconds": 600,
                      "estimated_p95_seconds": 1200, "profile_version": "v1"}
    # train consumes 3000s of a 3600s block: 600 left == reserve, no room for samples
    deps, calls = make(rep, rows_text(60), trial_seconds=3000)
    assert execute(raw, deps) == EXIT_OK
    assert calls["gen"] == []
    assert rep.results[-1][1]["samples_ok"] is False
