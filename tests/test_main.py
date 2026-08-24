from runner.main import execute, Deps, EXIT_OK, EXIT_SPEC, EXIT_DATA, EXIT_TRAIN, EXIT_UPLOAD

BASE = {
    "run_id": "r1", "kind": "finetune",
    "base_model": {"hf_repo": "Q/M", "revision": "sha"},
    "dataset_url": "https://d.invalid/x.jsonl", "quantize": "Q4_K_M",
    "progress_url": "https://p.invalid", "result_url": "https://res.invalid",
    "token": "t",
    "artifact_urls": {"adapter": "https://a.invalid", "gguf": "https://g.invalid",
                      "report": "https://r.invalid"},
    "recipe": {"seed": 1, "epochs": 1, "eval_split": 0.5, "batch_size": 1},
}

ROWS = "\n".join(
    '{"messages":[{"role":"user","content":"%d"},{"role":"assistant","content":"y"}]}' % i
    for i in range(10))


class SpyReporter:
    def __init__(self):
        self.results = []
        self.progress_calls = []

    def progress(self, phase, pct, **extra):
        self.progress_calls.append(phase)
        return True

    def result(self, status, payload):
        self.results.append((status, payload))
        return True


def make_deps(rep, **over):
    base = dict(
        fetch=lambda url: ROWS,
        train=lambda s, tr, ev, d, cb: {"baseline": {"eval_loss": 2.0, "perplexity": 7.4, "eval_tokens": 10},
                                        "final": {"eval_loss": 1.0, "perplexity": 2.7, "eval_tokens": 10},
                                        "steps": 5, "seed": 1},
        convert=lambda m, g, q: None,
        upload=lambda url, path: True,
        reporter_factory=lambda spec: rep,
        write_report=lambda path, obj: None,
        tar=lambda src, dest: dest,
    )
    base.update(over)
    return Deps(**base)


def test_happy_path_reports_completed_with_both_scores():
    rep = SpyReporter()
    assert execute(BASE, make_deps(rep)) == EXIT_OK
    status, payload = rep.results[-1]
    assert status == "completed"
    assert payload["baseline"]["eval_loss"] == 2.0
    assert payload["final"]["eval_loss"] == 1.0
    assert payload["improved"] is True
    assert payload["seed"] == 1


def test_bad_spec_exits_without_touching_the_gpu():
    rep = SpyReporter()
    called = {"train": False}

    def boom(*a):
        called["train"] = True

    code = execute({"kind": "finetune"}, make_deps(rep, train=boom))
    assert code == EXIT_SPEC
    assert called["train"] is False


def test_empty_dataset_fails_before_training():
    rep = SpyReporter()
    code = execute(BASE, make_deps(rep, fetch=lambda url: "not json\nnot json\n"))
    assert code == EXIT_DATA
    assert rep.results[-1][0] == "failed"
    assert "dataset" in rep.results[-1][1]["error"]


def test_training_failure_is_reported_not_swallowed():
    rep = SpyReporter()

    def boom(*a):
        raise RuntimeError("CUDA out of memory")

    assert execute(BASE, make_deps(rep, train=boom)) == EXIT_TRAIN
    assert "CUDA out of memory" in rep.results[-1][1]["error"]


def test_convert_failure_still_completes_the_run():
    rep = SpyReporter()

    def boom(*a):
        raise RuntimeError("quantize failed")

    assert execute(BASE, make_deps(rep, convert=boom)) == EXIT_OK
    status, payload = rep.results[-1]
    assert status == "completed"
    assert payload["gguf_ok"] is False
    assert "quantize failed" in payload["gguf_error"]


def test_adapter_upload_failure_fails_the_run():
    rep = SpyReporter()
    assert execute(BASE, make_deps(rep, upload=lambda url, path: False)) == EXIT_UPLOAD
    assert rep.results[-1][0] == "failed"


def test_no_improvement_is_reported_honestly():
    rep = SpyReporter()
    flat = lambda s, tr, ev, d, cb: {
        "baseline": {"eval_loss": 2.0, "perplexity": 7.4, "eval_tokens": 10},
        "final": {"eval_loss": 1.999, "perplexity": 7.4, "eval_tokens": 10},
        "steps": 5, "seed": 1}
    assert execute(BASE, make_deps(rep, train=flat)) == EXIT_OK
    assert rep.results[-1][1]["improved"] is False


def test_gguf_upload_failure_completes_but_report_says_gguf_failed():
    rep = SpyReporter()
    written = {}

    def fake_write(path, obj):
        written["report"] = dict(obj)

    def gguf_upload_fails(url, path):
        return "g.invalid" not in url   # only the gguf PUT fails

    code = execute(BASE, make_deps(rep, upload=gguf_upload_fails, write_report=fake_write))
    assert code == EXIT_OK
    status, payload = rep.results[-1]
    assert status == "completed"
    assert payload["gguf_ok"] is False
    assert "upload" in payload["gguf_error"]
    # the persisted report.json must agree with the result callback
    assert written["report"]["gguf_ok"] is False
    assert written["report"]["gguf_error"] == payload["gguf_error"]
