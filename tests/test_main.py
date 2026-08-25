from runner.errors import InfraError
from runner.main import (execute, Deps, EXIT_OK, EXIT_SPEC, EXIT_DATA, EXIT_TRAIN,
                          EXIT_UPLOAD, EXIT_INFRA)

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
                                        "steps": 5, "seed": 1,
                                        "device": "cuda", "dtype": "torch.bfloat16"},
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
    assert payload["device"] == "cuda"
    assert payload["dtype"] == "torch.bfloat16"


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


def test_kind_other_than_finetune_is_rejected():
    rep = SpyReporter()
    spec = dict(BASE, kind="scratch")
    code = execute(spec, make_deps(rep))
    assert code == EXIT_SPEC
    assert "not supported" in rep.results[-1][1]["error"]


def test_fetch_retries_and_gives_up_after_three_failures():
    rep = SpyReporter()
    calls = {"count": 0}

    def flaky(url):
        calls["count"] += 1
        raise ConnectionError("boom")

    sleeps = []
    code = execute(BASE, make_deps(rep, fetch=flaky, sleep=lambda s: sleeps.append(s)))
    assert code == EXIT_INFRA
    assert calls["count"] == 3
    assert len(sleeps) == 2
    assert "dataset fetch failed" in rep.results[-1][1]["error"]


def test_fetch_succeeds_after_two_failures():
    rep = SpyReporter()
    calls = {"count": 0}

    def flaky(url):
        calls["count"] += 1
        if calls["count"] < 3:
            raise ConnectionError("boom")
        return ROWS

    code = execute(BASE, make_deps(rep, fetch=flaky, sleep=lambda s: None))
    assert code == EXIT_OK
    assert calls["count"] == 3
    assert rep.results[-1][0] == "completed"


def test_tar_failure_is_reported_as_artifact_stage_failure():
    rep = SpyReporter()

    def boom(src, dest):
        raise OSError("disk full")

    code = execute(BASE, make_deps(rep, tar=boom))
    assert code == EXIT_UPLOAD
    assert "artifact stage failed" in rep.results[-1][1]["error"]


def test_non_utf8_dataset_bytes_are_a_data_fault_not_an_infra_fault():
    rep = SpyReporter()
    code = execute(BASE, make_deps(rep, fetch=lambda url: b"\xff\xfe garbage"))
    assert code == EXIT_DATA
    assert rep.results[-1][0] == "failed"
    assert "dataset" in rep.results[-1][1]["error"]


def test_report_carries_engine_and_fallback_reason_when_present():
    rep = SpyReporter()
    engine_train = lambda s, tr, ev, d, cb: {
        "baseline": {"eval_loss": 2.0, "perplexity": 7.4, "eval_tokens": 10},
        "final": {"eval_loss": 1.0, "perplexity": 2.7, "eval_tokens": 10},
        "steps": 5, "seed": 1, "device": "cuda", "dtype": "torch.bfloat16",
        "engine": "vanilla", "engine_fallback_reason": "ImportError"}
    assert execute(BASE, make_deps(rep, train=engine_train)) == EXIT_OK
    payload = rep.results[-1][1]
    assert payload["engine"] == "vanilla"
    assert payload["engine_fallback_reason"] == "ImportError"


def test_report_engine_fields_are_none_when_train_outcome_omits_them():
    rep = SpyReporter()
    assert execute(BASE, make_deps(rep)) == EXIT_OK
    payload = rep.results[-1][1]
    assert payload["engine"] is None
    assert payload["engine_fallback_reason"] is None


def test_infra_error_from_training_is_exit_infra_not_exit_train():
    rep = SpyReporter()

    def boom(*a):
        raise InfraError("CUDA is unavailable on this box")

    code = execute(BASE, make_deps(rep, train=boom))
    assert code == EXIT_INFRA
    assert "CUDA is unavailable" in rep.results[-1][1]["error"]


def test_failed_training_error_carries_the_engine_label_when_train_raised_it():
    # main.py can't know which engine was running when train() raised — it
    # never gets an `outcome` dict back, only the exception. trainer.py's
    # raise sites embed "[engine=...]" in the message itself (see
    # runner/trainer.py: require_engine_for_recipe, _load_vanilla,
    # _save_vanilla, _attempt_engine, trainer_unsloth.save_unsloth) precisely
    # so this string survives unmodified all the way into the failed-result
    # payload the buyer/backend sees.
    rep = SpyReporter()

    def boom_infra(*a):
        raise InfraError("[engine=vanilla] artifact save failed: boom")

    code = execute(BASE, make_deps(rep, train=boom_infra))
    assert code == EXIT_INFRA
    assert "[engine=vanilla]" in rep.results[-1][1]["error"]

    rep2 = SpyReporter()

    def boom_generic(*a):
        raise RuntimeError("[engine=unsloth] dataset produced no usable sequences after tokenization")

    code2 = execute(BASE, make_deps(rep2, train=boom_generic))
    assert code2 == EXIT_TRAIN
    assert "[engine=unsloth]" in rep2.results[-1][1]["error"]
