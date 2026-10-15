import json
from runner.report import Reporter, backoff_delays


class FakePost:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.calls = []

    def __call__(self, url, body, headers):
        self.calls.append((url, json.loads(body.decode()), headers))
        return self.statuses.pop(0) if self.statuses else 200


def test_progress_posts_to_progress_url_with_auth():
    post = FakePost([200])
    r = Reporter("https://p.invalid", "https://r.invalid", "tok", "run-1", post=post)
    assert r.progress("training", 0.5, step=10) is True
    url, body, headers = post.calls[0]
    assert url == "https://p.invalid"
    assert body["run_id"] == "run-1"
    assert body["phase"] == "training"
    assert body["pct"] == 0.5
    assert body["step"] == 10
    assert headers["Authorization"] == "Bearer tok"


def test_result_posts_to_result_url():
    post = FakePost([200])
    r = Reporter("https://p.invalid", "https://r.invalid", "tok", "run-1", post=post)
    assert r.result("completed", {"eval_loss": 1.5}) is True
    url, body, _ = post.calls[0]
    assert url == "https://r.invalid"
    assert body["status"] == "completed"
    assert body["eval_loss"] == 1.5


def test_canary_result_log_is_bounded_and_never_includes_token_or_buyer_fields(monkeypatch, capsys):
    monkeypatch.setenv("SGL_CANARY_LOG_RESULT", "1")
    reporter = Reporter("https://progress", "https://result", "secret-token", "run-1",
                        post=lambda *_: 200, sleep=lambda _: None)
    assert reporter.result("failed", {
        "error": "model load failed", "exit_code": 6,
        "gguf_error": "converter exited 1",
        "dataset": "buyer-private", "token": "must-not-log",
    }) is True
    logged = capsys.readouterr().err
    assert "model load failed" in logged
    assert "converter exited 1" in logged
    assert "buyer-private" not in logged
    assert "must-not-log" not in logged
    assert "secret-token" not in logged


def test_result_retries_on_5xx_then_succeeds():
    post = FakePost([500, 502, 200])
    r = Reporter("https://p.invalid", "https://r.invalid", "tok", "run-1",
                 post=post, sleep=lambda _: None)
    assert r.result("completed", {}) is True
    assert len(post.calls) == 3


def test_result_does_not_retry_on_4xx():
    post = FakePost([400])
    r = Reporter("https://p.invalid", "https://r.invalid", "tok", "run-1",
                 post=post, sleep=lambda _: None)
    assert r.result("completed", {}) is False
    assert len(post.calls) == 1


def test_progress_failure_is_not_fatal():
    post = FakePost([500, 500, 500, 500, 500, 500])
    r = Reporter("https://p.invalid", "https://r.invalid", "tok", "run-1",
                 post=post, sleep=lambda _: None)
    assert r.progress("training", 0.1) is False


def test_backoff_grows_and_is_bounded():
    d = backoff_delays(6, base=1.0)
    assert len(d) == 6
    assert d[0] < d[1] < d[2]
    assert all(x <= 60.0 for x in d)
