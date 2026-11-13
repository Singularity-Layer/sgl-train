import json
import sys

import pytest

from runner.bootstrap import RuntimeNotReady, prepare_runtime, run, runtime_probe


def test_prepare_runtime_requires_exact_marker(tmp_path):
    env = {"SGL_RUNTIME_VERSION": "v0.5.0", "SGL_RUNTIME_ROOT": str(tmp_path)}
    with pytest.raises(RuntimeNotReady, match="marker"):
        prepare_runtime(env)
    (tmp_path / "READY.json").write_text(json.dumps({"schema": 1, "version": "v0.4.0"}))
    with pytest.raises(RuntimeNotReady, match="does not match"):
        prepare_runtime(env)


def test_prepare_runtime_wires_exact_cached_paths(tmp_path):
    site = tmp_path / "site-packages"
    site.mkdir()
    (tmp_path / "READY.json").write_text(json.dumps({"schema": 1, "version": "v0.5.0"}))
    env = {
        "SGL_RUNTIME_VERSION": "v0.5.0",
        "SGL_RUNTIME_ROOT": str(tmp_path),
        "SGL_WORKDIR": str(tmp_path / "work"),
        "HF_HOME": str(tmp_path / "hf"),
        "PATH": "/usr/bin",
        "LD_LIBRARY_PATH": "",
    }
    old_path = list(sys.path)
    try:
        assert prepare_runtime(env) == tmp_path
        assert sys.path[0] == str(site)
        assert env["SGL_LLAMA_ROOT"] == str(tmp_path / "llama.cpp")
        assert env["PATH"].startswith(str(tmp_path / "bin"))
        assert (tmp_path / "work").is_dir()
        assert (tmp_path / "hf").is_dir()
    finally:
        sys.path[:] = old_path


def test_run_fails_closed_before_entrypoint_when_cache_is_absent(monkeypatch):
    monkeypatch.setenv("SGL_RUNTIME_VERSION", "v0.5.0")
    monkeypatch.setenv("SGL_RUNTIME_ROOT", "/definitely/missing")
    called = False

    def entrypoint():
        nonlocal called
        called = True
        return 0

    assert run(entrypoint) == 6
    assert called is False


def test_runtime_probe_fails_closed_without_the_gpu_stack(capsys):
    assert runtime_probe() == 6
    assert '"probe": "failed"' in capsys.readouterr().err


class _FakeReporter:
    sent: list = []

    def __init__(self, progress_url, result_url, token, run_id):
        self.args = (progress_url, result_url, token, run_id)

    def _send(self, url, body, attempts):
        _FakeReporter.sent.append((url, body, attempts, self.args[2]))
        return True


def test_boot_ping_posts_one_booting_progress_with_the_run_token():
    from runner.bootstrap import boot_ping
    _FakeReporter.sent = []
    raw = {"run_id": "r1", "progress_url": "https://x.example/training/executor/progress",
           "result_url": "https://x.example/training/executor/result", "token": "tok"}
    assert boot_ping(raw, reporter_cls=_FakeReporter) is True
    assert _FakeReporter.sent == [("https://x.example/training/executor/progress",
                                   {"run_id": "r1", "phase": "booting", "pct": 0.0}, 1, "tok")]


def test_boot_ping_never_posts_to_an_unsafe_or_missing_url_and_never_raises():
    from runner.bootstrap import boot_ping
    _FakeReporter.sent = []
    for raw in ({"run_id": "r1", "progress_url": "http://evil.example/p", "token": "t"},
                {"run_id": "r1", "progress_url": "https://x.example/p"},
                {"progress_url": "https://x.example/p", "token": "t"},
                {"run_id": "r1", "progress_url": 5, "token": "t"}):
        assert boot_ping(raw, reporter_cls=_FakeReporter) is False
    assert _FakeReporter.sent == []

    class Boom(_FakeReporter):
        def _send(self, url, body, attempts):
            raise OSError("network down")
    assert boot_ping({"run_id": "r1", "progress_url": "https://x.example/p", "token": "t"}, reporter_cls=Boom) is False


def test_run_pings_before_loading_the_heavy_libraries(monkeypatch, capsys):
    import runner.bootstrap as b
    import runner.main as m
    order = []
    monkeypatch.setattr(b, "prepare_runtime", lambda env=None: None)
    monkeypatch.setattr(b, "boot_ping", lambda raw, reporter_cls=None: order.append("ping") or True)
    monkeypatch.setattr(b, "log_gpu_probe", lambda: order.append("probe"))
    monkeypatch.setattr(m, "_default_deps", lambda: order.append("load") or object())
    monkeypatch.setattr(m, "execute", lambda raw, deps: order.append("execute") or 0)
    monkeypatch.setenv("SGL_JOB_SPEC", json.dumps({"run_id": "r1"}))
    monkeypatch.delenv("SGL_RUNTIME_PROBE", raising=False)
    assert b.run() == 0
    assert order == ["ping", "load", "probe", "execute"]
    assert "[sgl] runtime libraries loaded in" in capsys.readouterr().out


def test_run_still_rejects_a_bad_spec_before_any_ping(monkeypatch):
    import runner.bootstrap as b
    pinged = []
    monkeypatch.setattr(b, "prepare_runtime", lambda env=None: None)
    monkeypatch.setattr(b, "boot_ping", lambda raw, reporter_cls=None: pinged.append(raw))
    monkeypatch.setenv("SGL_JOB_SPEC", "not json")
    monkeypatch.delenv("SGL_RUNTIME_PROBE", raising=False)
    assert b.run() == 2
    assert pinged == []
