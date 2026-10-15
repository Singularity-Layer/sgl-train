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
