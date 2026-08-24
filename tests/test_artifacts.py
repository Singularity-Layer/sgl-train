import os
import tarfile

import pytest
from runner.artifacts import paths, tar_dir, put_file


def test_paths_are_scoped_to_the_run():
    p = paths("run-9")
    assert p["root"].endswith("/run-9")
    assert p["adapter_tar"].endswith("adapter.tar.gz")
    assert p["gguf"].endswith("model.gguf")
    assert p["report"].endswith("report.json")


def test_paths_rejects_a_path_traversal_run_id():
    with pytest.raises(ValueError):
        paths("../../etc/passwd")


def test_tar_dir_round_trips(tmp_path):
    src = tmp_path / "adapter"
    src.mkdir()
    (src / "a.bin").write_bytes(b"x" * 10)
    (src / "cfg.json").write_text("{}")
    dest = str(tmp_path / "out.tar.gz")
    tar_dir(str(src), dest)
    assert os.path.exists(dest)
    with tarfile.open(dest) as t:
        names = sorted(os.path.basename(n) for n in t.getnames() if n.strip("./"))
    assert "a.bin" in names and "cfg.json" in names


def test_put_file_streams_the_path_and_reports_success(tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"hello")
    seen = {}

    def fake_put(url, path, headers):
        seen["url"] = url
        seen["path"] = path
        with open(path, "rb") as fh:
            seen["data"] = fh.read()
        return 200

    assert put_file("https://x.invalid/put", str(f), put=fake_put) is True
    assert seen["url"] == "https://x.invalid/put"
    assert seen["path"] == str(f)
    assert seen["data"] == b"hello"


def test_put_file_returns_false_on_error(tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"hello")
    assert put_file("https://x.invalid/put", str(f), put=lambda *a: 403) is False
