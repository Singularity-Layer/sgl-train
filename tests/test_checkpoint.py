import hashlib
import io
import json
import tarfile
import random
import sys
import types
from pathlib import Path

import pytest
from runner.checkpoint import (CheckpointPublisher, _download, canonical_sha256,
                               prepare_resume_checkpoint, upload_checkpoint,
                               validate_resume_provenance)
from runner.spec import ResumeCheckpoint


def _resume(payload: bytes, **over):
    values = dict(
        source_run_id="source-1", archive_url="https://x/checkpoint",
        archive_sha256=hashlib.sha256(payload).hexdigest(), archive_bytes=len(payload),
        step=10, epoch=1, planned_steps=20, slot="a", model_id="Q/M@sha",
        dataset_sha256="ab" * 32, recipe_sha256="cd" * 32,
    )
    values.update(over)
    return ResumeCheckpoint(**values)


def test_checkpoint_archive_uploads_before_atomic_manifest(tmp_path):
    checkpoint = tmp_path / "checkpoint-a"
    checkpoint.mkdir()
    (checkpoint / "adapter.bin").write_bytes(b"adapter")
    archive = tmp_path / "checkpoint-a.tar.gz"
    manifest_path = tmp_path / "manifest.json"
    events = []

    def upload(url, path, headers=None):
        events.append((url, Path(path).name))
        if url == "https://x/a":
            assert headers == {"x-amz-meta-sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        return True

    result = upload_checkpoint(
        urls={"a": "https://x/a", "b": "https://x/b", "manifest": "https://x/manifest"},
        checkpoint_dir=str(checkpoint), archive_path=str(archive), manifest_path=str(manifest_path),
        slot="a", step=20, epoch=1, planned_steps=100,
        model_id="Q/M@sha", dataset_sha256="ab" * 32, recipe_sha256="cd" * 32,
        upload=upload,
    )
    assert result is not None
    assert events == [("https://x/a", "checkpoint-a.tar.gz"), ("https://x/manifest", "manifest.json")]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["slot"] == "a"
    assert manifest["step"] == 20
    assert manifest["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert result == {
        "checkpoint_step": 20,
        "checkpoint_bytes": manifest["archive_bytes"],
        "checkpoint_sha256": manifest["archive_sha256"],
        "checkpoint_slot": "a",
        "checkpoint_epoch": 1,
        "checkpoint_planned_steps": 100,
    }


def test_failed_archive_never_publishes_manifest(tmp_path):
    checkpoint = tmp_path / "checkpoint-b"
    checkpoint.mkdir()
    (checkpoint / "state").write_text("x")
    events = []
    result = upload_checkpoint(
        urls={"a": "https://x/a", "b": "https://x/b", "manifest": "https://x/manifest"},
        checkpoint_dir=str(checkpoint), archive_path=str(tmp_path / "b.tar.gz"),
        manifest_path=str(tmp_path / "manifest.json"), slot="b", step=1, epoch=1,
        planned_steps=2, model_id="m", dataset_sha256="a" * 64, recipe_sha256="b" * 64,
        upload=lambda url, path, **kwargs: events.append(url) or False,
    )
    assert result is None
    assert events == ["https://x/b"]


def test_resume_download_verifies_and_extracts_checkpoint(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "training_state.pt").write_bytes(b"state")
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source, arcname=".")
    payload = archive.read_bytes()
    resume = _resume(payload)
    output = tmp_path / "download.tar.gz"
    target = tmp_path / "restored"
    prepare_resume_checkpoint(
        resume, str(output), str(target),
        download=lambda _url, path, _limit: Path(path).write_bytes(payload))
    assert (target / "training_state.pt").read_bytes() == b"state"


def test_resume_rejects_archive_path_traversal(tmp_path):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as handle:
        info = tarfile.TarInfo("../../escape")
        info.size = 1
        handle.addfile(info, io.BytesIO(b"x"))
    payload = stream.getvalue()
    resume = _resume(payload, step=1)
    with pytest.raises(ValueError, match="unsafe checkpoint"):
        prepare_resume_checkpoint(
            resume, str(tmp_path / "x.tar.gz"), str(tmp_path / "out"),
            download=lambda _url, path, _limit: Path(path).write_bytes(payload),
        )


def test_identity_hash_serializes_dataset_dataclasses_deterministically():
    from runner.dataset import Example

    value = {"train": [Example(messages=[{"role": "user", "content": "hello"}])], "eval": []}
    assert canonical_sha256(value) == canonical_sha256(value)
    assert len(canonical_sha256(value)) == 64


def test_publisher_uses_opposite_slot_and_advances_only_after_manifest_commit(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "state").write_text("ok")
    uploads = []
    fail_manifest_once = {"value": True}

    def upload(url, path, **kwargs):
        uploads.append(url)
        if url == "manifest" and fail_manifest_once["value"]:
            fail_manifest_once["value"] = False
            return False
        return True

    publisher = CheckpointPublisher(
        urls={"a": "slot-a", "b": "slot-b", "manifest": "manifest"},
        out_dirs={
            "checkpoint_a_tar": str(tmp_path / "a.tar.gz"),
            "checkpoint_b_tar": str(tmp_path / "b.tar.gz"),
            "checkpoint_manifest": str(tmp_path / "manifest.json"),
        },
        provenance={"model_id": "Q/M@sha", "dataset_sha256": "ab" * 32,
                    "recipe_sha256": "cd" * 32},
        upload=upload, active_slot="a",
    )
    assert publisher.next_slot() == "b"
    assert publisher("b", str(checkpoint_dir), 10, 1, 20) is None
    assert publisher.active_slot == "a"
    assert publisher.next_slot() == "b"
    result = publisher("b", str(checkpoint_dir), 10, 1, 20)
    assert result["checkpoint_slot"] == "b"
    assert publisher.active_slot == "b"
    assert publisher.next_slot() == "a"
    assert uploads == ["slot-b", "manifest", "slot-b", "manifest"]


def test_resume_provenance_rejects_a_different_dataset():
    resume = _resume(b"x", dataset_sha256="11" * 32)
    with pytest.raises(ValueError, match="provenance"):
        validate_resume_provenance(
            resume,
            {"model_id": resume.model_id, "dataset_sha256": "22" * 32,
             "recipe_sha256": resume.recipe_sha256},
            resume.planned_steps,
        )


def test_download_stops_before_writing_more_than_manifest_size(monkeypatch, tmp_path):
    class Response:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, _size):
            if hasattr(self, "done"):
                return b""
            self.done = True
            return b"12345"

    monkeypatch.setattr("runner.checkpoint.urllib.request.urlopen", lambda *_a, **_k: Response())
    with pytest.raises(ValueError, match="exceeded"):
        _download("https://x/checkpoint", str(tmp_path / "checkpoint"), 4)


def test_restore_uses_safe_weights_only_loading(monkeypatch, tmp_path):
    from runner import trainer

    root = tmp_path / "resume"
    (root / "adapter").mkdir(parents=True)
    (root / "training_state.pt").write_bytes(b"not-read-by-fake")
    provenance = {"model_id": "Q/M@sha", "dataset_sha256": "ab" * 32,
                  "recipe_sha256": "cd" * 32}
    resume = _resume(b"x", **provenance)
    captured = {}

    class Cuda:
        @staticmethod
        def set_rng_state_all(_state):
            pass

    torch = types.SimpleNamespace(
        load=lambda path, **kw: captured.update(kw) or {
            "step": 10, "epoch": 1, "planned_steps": 20, **provenance,
            "optimizer": {}, "python_rng": random.getstate(), "torch_rng": types.SimpleNamespace(cpu=lambda: b"rng"),
            "cuda_rng": None,
        },
        set_rng_state=lambda _state: None,
        cuda=Cuda(),
    )
    peft_save = types.ModuleType("peft.utils.save_and_load")
    peft_save.load_peft_weights = lambda *_a, **_k: {}
    peft_save.set_peft_model_state_dict = lambda *_a, **_k: types.SimpleNamespace(unexpected_keys=[])
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "peft", types.ModuleType("peft"))
    monkeypatch.setitem(sys.modules, "peft.utils", types.ModuleType("peft.utils"))
    monkeypatch.setitem(sys.modules, "peft.utils.save_and_load", peft_save)

    optimizer = types.SimpleNamespace(load_state_dict=lambda _state: None)
    trainer._restore_training_checkpoint(
        object(), optimizer, {"resume_dir": str(root)}, "cpu", 20, resume, provenance)
    assert captured["weights_only"] is True
