import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
from runner.checkpoint import prepare_resume_checkpoint, upload_checkpoint
from runner.spec import ResumeCheckpoint


def test_checkpoint_archive_uploads_before_atomic_manifest(tmp_path):
    checkpoint = tmp_path / "checkpoint-a"
    checkpoint.mkdir()
    (checkpoint / "adapter.bin").write_bytes(b"adapter")
    archive = tmp_path / "checkpoint-a.tar.gz"
    manifest_path = tmp_path / "manifest.json"
    events = []

    def upload(url, path):
        events.append((url, Path(path).name))
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
        upload=lambda url, path: events.append(url) or False,
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
    resume = ResumeCheckpoint(
        source_run_id="source-1", archive_url="https://x/checkpoint",
        archive_sha256=hashlib.sha256(payload).hexdigest(), archive_bytes=len(payload), step=10, epoch=1,
    )
    output = tmp_path / "download.tar.gz"
    target = tmp_path / "restored"
    prepare_resume_checkpoint(resume, str(output), str(target), download=lambda _url, path: Path(path).write_bytes(payload))
    assert (target / "training_state.pt").read_bytes() == b"state"


def test_resume_rejects_archive_path_traversal(tmp_path):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as handle:
        info = tarfile.TarInfo("../../escape")
        info.size = 1
        handle.addfile(info, io.BytesIO(b"x"))
    payload = stream.getvalue()
    resume = ResumeCheckpoint("source-1", "https://x/checkpoint", hashlib.sha256(payload).hexdigest(), len(payload), 1, 1)
    with pytest.raises(ValueError, match="unsafe checkpoint"):
        prepare_resume_checkpoint(
            resume, str(tmp_path / "x.tar.gz"), str(tmp_path / "out"),
            download=lambda _url, path: Path(path).write_bytes(payload),
        )
