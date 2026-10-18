"""Private resumable checkpoint publication with manifest-last atomicity."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, is_dataclass

from .artifacts import tar_dir
from .spec import ResumeCheckpoint

MAX_CHECKPOINT_MEMBERS = 10_000
MAX_CHECKPOINT_EXTRACTED_BYTES = 32 * 1024**3


def _jsonable(value):
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"checkpoint identity does not support {type(value).__name__}")


def canonical_sha256(value) -> str:
    """Hash structured runner values without relying on object reprs."""
    payload = json.dumps(
        value, default=_jsonable, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def checkpoint_provenance(spec, train_rows, eval_rows) -> dict[str, str]:
    return {
        "model_id": f"{spec.hf_repo}@{spec.hf_revision}",
        "dataset_sha256": canonical_sha256({"train": train_rows, "eval": eval_rows}),
        "recipe_sha256": canonical_sha256(spec.recipe),
    }


def validate_resume_provenance(resume: ResumeCheckpoint, provenance: dict[str, str],
                               planned_steps: int) -> None:
    expected = {
        "model_id": resume.model_id,
        "dataset_sha256": resume.dataset_sha256,
        "recipe_sha256": resume.recipe_sha256,
    }
    if expected != provenance:
        raise ValueError("resume checkpoint provenance does not match this model, dataset, or Training Recipe")
    if resume.planned_steps != planned_steps or resume.step > planned_steps:
        raise ValueError("resume checkpoint step plan does not match this Training Recipe")


def _sha256(path: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def upload_checkpoint(
    *, urls: dict[str, str], checkpoint_dir: str, archive_path: str,
    manifest_path: str, slot: str, step: int, epoch: int, planned_steps: int,
    model_id: str, dataset_sha256: str, recipe_sha256: str,
    upload: Callable[[str, str], bool],
) -> dict | None:
    if slot not in {"a", "b"}:
        raise ValueError("checkpoint slot must be a or b")
    tar_dir(checkpoint_dir, archive_path)
    archive_sha256, archive_bytes = _sha256(archive_path)
    if not upload(urls[slot], archive_path):
        return None
    manifest = {
        "version": 1,
        "slot": slot,
        "step": int(step),
        "epoch": int(epoch),
        "planned_steps": int(planned_steps),
        "archive_sha256": archive_sha256,
        "archive_bytes": archive_bytes,
        "model_id": model_id,
        "dataset_sha256": dataset_sha256,
        "recipe_sha256": recipe_sha256,
    }
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    temp = manifest_path + ".tmp"
    with open(temp, "w") as handle:
        json.dump(manifest, handle, sort_keys=True, separators=(",", ":"))
    os.replace(temp, manifest_path)
    if not upload(urls["manifest"], manifest_path):
        return None
    return {
        "checkpoint_step": int(step),
        "checkpoint_bytes": archive_bytes,
        "checkpoint_sha256": archive_sha256,
        "checkpoint_slot": slot,
        "checkpoint_epoch": int(epoch),
        "checkpoint_planned_steps": int(planned_steps),
    }


class CheckpointPublisher:
    """Publish to the slot opposite the last committed manifest."""

    def __init__(self, *, urls: dict[str, str], out_dirs: dict[str, str],
                 provenance: dict[str, str], upload: Callable[[str, str], bool],
                 active_slot: str | None = None):
        self.urls = urls
        self.out_dirs = out_dirs
        self.provenance = provenance
        self.upload = upload
        self.active_slot = active_slot

    def next_slot(self) -> str:
        return "b" if self.active_slot == "a" else "a"

    def __call__(self, slot: str, checkpoint_dir: str, step: int, epoch: int,
                 planned_steps: int) -> dict | None:
        if slot != self.next_slot():
            raise ValueError("checkpoint must target the slot opposite the committed manifest")
        result = upload_checkpoint(
            urls=self.urls,
            checkpoint_dir=checkpoint_dir,
            archive_path=self.out_dirs[f"checkpoint_{slot}_tar"],
            manifest_path=self.out_dirs["checkpoint_manifest"],
            slot=slot, step=step, epoch=epoch, planned_steps=planned_steps,
            upload=self.upload, **self.provenance,
        )
        if result is not None:
            self.active_slot = slot
        return result


def _download(url: str, path: str, expected_bytes: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with urllib.request.urlopen(url, timeout=600) as response, open(path, "wb") as output:
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) != expected_bytes:
            raise ValueError("resume checkpoint download size does not match its manifest")
        written = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            written += len(chunk)
            if written > expected_bytes:
                raise ValueError("resume checkpoint download exceeded its declared size")
            output.write(chunk)


def prepare_resume_checkpoint(
    resume: ResumeCheckpoint, archive_path: str, target_dir: str,
    download: Callable[[str, str, int], object] = _download,
) -> str:
    try:
        download(resume.archive_url, archive_path, resume.archive_bytes)
        digest, size = _sha256(archive_path)
        if size != resume.archive_bytes or digest != resume.archive_sha256:
            raise ValueError("resume checkpoint integrity mismatch")
        shutil.rmtree(target_dir, ignore_errors=True)
        os.makedirs(target_dir, exist_ok=True)
        root = os.path.realpath(target_dir)
        with tarfile.open(archive_path, "r:gz") as handle:
            members = handle.getmembers()
            if len(members) > MAX_CHECKPOINT_MEMBERS:
                raise ValueError("resume checkpoint archive contains too many files")
            extracted_bytes = 0
            expansion_limit = min(
                MAX_CHECKPOINT_EXTRACTED_BYTES,
                max(64 * 1024**2, resume.archive_bytes * 20),
            )
            for member in members:
                resolved = os.path.realpath(os.path.join(root, member.name))
                if (os.path.commonpath((root, resolved)) != root
                        or member.size < 0
                        or not (member.isfile() or member.isdir())
                        or member.issym() or member.islnk()):
                    raise ValueError("unsafe checkpoint archive member")
                extracted_bytes += member.size
                if extracted_bytes > expansion_limit:
                    raise ValueError("resume checkpoint archive expands beyond its safety limit")
            try:
                handle.extractall(target_dir, members=members, filter="data")
            except TypeError:  # Python <3.12; members were already validated above.
                handle.extractall(target_dir, members=members)
    except Exception:
        try:
            os.remove(archive_path)
        except FileNotFoundError:
            pass
        shutil.rmtree(target_dir, ignore_errors=True)
        raise
    return target_dir
