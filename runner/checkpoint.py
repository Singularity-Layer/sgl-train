"""Private resumable checkpoint publication with manifest-last atomicity."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import urllib.request
from collections.abc import Callable

from .artifacts import tar_dir
from .spec import ResumeCheckpoint


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
    }


def _download(url: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with urllib.request.urlopen(url, timeout=600) as response, open(path, "wb") as output:
        shutil.copyfileobj(response, output, length=1 << 20)


def prepare_resume_checkpoint(
    resume: ResumeCheckpoint, archive_path: str, target_dir: str,
    download: Callable[[str, str], object] = _download,
) -> str:
    download(resume.archive_url, archive_path)
    digest, size = _sha256(archive_path)
    if size != resume.archive_bytes or digest != resume.archive_sha256:
        raise ValueError("resume checkpoint integrity mismatch")
    shutil.rmtree(target_dir, ignore_errors=True)
    os.makedirs(target_dir, exist_ok=True)
    root = os.path.realpath(target_dir)
    with tarfile.open(archive_path, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            resolved = os.path.realpath(os.path.join(root, member.name))
            if not (resolved == root or resolved.startswith(root + os.sep)) or member.issym() or member.islnk():
                raise ValueError("unsafe checkpoint archive member")
        try:
            handle.extractall(target_dir, members=members, filter="data")
        except TypeError:  # Python <3.12; members were already validated above.
            handle.extractall(target_dir, members=members)
    return target_dir
