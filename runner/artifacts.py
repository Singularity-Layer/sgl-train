"""Local artifact layout and presigned uploads. Torch-free, stdlib only.

The runner never sees an R2 credential — it is handed presigned PUT URLs, the same
posture as agent backups where the worker never touches the bytes either.
"""
from __future__ import annotations

import os
import re
import tarfile
import urllib.error
import urllib.request

WORKDIR = os.environ.get("SGL_WORKDIR", "/opt/sgl/work")

_RUN_ID_RE = re.compile(r"[A-Za-z0-9_-]+")


def paths(run_id: str) -> dict[str, str]:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError("invalid run_id")
    root = os.path.join(WORKDIR, run_id)
    return {
        "root": root,
        "adapter_dir": os.path.join(root, "adapter"),
        "merged_dir": os.path.join(root, "merged"),
        "adapter_tar": os.path.join(root, "adapter.tar.gz"),
        "merged_tar": os.path.join(root, "merged.tar.gz"),
        "gguf": os.path.join(root, "model.gguf"),
        "report": os.path.join(root, "report.json"),
        "checkpoint_a_dir": os.path.join(root, "checkpoint-a"),
        "checkpoint_b_dir": os.path.join(root, "checkpoint-b"),
        "checkpoint_a_tar": os.path.join(root, "checkpoint-a.tar.gz"),
        "checkpoint_b_tar": os.path.join(root, "checkpoint-b.tar.gz"),
        "checkpoint_manifest": os.path.join(root, "checkpoint-manifest.json"),
        "resume_archive": os.path.join(root, "resume-checkpoint.tar.gz"),
        "resume_dir": os.path.join(root, "resume-checkpoint"),
    }


def tar_dir(src_dir: str, dest_tar: str) -> str:
    os.makedirs(os.path.dirname(dest_tar), exist_ok=True)
    with tarfile.open(dest_tar, "w:gz") as t:
        t.add(src_dir, arcname=".")
    return dest_tar


def _urllib_put(url: str, path: str, headers: dict[str, str]) -> int:
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        req = urllib.request.Request(url, data=fh, headers={**headers, "Content-Length": str(size)}, method="PUT")
        try:
            with urllib.request.urlopen(req, timeout=600) as res:
                return res.status
        except urllib.error.HTTPError as e:
            return e.code
        except Exception:
            return 599


def put_file(url: str, path: str, put=None) -> bool:
    # Streamed, not read into RAM: adapter/gguf artifacts can be gigabytes and this
    # runs on the same box that just held a model in memory for training.
    sender = put or _urllib_put
    status = sender(url, path, {"Content-Type": "application/octet-stream"})
    return 200 <= status < 300
