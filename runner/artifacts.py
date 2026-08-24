"""Local artifact layout and presigned uploads. Torch-free, stdlib only.

The runner never sees an R2 credential — it is handed presigned PUT URLs, the same
posture as agent backups where the worker never touches the bytes either.
"""
from __future__ import annotations

import os
import tarfile
import urllib.error
import urllib.request

WORKDIR = "/opt/sgl/work"


def paths(run_id: str) -> dict[str, str]:
    root = os.path.join(WORKDIR, run_id)
    return {
        "root": root,
        "adapter_dir": os.path.join(root, "adapter"),
        "merged_dir": os.path.join(root, "merged"),
        "adapter_tar": os.path.join(root, "adapter.tar.gz"),
        "gguf": os.path.join(root, "model.gguf"),
        "report": os.path.join(root, "report.json"),
    }


def tar_dir(src_dir: str, dest_tar: str) -> str:
    os.makedirs(os.path.dirname(dest_tar), exist_ok=True)
    with tarfile.open(dest_tar, "w:gz") as t:
        t.add(src_dir, arcname=".")
    return dest_tar


def _urllib_put(url: str, data: bytes, headers: dict[str, str]) -> int:
    req = urllib.request.Request(url, data=data, headers=headers, method="PUT")
    try:
        with urllib.request.urlopen(req, timeout=600) as res:
            return res.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 599


def put_file(url: str, path: str, put=None) -> bool:
    sender = put or _urllib_put
    with open(path, "rb") as fh:
        data = fh.read()
    status = sender(url, data, {"Content-Type": "application/octet-stream"})
    return 200 <= status < 300
