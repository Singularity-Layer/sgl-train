"""Progress and result reporting. Torch-free, stdlib only.

Progress is best-effort: a dropped heartbeat must never kill a run that is
otherwise fine. The RESULT post is the one that matters, so it retries harder —
but only on 5xx. A 4xx means the backend rejected us and repeating will not help.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

PROGRESS_ATTEMPTS = 3
RESULT_ATTEMPTS = 6
MAX_DELAY = 60.0


def backoff_delays(attempts: int, base: float = 1.0) -> list[float]:
    """Exponential backoff with jitter, capped. Jitter stops a fleet of boxes
    from retrying in lockstep after a worker blip."""
    out = []
    for i in range(attempts):
        raw = min(MAX_DELAY, base * (2 ** i))
        out.append(round(raw * (0.5 + random.random() * 0.5), 3))
    return out


def _urllib_post(url: str, body: bytes, headers: dict[str, str]) -> int:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=20) as res:
            return res.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 599  # network-level failure — treated as retryable


class Reporter:
    def __init__(self, progress_url: str, result_url: str, token: str, run_id: str,
                 post=None, sleep=time.sleep):
        self.progress_url = progress_url
        self.result_url = result_url
        self.token = token
        self.run_id = run_id
        self._post = post or _urllib_post
        self._sleep = sleep

    def _send(self, url: str, body: dict, attempts: int) -> bool:
        payload = json.dumps(body).encode()
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.token}"}
        delays = backoff_delays(attempts)
        for i in range(attempts):
            status = self._post(url, payload, headers)
            if 200 <= status < 300:
                return True
            if 400 <= status < 500:
                return False  # our fault, not transient
            if i < attempts - 1:
                self._sleep(delays[i])
        return False

    def progress(self, phase: str, pct: float, **extra) -> bool:
        body = {"run_id": self.run_id, "phase": phase, "pct": pct, **extra}
        return self._send(self.progress_url, body, PROGRESS_ATTEMPTS)

    def experiment(self, url: str, payload: dict) -> bool:
        """One auto-tune trial record. Best-effort like progress: the full
        trial list also rides the final result, so a dropped record loses
        nothing that matters."""
        body = {"run_id": self.run_id, **payload}
        return self._send(url, body, PROGRESS_ATTEMPTS)

    def result(self, status: str, payload: dict) -> bool:
        body = {"run_id": self.run_id, "status": status, **payload}
        if os.environ.get("SGL_CANARY_LOG_RESULT") == "1":
            allow = {
                "run_id", "status", "error", "exit_code", "engine",
                "engine_fallback_reason", "trained_repo", "steps", "gguf_ok",
                "gguf_error",
            }
            safe = {key: value for key, value in body.items() if key in allow}
            print(f"[sgl-canary-result] {json.dumps(safe, default=str)[:2000]}", file=sys.stderr, flush=True)
        return self._send(self.result_url, body, RESULT_ATTEMPTS)
