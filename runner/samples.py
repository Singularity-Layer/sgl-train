"""Before/after sample answers. Torch-free: the model work is injected.

Sample answers let a buyer read what changed, side by side, instead of
trusting one loss number. They are strictly best-effort: nothing here may
fail a run, change its exit code, or hold up the adapter.

NEVER log a row's content: prompts and answers are buyer data.
"""
from __future__ import annotations

import hashlib
import json

MAX_SAMPLES = 40
MAX_NEW_TOKENS = 256
MAX_CHARS = 2000
SHORT_LABEL_CHARS = 32
# Two model loads plus up to 80 greedy generations. Used by the budget gate:
# samples start only when this much time is left on top of the reserve.
ESTIMATE_SECONDS = 300


def pick(rows, limit: int = MAX_SAMPLES) -> list[tuple[list[dict], str]]:
    """The first `limit` rows of the TEST split, in split order (the split is
    already a seeded shuffle, so this is deterministic and unbiased). The
    prompt is every message before the LAST assistant turn; the reference is
    that last assistant turn. A row whose last assistant turn is its first
    message has no prompt and is skipped."""
    out: list[tuple[list[dict], str]] = []
    for r in rows:
        msgs = list(r.messages)
        last = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=None)
        if not last:   # None (no assistant) or 0 (nothing to prompt with)
            continue
        out.append((msgs[:last], str(msgs[last]["content"])))
        if len(out) >= limit:
            break
    return out


def render_prompt(messages: list[dict]) -> str:
    if len(messages) == 1:
        return str(messages[0].get("content", ""))
    return "\n\n".join(f"{m.get('role')}: {m.get('content', '')}" for m in messages)


def _trim(text: object) -> str:
    return str(text if text is not None else "")[:MAX_CHARS]


def build(picked: list[tuple[list[dict], str]], before: list[str], after: list[str]) -> tuple[bytes, list[dict]]:
    if not (len(picked) == len(before) == len(after)):
        raise ValueError("sample answer count does not match the prompts")
    lines = [{"prompt": _trim(render_prompt(msgs)), "reference": _trim(ref),
              "before": _trim(b), "after": _trim(a)}
             for (msgs, ref), b, a in zip(picked, before, after)]
    data = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines).encode("utf-8")
    return data, lines


def _norm(text: object) -> str:
    return str(text if text is not None else "").strip().lower()


def exact_match(references: list[str], before: list[str], after: list[str]) -> dict | None:
    """Only meaningful when the references are short labels (classification,
    extraction). Free-text references would score ~0 for any model and say
    nothing, so then there is no number at all rather than a misleading one."""
    n = len(references)
    if n == 0 or any(len(str(r).strip()) > SHORT_LABEL_CHARS for r in references):
        return None
    refs = [_norm(r) for r in references]
    return {
        "exact_match_before": sum(_norm(b) == r for r, b in zip(refs, before)) / n,
        "exact_match_after": sum(_norm(a) == r for r, a in zip(refs, after)) / n,
        "n": n,
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
