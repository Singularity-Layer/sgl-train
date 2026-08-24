"""Dataset loading and validation. Torch-free.

NEVER log a row's content — it is buyer data. Counts and reason codes only.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

ROLES = {"system", "user", "assistant"}


@dataclass(frozen=True)
class Example:
    messages: list[dict]


@dataclass(frozen=True)
class DatasetStats:
    total: int
    kept: int
    dropped: int
    reasons: dict[str, int] = field(default_factory=dict)


def validate_row(obj: object) -> str | None:
    """Return a reason code when the row is unusable, else None."""
    if not isinstance(obj, dict):
        return "not_an_object"
    messages = obj.get("messages")
    if not isinstance(messages, list) or not messages:
        return "no_messages"
    saw_assistant = False
    for m in messages:
        if not isinstance(m, dict):
            return "bad_message"
        role = m.get("role")
        if role not in ROLES:
            return "bad_role"
        content = m.get("content")
        if not isinstance(content, str) or not content.strip():
            return "empty_content"
        if role == "assistant":
            saw_assistant = True
    if not saw_assistant:
        return "no_assistant_turn"
    return None


def load_jsonl(text: str) -> tuple[list[Example], DatasetStats]:
    kept: list[Example] = []
    reasons: dict[str, int] = {}
    total = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            reasons["bad_json"] = reasons.get("bad_json", 0) + 1
            continue
        reason = validate_row(obj)
        if reason is not None:
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        kept.append(Example(messages=obj["messages"]))
    return kept, DatasetStats(total=total, kept=len(kept),
                              dropped=total - len(kept), reasons=reasons)


def split(examples: list[Example], eval_split: float, seed: int
          ) -> tuple[list[Example], list[Example]]:
    """Deterministic held-out split. Same seed, same rows, always.

    Requires at least 2 examples to guarantee both train and eval sets have
    at least one row. Raises ValueError if len(examples) < 2.

    Guarantees at least one eval row (an eval set of zero would make the run
    unmeasurable, which is worse than a small train set).
    """
    if len(examples) < 2:
        raise ValueError("need at least 2 examples to split")
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    n_eval = max(1, int(round(len(shuffled) * eval_split))) if shuffled else 0
    n_eval = min(n_eval, max(0, len(shuffled) - 1))
    return shuffled[n_eval:], shuffled[:n_eval]
