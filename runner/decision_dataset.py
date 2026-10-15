"""Strict, content-safe validation for Clef decision datasets.

Rows are never included in an exception or log.  The caller gets a row number
and a stable reason only, which is enough to repair a dataset without leaking
buyer inputs into provider logs.
"""
from __future__ import annotations

import json
import hashlib
import random
from dataclasses import dataclass


class DecisionDatasetError(ValueError):
    pass


@dataclass(frozen=True)
class DecisionDatasetStats:
    total: int
    questions: int
    groups: int
    deduplicated: int


def _nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_question(name: object, question: object, gold: object) -> None:
    if not _nonempty_text(name):
        raise DecisionDatasetError("question name must be a non-empty string")
    if not isinstance(question, dict):
        raise DecisionDatasetError("question must be an object")
    kind = question.get("type")
    if kind not in {"choice", "noul", "score"}:
        raise DecisionDatasetError("question type must be choice, noul, or score")
    if not _nonempty_text(question.get("instructions")):
        raise DecisionDatasetError("question instructions must be non-empty")

    criteria = question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise DecisionDatasetError("choice criteria must contain at least two options")
        if any(not _nonempty_text(k) or not _nonempty_text(v) for k, v in criteria.items()):
            raise DecisionDatasetError("choice criteria keys and descriptions must be non-empty strings")
        if not isinstance(gold, str) or gold not in criteria:
            raise DecisionDatasetError("choice gold must name one criterion")
    elif kind == "noul":
        if not isinstance(gold, bool):
            raise DecisionDatasetError("noul gold must be a boolean")
        if criteria is not None:
            if not isinstance(criteria, dict) or set(criteria) - {"true", "false"}:
                raise DecisionDatasetError("noul criteria may only describe true and false")
            if any(not _nonempty_text(v) for v in criteria.values()):
                raise DecisionDatasetError("noul criteria descriptions must be non-empty strings")
    else:
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise DecisionDatasetError("score criteria must contain at least two ordered levels")
        if any(not _nonempty_text(v) for v in criteria):
            raise DecisionDatasetError("score criteria must be non-empty strings")
        if isinstance(gold, bool) or not isinstance(gold, int) or not 0 <= gold < len(criteria):
            raise DecisionDatasetError("score gold must be a valid zero-based level")


def validate_row(row: object) -> int:
    if not isinstance(row, dict):
        raise DecisionDatasetError("row must be an object")
    if "state" not in row or row["state"] is None:
        raise DecisionDatasetError("state is required")
    if isinstance(row["state"], str) and not row["state"].strip():
        raise DecisionDatasetError("state string must be non-empty")
    if row.get("images") is not None:
        raise DecisionDatasetError("images are not supported by this runner kind")
    supplied_groups = [row.get(name) for name in ("group_key", "group_id")
                       if row.get(name) is not None]
    if any(not _nonempty_text(value) or len(value) > 256 for value in supplied_groups):
        raise DecisionDatasetError("group key must be a non-empty string of at most 256 characters")
    if len(set(supplied_groups)) > 1:
        raise DecisionDatasetError("group_key and group_id must match when both are present")
    questions, gold = row.get("questions"), row.get("gold")
    if not isinstance(questions, dict) or not questions:
        raise DecisionDatasetError("questions must be a non-empty object")
    if not isinstance(gold, dict):
        raise DecisionDatasetError("gold must be an object")
    if set(gold) != set(questions):
        raise DecisionDatasetError("gold keys must exactly match question keys")
    for name, question in questions.items():
        _validate_question(name, question, gold[name])
    return len(questions)


def load_decision_jsonl(text: str) -> tuple[list[dict], DecisionDatasetStats]:
    if not isinstance(text, str):
        raise DecisionDatasetError("dataset must be UTF-8 JSONL")
    rows: list[dict] = []
    questions = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise DecisionDatasetError(f"row {line_number}: invalid JSON") from None
        try:
            questions += validate_row(row)
        except DecisionDatasetError as exc:
            raise DecisionDatasetError(f"row {line_number}: {exc}") from None
        rows.append(row)
    if len(rows) < 3:
        raise DecisionDatasetError("dataset needs at least 3 valid rows for train, holdout, and test")
    groups = _group_records(rows)
    unique = sum(len(group) for group in groups.values())
    return rows, DecisionDatasetStats(
        total=len(rows), questions=questions, groups=len(groups),
        deduplicated=len(rows) - unique)


def _content_key(row: dict) -> str:
    canonical = json.dumps(
        {"state": row["state"], "questions": row["questions"]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "content:" + hashlib.sha256(canonical).hexdigest()


def _full_key(row: dict) -> str:
    canonical = json.dumps(
        {"state": row["state"], "questions": row["questions"], "gold": row["gold"]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _group_records(rows: list[dict]) -> dict[str, list[dict]]:
    """Connected components over explicit family keys and canonical content."""
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            low, high = sorted((a, b))
            parent[high] = low

    row_keys: list[list[str]] = []
    for row in rows:
        keys = [_content_key(row)]
        supplied = row.get("group_key", row.get("group_id"))
        if supplied is not None:
            keys.append("supplied:" + str(supplied))
            union(keys[0], keys[1])
        else:
            find(keys[0])
        row_keys.append(keys)

    grouped: dict[str, list[dict]] = {}
    seen: set[str] = set()
    for row, keys in zip(rows, row_keys):
        root = min(find(key) for key in keys)
        duplicate = _full_key(row)
        if duplicate in seen:
            continue
        seen.add(duplicate)
        grouped.setdefault(root, []).append(row)
    return grouped


def split_three(rows: list[dict], fraction: float, seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    """Deterministic group-level train / calibration-holdout / untouched-test split.

    Explicit group keys keep provider/service families together. Without one,
    canonical state+questions content is hashed so duplicates with changed labels
    cannot leak across splits.
    """
    if len(rows) < 3:
        raise DecisionDatasetError("dataset needs at least 3 rows to split")
    grouped = _group_records(rows)
    if len(grouped) < 3:
        raise DecisionDatasetError(
            "dataset needs at least 3 independent groups for train, holdout, and test")
    keys = sorted(grouped)
    random.Random(seed).shuffle(keys)
    unique_rows = sum(len(group) for group in grouped.values())
    target = max(1, int(round(unique_rows * fraction)))

    def take(remaining: list[str], leave: int) -> list[str]:
        selected, count = [], 0
        while len(remaining) > leave and (not selected or count < target):
            key = remaining.pop(0)
            selected.append(key)
            count += len(grouped[key])
        return selected

    test_keys = take(keys, leave=2)
    holdout_keys = take(keys, leave=1)
    train_keys = keys

    def expand(selected: list[str]) -> list[dict]:
        return [row for key in selected for row in grouped[key]]

    train, holdout, test = expand(train_keys), expand(holdout_keys), expand(test_keys)
    return train, holdout, test
