"""Strict, content-safe validation for Clef decision datasets.

Rows are never included in an exception or log.  The caller gets a row number
and a stable reason only, which is enough to repair a dataset without leaking
buyer inputs into provider logs.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass


class DecisionDatasetError(ValueError):
    pass


@dataclass(frozen=True)
class DecisionDatasetStats:
    total: int
    questions: int


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
    return rows, DecisionDatasetStats(total=len(rows), questions=questions)


def split_three(rows: list[dict], fraction: float, seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    """Deterministic row-level train / calibration-holdout / untouched-test split."""
    if len(rows) < 3:
        raise DecisionDatasetError("dataset needs at least 3 rows to split")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    requested = max(1, int(round(len(shuffled) * fraction)))
    n_test = min(requested, len(shuffled) - 2)
    n_holdout = min(requested, len(shuffled) - n_test - 1)
    test = shuffled[:n_test]
    holdout = shuffled[n_test:n_test + n_holdout]
    train = shuffled[n_test + n_holdout:]
    return train, holdout, test
