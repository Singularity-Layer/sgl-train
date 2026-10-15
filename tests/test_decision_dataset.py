import json

import pytest

from runner.decision_dataset import (
    DecisionDatasetError,
    load_decision_jsonl,
    split_three,
    validate_row,
)


def row(i=0):
    return {
        "state": f"request {i}",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which endpoint should be called?",
                "criteria": {"a": "first endpoint", "b": "second endpoint"},
            },
            "pay": {"type": "noul", "instructions": "Should it be paid?"},
            "risk": {
                "type": "score",
                "instructions": "How risky is it?",
                "criteria": ["low", "medium", "high"],
            },
        },
        "gold": {"route": "a", "pay": True, "risk": 1},
    }


def test_accepts_all_documented_decision_types():
    assert validate_row(row()) == 3


def test_accepts_structured_json_state_for_system_one_inputs():
    candidate = row()
    candidate["state"] = {"intent": "image generation", "budget_usd": 0.05}
    assert validate_row(candidate) == 3


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda r: r.pop("state"), "state"),
        (lambda r: r.update(images=["secret.png"]), "images"),
        (lambda r: r.update(questions={}), "questions"),
        (lambda r: r["gold"].pop("route"), "exactly match"),
        (lambda r: r["gold"].update(route="missing"), "choice gold"),
        (lambda r: r["gold"].update(pay="true"), "boolean"),
        (lambda r: r["gold"].update(risk=9), "zero-based"),
    ],
)
def test_strict_validation_rejects_malformed_rows_without_echoing_content(mutate, reason):
    candidate = row()
    candidate["state"] = "PRIVATE BUYER CONTENT"
    mutate(candidate)
    with pytest.raises(DecisionDatasetError, match=reason) as caught:
        validate_row(candidate)
    assert "PRIVATE BUYER CONTENT" not in str(caught.value)


def test_jsonl_is_fail_closed_instead_of_silently_dropping_bad_rows():
    text = json.dumps(row(1)) + "\nnot-json\n" + json.dumps(row(2))
    with pytest.raises(DecisionDatasetError, match="row 2: invalid JSON"):
        load_decision_jsonl(text)


def test_requires_three_rows_for_three_way_evaluation():
    text = "\n".join(json.dumps(row(i)) for i in range(2))
    with pytest.raises(DecisionDatasetError, match="at least 3"):
        load_decision_jsonl(text)


def test_three_way_split_is_deterministic_disjoint_and_nonempty():
    rows = [row(i) for i in range(20)]
    first = split_three(rows, 0.2, 3407)
    second = split_three(rows, 0.2, 3407)
    assert first == second
    train, holdout, test = first
    assert (len(train), len(holdout), len(test)) == (12, 4, 4)
    states = [set(r["state"] for r in part) for part in first]
    assert not states[0] & states[1]
    assert not states[0] & states[2]
    assert not states[1] & states[2]


def test_three_rows_still_produce_one_row_per_split():
    assert [len(part) for part in split_three([row(i) for i in range(3)], 0.3, 1)] == [1, 1, 1]
