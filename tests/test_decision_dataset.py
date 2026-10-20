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


def _split_index(parts, state):
    return next(index for index, part in enumerate(parts)
                if any(candidate["state"] == state for candidate in part))


def test_duplicate_content_cannot_leak_when_gold_is_changed():
    first = row(1)
    duplicate = json.loads(json.dumps(first))
    duplicate["gold"]["route"] = "b"
    rows = [first, duplicate, row(2), row(3), row(4)]
    parts = split_three(rows, 0.2, 9)
    containing = [index for index, part in enumerate(parts)
                  if sum(candidate["state"] == first["state"] for candidate in part)]
    assert containing == [_split_index(parts, first["state"])]
    assert sum(candidate["state"] == first["state"] for candidate in parts[containing[0]]) == 2


def test_explicit_group_key_keeps_provider_or_service_family_in_one_split():
    family = [row(10), row(11), row(12)]
    for candidate in family:
        candidate["group_key"] = "provider-a/service-family"
    rows = family + [row(20), row(21), row(22), row(23)]
    parts = split_three(rows, 0.25, 3407)
    locations = {index for index, part in enumerate(parts)
                 for candidate in part if candidate.get("group_key") == "provider-a/service-family"}
    assert len(locations) == 1
    assert sum(len(part) for part in parts) == len(rows)


def test_group_id_alias_is_accepted_and_reported_in_group_count():
    rows = [row(1), row(2), row(3), row(4)]
    rows[0]["group_id"] = rows[1]["group_id"] = "same-family"
    parsed, stats = load_decision_jsonl("\n".join(json.dumps(candidate) for candidate in rows))
    assert len(parsed) == 4
    assert stats.groups == 3


def test_three_rows_with_identical_content_are_not_three_independent_splits():
    same = [row(1), row(1), row(1)]
    with pytest.raises(DecisionDatasetError, match="3 independent groups"):
        split_three(same, 0.2, 1)


def test_conflicting_or_malformed_group_keys_are_rejected():
    candidate = row()
    candidate.update(group_key="a", group_id="b")
    with pytest.raises(DecisionDatasetError, match="must match"):
        validate_row(candidate)
    candidate = row()
    candidate["group_key"] = "   "
    with pytest.raises(DecisionDatasetError, match="group key"):
        validate_row(candidate)


def test_exact_duplicates_are_removed_before_training_or_evaluation():
    duplicate = row(1)
    rows = [duplicate, json.loads(json.dumps(duplicate)), row(2), row(3), row(4)]
    parts = split_three(rows, 0.2, 4)
    flattened = [candidate for part in parts for candidate in part]
    assert len(flattened) == 4
    assert sum(candidate["state"] == "request 1" for candidate in flattened) == 1


def test_duplicate_content_bridges_different_supplied_families_transitively():
    bridge_a = row(1)
    bridge_a["group_key"] = "provider-a"
    bridge_b = json.loads(json.dumps(bridge_a))
    bridge_b["group_key"] = "provider-b"
    related_a, related_b = row(2), row(3)
    related_a["group_key"] = "provider-a"
    related_b["group_key"] = "provider-b"
    rows = [bridge_a, bridge_b, related_a, related_b, row(4), row(5), row(6)]
    parts = split_three(rows, 0.2, 8)
    locations = {
        index for index, part in enumerate(parts) for candidate in part
        if candidate.get("group_key") in {"provider-a", "provider-b"}
    }
    assert len(locations) == 1
