from runner.dataset import validate_row, load_jsonl, split

GOOD = {"messages": [{"role": "user", "content": "hi"},
                     {"role": "assistant", "content": "hello"}]}


def test_accepts_a_good_row():
    assert validate_row(GOOD) is None


def test_rejects_non_object():
    assert validate_row([1, 2, 3]) == "not_an_object"


def test_rejects_missing_messages():
    assert validate_row({"text": "hi"}) == "no_messages"


def test_rejects_bad_role():
    row = {"messages": [{"role": "wizard", "content": "hi"},
                        {"role": "assistant", "content": "yo"}]}
    assert validate_row(row) == "bad_role"


def test_rejects_empty_content():
    row = {"messages": [{"role": "user", "content": "  "},
                        {"role": "assistant", "content": "yo"}]}
    assert validate_row(row) == "empty_content"


def test_rejects_no_assistant_turn():
    row = {"messages": [{"role": "user", "content": "hi"}]}
    assert validate_row(row) == "no_assistant_turn"


def test_load_jsonl_counts_and_drops():
    text = "\n".join([
        '{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"}]}',
        "not json",
        '{"messages":[{"role":"user","content":"e"}]}',
        '{"messages":[{"role":"user","content":"c"},{"role":"assistant","content":"d"}]}',
        "",
    ])
    rows, stats = load_jsonl(text)
    assert len(rows) == 2
    assert stats.total == 3
    assert stats.kept == 2
    assert stats.dropped == 1
    assert stats.reasons["bad_json"] == 1
    assert stats.reasons["no_assistant_turn"] == 1


def test_split_is_deterministic_for_a_seed():
    rows, _ = load_jsonl("\n".join(
        '{"messages":[{"role":"user","content":"%d"},{"role":"assistant","content":"x"}]}' % i
        for i in range(100)))
    a_train, a_eval = split(rows, 0.1, seed=7)
    b_train, b_eval = split(rows, 0.1, seed=7)
    assert [e.messages for e in a_eval] == [e.messages for e in b_eval]
    assert len(a_eval) == 10
    assert len(a_train) == 90
    c_train, c_eval = split(rows, 0.1, seed=8)
    assert [e.messages for e in c_eval] != [e.messages for e in a_eval]


def test_split_always_leaves_at_least_one_eval_row():
    rows, _ = load_jsonl(
        '{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"}]}\n'
        '{"messages":[{"role":"user","content":"c"},{"role":"assistant","content":"d"}]}')
    train, ev = split(rows, 0.05, seed=1)
    assert len(ev) == 1
    assert len(train) == 1
