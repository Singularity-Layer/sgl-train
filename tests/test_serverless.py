from runner.serverless import handle_job


def test_rejects_malformed_serverless_envelopes_without_starting_training():
    called = []

    def run(_spec):
        called.append(True)
        return 0

    for job in (None, [], {}, {"input": None}, {"input": {}}, {"input": {"spec": []}}):
        result = handle_job(job, run)
        assert result["ok"] is False
        assert result["exit_code"] == 2
    assert called == []


def test_runs_the_existing_training_contract_and_returns_only_terminal_metadata():
    seen = []

    def run(spec):
        seen.append(spec)
        return 0

    spec = {"run_id": "run-123", "token": "must-not-be-returned", "dataset_url": "https://private"}
    result = handle_job({"input": {"spec": spec}}, run)
    assert seen == [spec]
    assert result == {"ok": True, "exit_code": 0, "run_id": "run-123"}
    assert "token" not in result


def test_preserves_nonzero_runner_exit_codes_for_backend_reconciliation():
    result = handle_job({"input": {"spec": {"run_id": "run-failed"}}}, lambda _spec: 6)
    assert result == {"ok": False, "exit_code": 6, "run_id": "run-failed"}
