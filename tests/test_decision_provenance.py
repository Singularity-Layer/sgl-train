import hashlib
from pathlib import Path


ROOT = Path(__file__).parents[1]
VENDOR = ROOT / "runner/_vendor/cloudflare_clef"


def test_vendored_cloudflare_runtime_matches_recorded_apache_source():
    source = VENDOR / "joint_schema_model.py"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"
    notice = (VENDOR / "NOTICE").read_text()
    assert "0b331204bb13fbd2ca93a64df1956f5b55478ce5" in notice
    assert digest in notice
    assert "Apache License" in (VENDOR / "LICENSE").read_text()


def test_decision_trainer_has_no_unsloth_decision_dependency():
    source = (ROOT / "runner/trainer_decision.py").read_text()
    assert "FastDecisionModel" not in source
    assert "DecisionTrainer" not in source
    assert "from unsloth" not in source
