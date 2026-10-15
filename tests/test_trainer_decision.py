from pathlib import Path

import pytest

from runner.errors import InfraError
from runner.spec import parse_spec
from runner.trainer_decision import decision_metrics, fit_temperature, run_decision_finetune


BASE = {
    "run_id": "decision-1",
    "kind": "decision_finetune",
    "base_model": {
        "hf_repo": "Qwen/Qwen3.5-0.8B",
        "revision": "canonical-sha",
        "decision_repo": "Qwen/Qwen3.5-0.8B",
        "decision_revision": "decision-sha",
    },
    "dataset_url": "https://d.invalid/decisions.jsonl",
    "progress_url": "https://p.invalid",
    "result_url": "https://r.invalid",
    "token": "t",
    "artifact_urls": {
        "adapter": "https://a.invalid",
        "merged": "https://m.invalid",
        "report": "https://report.invalid",
    },
    "recipe": {
        "seed": 3407,
        "epochs": 1,
        "eval_split": 0.2,
        "batch_size": 2,
        "grad_accum": 4,
        "lora_rank": 64,
        "lora_alpha": 64,
        "lr": 2e-4,
        "load_in_4bit": True,
    },
}


class FakeBackend:
    def __init__(self, cuda=True, empty=False, broken_adapter=False, broken_merged=False):
        self.cuda = cuda
        self.empty = empty
        self.broken_adapter = broken_adapter
        self.broken_merged = broken_merged
        self.events = []
        self.evaluations = 0

    def cuda_available(self):
        return self.cuda

    def load(self, spec):
        self.events.append("load")
        return object()

    def build_items(self, rows, session, max_length):
        self.events.append("encode")
        return [] if self.empty else list(rows)

    def evaluate(self, session, items, batch_size, temperature=1.0):
        self.events.append("evaluate")
        self.evaluations += 1
        accuracy = 0.2 if self.evaluations < 3 else 0.8
        return {"accuracy": accuracy, "ece": 0.2, "loss": 1.0,
                "record_accuracy": accuracy}

    def train(self, session, items, recipe, progress):
        self.events.append("train")
        progress("training", 0.5, step=1, total_steps=1, train_loss=0.7)
        return 12

    def calibrate(self, session, items, batch_size):
        self.events.append("calibrate")
        return 1.25, {"accuracy": 0.75, "ece": 0.05, "loss": 0.4,
                      "record_accuracy": 0.7}

    @staticmethod
    def _head(path):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "joint_head.safetensors").write_bytes(b"head")
        (p / "joint_head_config.json").write_text("{}")
        for name in (
            "sgl_decision_config.json", "joint_schema_model.py", "sgl_systemone.py",
            "LICENSE.cloudflare-clef", "NOTICE.cloudflare-clef",
        ):
            (p / name).write_text("fixture")

    def save_adapter(self, session, path, temperature, spec):
        self.events.append("save_adapter")
        self._head(path)
        if not self.broken_adapter:
            (Path(path) / "adapter_model.safetensors").write_bytes(b"adapter")

    def save_merged(self, session, path, temperature, spec):
        self.events.append("save_merged")
        self._head(path)
        if not self.broken_merged:
            (Path(path) / "model.safetensors").write_bytes(b"merged")

    @staticmethod
    def describe(session):
        return "cuda:0", "torch.bfloat16"


def out_dirs(tmp_path):
    root = tmp_path / "run"
    return {"root": str(root), "adapter_dir": str(root / "adapter"),
            "merged_dir": str(root / "merged")}


def test_decision_training_uses_three_splits_calibrates_and_stages_artifacts(tmp_path):
    backend = FakeBackend()
    progress = []
    rows = [{"state": "x"}] * 4
    result = run_decision_finetune(
        parse_spec(BASE), rows, rows[:2], rows[:2], out_dirs(tmp_path),
        lambda phase, pct, **extra: progress.append((phase, pct, extra)), backend=backend)
    assert backend.events.index("evaluate") < backend.events.index("train")
    assert backend.events.index("train") < backend.events.index("calibrate")
    assert result["baseline"]["accuracy"] == 0.2
    assert result["final"]["accuracy"] == 0.8
    assert result["calibration"]["ece"] == 0.05
    assert result["calibration_temperature"] == 1.25
    assert result["engine"] == "clef-peft"
    assert result["steps"] == 12
    assert result["trained_repo"] == "Qwen/Qwen3.5-0.8B"
    assert result["merged_ok"] is True
    assert (tmp_path / "run/adapter/joint_head.safetensors").is_file()
    assert (tmp_path / "run/merged/model.safetensors").is_file()
    assert any(phase == "training" and extra.get("train_loss") == 0.7
               for phase, _, extra in progress)


def test_no_cuda_is_a_refundable_infrastructure_failure(tmp_path, monkeypatch):
    monkeypatch.delenv("SGL_ALLOW_CPU", raising=False)
    with pytest.raises(InfraError, match="CUDA"):
        run_decision_finetune(
            parse_spec(BASE), [{}], [{}], [{}], out_dirs(tmp_path), lambda *_a, **_k: True,
            backend=FakeBackend(cuda=False))


def test_empty_encoded_split_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="empty split"):
        run_decision_finetune(
            parse_spec(BASE), [{}], [{}], [{}], out_dirs(tmp_path), lambda *_a, **_k: True,
            backend=FakeBackend(empty=True))


def test_incomplete_adapter_save_is_rejected(tmp_path):
    with pytest.raises(InfraError, match="incomplete"):
        run_decision_finetune(
            parse_spec(BASE), [{}], [{}], [{}], out_dirs(tmp_path), lambda *_a, **_k: True,
            backend=FakeBackend(broken_adapter=True))
    assert not (tmp_path / "run/adapter").exists()


def test_incomplete_optional_merged_bundle_does_not_erase_adapter(tmp_path):
    result = run_decision_finetune(
        parse_spec(BASE), [{}], [{}], [{}], out_dirs(tmp_path), lambda *_a, **_k: True,
        backend=FakeBackend(broken_merged=True))
    assert result["merged_ok"] is False
    assert "incomplete" in result["merged_error"]
    assert (tmp_path / "run/adapter/adapter_model.safetensors").is_file()


def test_metrics_include_accuracy_ece_loss_and_record_accuracy():
    metrics = decision_metrics(
        [[4.0, 1.0], [0.0, 3.0], [2.0, 1.0]], [0, 1, 1], [0, 0, 1])
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["record_accuracy"] == pytest.approx(0.5)
    assert 0 <= metrics["ece"] <= 1
    assert metrics["loss"] > 0


def test_temperature_is_fitted_only_from_supplied_holdout_logits():
    temperature, metrics = fit_temperature(
        [[8.0, 0.0], [8.0, 0.0], [0.0, 8.0]], [0, 1, 1], [0, 1, 2])
    assert 0.25 <= temperature <= 4.0
    assert set(metrics) == {"accuracy", "ece", "loss", "record_accuracy"}


