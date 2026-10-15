from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from runner.errors import InfraError
from runner.spec import parse_spec
from runner.trainer_decision import (
    decision_metrics,
    fit_temperature,
    run_decision_finetune,
    TorchDecisionBackend,
    validate_artifact_manifest,
    verify_systemone_artifact,
)


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
    def __init__(self, cuda=True, empty=False, broken_adapter=False, broken_merged=False,
                 adapter_reload_fails=False, merged_reload_fails=False):
        self.cuda = cuda
        self.empty = empty
        self.broken_adapter = broken_adapter
        self.broken_merged = broken_merged
        self.adapter_reload_fails = adapter_reload_fails
        self.merged_reload_fails = merged_reload_fails
        self.events = []
        self.evaluations = 0

    def cuda_available(self):
        return self.cuda

    def seed(self, seed):
        self.events.append(("seed", seed))

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
        (p / "sgl_decision_config.json").write_text('{"temperature": 1.25}')
        (p / "tokenizer_config.json").write_text("{}")
        (p / "tokenizer.json").write_text("{}")
        (p / "processor_config.json").write_text("{}")
        for name in ("joint_schema_model.py", "sgl_systemone.py",
                     "LICENSE.cloudflare-clef", "NOTICE.cloudflare-clef"):
            (p / name).write_text("fixture")

    def save_adapter(self, session, path, temperature, spec):
        self.events.append("save_adapter")
        self._head(path)
        (Path(path) / "adapter_config.json").write_text("{}")
        if not self.broken_adapter:
            (Path(path) / "adapter_model.safetensors").write_bytes(b"adapter")

    def save_merged(self, session, path, temperature, spec):
        self.events.append("save_merged")
        self._head(path)
        (Path(path) / "config.json").write_text("{}")
        if not self.broken_merged:
            (Path(path) / "model.safetensors").write_bytes(b"merged")

    @staticmethod
    def describe(session):
        return "cuda:0", "torch.bfloat16"

    def release(self, session):
        self.events.append("release")

    def verify_artifact(self, directory, merged):
        self.events.append("verify_merged" if merged else "verify_adapter")
        if (merged and self.merged_reload_fails) or (not merged and self.adapter_reload_fails):
            raise RuntimeError("reload failed")
        verify_systemone_artifact(
            directory,
            merged,
            loader=lambda path: (object(), object(), 1.25),
            decide=lambda model, processor, request, temperature: {
                "model": request["model"],
                "answers": {"ready": {"type": "noul", "noul": 0.9}},
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        )


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
    assert backend.events[0] == ("seed", 3407)
    assert backend.events.index(("seed", 3407)) < backend.events.index("load")
    assert backend.events.index("release") < backend.events.index("verify_adapter")
    assert backend.events.index("verify_adapter") < backend.events.index("verify_merged")
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


def test_seed_initializes_python_numpy_transformers_cpu_and_all_cuda_devices(monkeypatch):
    calls = []
    backend = TorchDecisionBackend.__new__(TorchDecisionBackend)
    backend.torch = SimpleNamespace(
        manual_seed=lambda seed: calls.append(("torch", seed)),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            manual_seed_all=lambda seed: calls.append(("cuda", seed)),
        ),
    )
    monkeypatch.setattr("runner.trainer_decision.random.seed",
                        lambda seed: calls.append(("python", seed)))
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(
        random=SimpleNamespace(seed=lambda seed: calls.append(("numpy", seed)))))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        set_seed=lambda seed: calls.append(("transformers", seed))))
    backend.seed(41)
    assert calls == [
        ("python", 41), ("numpy", 41), ("transformers", 41),
        ("torch", 41), ("cuda", 41),
    ]


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


def test_required_adapter_reload_failure_removes_all_undeliverable_artifacts(tmp_path):
    with pytest.raises(InfraError, match="reload verification"):
        run_decision_finetune(
            parse_spec(BASE), [{}], [{}], [{}], out_dirs(tmp_path),
            lambda *_a, **_k: True, backend=FakeBackend(adapter_reload_fails=True))
    assert not (tmp_path / "run/adapter").exists()
    assert not (tmp_path / "run/merged").exists()


def test_optional_merged_reload_failure_preserves_verified_adapter(tmp_path):
    result = run_decision_finetune(
        parse_spec(BASE), [{}], [{}], [{}], out_dirs(tmp_path),
        lambda *_a, **_k: True, backend=FakeBackend(merged_reload_fails=True))
    assert result["merged_ok"] is False
    assert result["merged_error"] == "merged reload verification failed: RuntimeError"
    assert (tmp_path / "run/adapter/adapter_model.safetensors").is_file()
    assert not (tmp_path / "run/merged").exists()


def test_manifest_requires_reloadable_adapter_processor_tokenizer_and_head(tmp_path):
    backend = FakeBackend()
    path = tmp_path / "adapter"
    backend.save_adapter(object(), str(path), 1.25, parse_spec(BASE))
    manifest = validate_artifact_manifest(path, merged=False)
    assert manifest["adapter_config.json"] > 0
    assert manifest["adapter_model.safetensors"] > 0
    assert manifest["tokenizer_config.json"] > 0
    assert manifest["processor_config.json"] > 0
    assert manifest["joint_head.safetensors"] > 0
    (path / "processor_config.json").unlink()
    with pytest.raises(InfraError, match="processor_config"):
        validate_artifact_manifest(path, merged=False)


def test_systemone_reload_probe_uses_real_manifest_and_mocked_compute_boundary(tmp_path):
    backend = FakeBackend()
    path = tmp_path / "merged"
    backend.save_merged(object(), str(path), 1.25, parse_spec(BASE))
    calls = []

    def loader(actual):
        calls.append(("load", Path(actual).name))
        return object(), object(), 1.25

    def decide(model, processor, request, temperature):
        calls.append(("decide", request["questions"]["ready"]["type"], temperature))
        return {"answers": {"ready": {"type": "noul", "noul": 0.8}}}

    verify_systemone_artifact(path, merged=True, loader=loader, decide=decide)
    assert calls == [("load", "merged"), ("decide", "noul", 1.25)]


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


