import hashlib
import json
from pathlib import Path

import pytest

from runner.errors import DataError
from runner.main import Deps, EXIT_DATA, EXIT_OK, EXIT_UPLOAD, execute


SPEC = {
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
    "token": "token",
    "artifact_urls": {
        "adapter": "https://a.invalid",
        "merged": "https://m.invalid",
        "report": "https://report.invalid",
    },
    "recipe": {"seed": 3, "epochs": 1, "eval_split": 0.2, "load_in_4bit": True},
}


def decision_rows(n=10):
    question = {
        "route": {
            "type": "choice",
            "instructions": "Choose an endpoint",
            "criteria": {"a": "first", "b": "second"},
        }
    }
    return "\n".join(json.dumps({
        "state": f"request {i}", "questions": question, "gold": {"route": "a"}
    }) for i in range(n))


class Reporter:
    def __init__(self):
        self.results = []
        self.progresses = []

    def progress(self, phase, pct, **extra):
        self.progresses.append((phase, pct))
        return True

    def result(self, status, payload):
        self.results.append((status, payload))
        return True


@pytest.fixture(autouse=True)
def isolated_paths(monkeypatch, tmp_path):
    def build(run_id):
        root = tmp_path / run_id
        return {
            "root": str(root),
            "adapter_dir": str(root / "adapter"),
            "merged_dir": str(root / "merged"),
            "adapter_tar": str(root / "adapter.tar.gz"),
            "merged_tar": str(root / "merged.tar.gz"),
            "gguf": str(root / "model.gguf"),
            "report": str(root / "report.json"),
        }
    monkeypatch.setattr("runner.main.paths", build)

def outcome(merged=True):
    return {
        "seed": 3,
        "steps": 5,
        "baseline": {"accuracy": 0.2, "ece": 0.4, "loss": 1.2},
        "baseline_holdout": {"accuracy": 0.3, "ece": 0.3, "loss": 1.1},
        "final": {"accuracy": 0.8, "ece": 0.08, "loss": 0.4},
        "calibration": {"accuracy": 0.75, "ece": 0.05, "loss": 0.45},
        "engine": "clef-peft",
        "trained_repo": "Qwen/Qwen3.5-0.8B",
        "trained_revision": "decision-sha",
        "device": "cuda:0",
        "dtype": "bfloat16",
        "merged_ok": merged,
        "merged_error": None,
    }


def deps(rep, tmp_path, **overrides):
    def train(spec, train, holdout, test, dirs, progress):
        root = Path(dirs["root"])
        for name in ("adapter", "merged"):
            (root / name).mkdir(parents=True, exist_ok=True)
            (root / name / "file.bin").write_bytes(name.encode())
        return outcome()

    def tar(src, dest):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(Path(src).name.encode())
        return dest

    values = dict(
        fetch=lambda _: decision_rows(),
        train=lambda *_: None,
        convert=lambda *_: None,
        upload=lambda *_: True,
        reporter_factory=lambda _: rep,
        write_report=lambda path, obj: Path(path).write_text(json.dumps(obj)),
        tar=tar,
        sha256=lambda path: (
            hashlib.sha256(Path(path).read_bytes()).hexdigest(), Path(path).stat().st_size),
        decision_train=train,
    )
    values.update(overrides)
    return Deps(**values)


def test_decision_job_delivers_adapter_head_and_optional_merged_without_gguf(tmp_path, monkeypatch):
    rep = Reporter()
    assert execute(SPEC, deps(rep, tmp_path)) == EXIT_OK
    status, report = rep.results[-1]
    assert status == "completed"
    assert report["artifact_format"] == "clef-peft-adapter-head"
    assert report["improved"] is True
    assert report["merged_ok"] is True
    assert report["adapter_bytes"] == len(b"adapter")
    assert report["merged_bytes"] == len(b"merged")
    assert not any("gguf" in key for key in report)


def test_decision_adapter_upload_is_required(tmp_path):
    rep = Reporter()
    calls = []

    def upload(url, path):
        calls.append(url)
        return url != SPEC["artifact_urls"]["adapter"]

    assert execute(SPEC, deps(rep, tmp_path, upload=upload)) == EXIT_UPLOAD
    assert rep.results[-1][0] == "failed"
    assert calls == [SPEC["artifact_urls"]["adapter"]]


def test_decision_merged_upload_failure_is_nonfatal_and_honest(tmp_path):
    rep = Reporter()

    def upload(url, path):
        return url != SPEC["artifact_urls"]["merged"]

    assert execute(SPEC, deps(rep, tmp_path, upload=upload)) == EXIT_OK
    report = rep.results[-1][1]
    assert report["merged_ok"] is False
    assert report["merged_error"] == "merged bundle upload failed"
    assert "merged_sha256" not in report


def test_decision_dataset_failure_happens_before_gpu_call(tmp_path):
    rep = Reporter()
    called = []
    broken = deps(
        rep, tmp_path,
        fetch=lambda _: decision_rows(3) + "\nnot-json",
        decision_train=lambda *_: called.append(True),
    )
    assert execute(SPEC, broken) == EXIT_DATA
    assert called == []
    assert "row 4" in rep.results[-1][1]["error"]


def test_tokenization_context_failure_is_a_data_fault(tmp_path):
    rep = Reporter()

    def reject(*_args):
        raise DataError("decision dataset cannot be encoded within the selected model context")

    assert execute(SPEC, deps(rep, tmp_path, decision_train=reject)) == EXIT_DATA
    assert rep.results[-1][0] == "failed"
