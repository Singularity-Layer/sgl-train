import os
import sys
import types

from runner import trainer
from runner.spec import parse_spec


BASE = {
    "run_id": "r1", "kind": "finetune",
    "base_model": {"hf_repo": "Q/M", "revision": "sha"},
    "dataset_url": "https://d.invalid/x.jsonl", "quantize": "Q4_K_M",
    "progress_url": "https://p.invalid", "result_url": "https://res.invalid",
    "token": "t",
    "artifact_urls": {"adapter": "https://a.invalid", "gguf": "https://g.invalid",
                      "report": "https://r.invalid"},
    "recipe": {"seed": 1, "epochs": 1, "eval_split": 0.5, "batch_size": 1,
               "grad_accum": 4},
}


class Tensor:
    def clone(self):
        return Tensor()

    def __eq__(self, _other):
        return False

    def __setitem__(self, _key, _value):
        pass


class Loss:
    def __init__(self, model):
        self.model = model

    def __truediv__(self, _other):
        return self

    def backward(self):
        self.model.backward_calls += 1

    def item(self):
        return 1.0


class Model:
    def __init__(self):
        self.backward_calls = 0

    def parameters(self):
        return iter([types.SimpleNamespace(dtype="torch.bfloat16", requires_grad=True)])

    def train(self):
        pass

    def __call__(self, **_kwargs):
        return types.SimpleNamespace(loss=Loss(self))


class Optimizer:
    def __init__(self, *_args, **_kwargs):
        self.steps = 0

    def step(self):
        self.steps += 1

    def zero_grad(self):
        pass


class Runtime:
    def __init__(self, model, finalize_after=None):
        self.model = model
        self.finalize_after = finalize_after
        self.phase_seconds = {}

    def phase_started(self):
        return 0.0

    def record_phase(self, name, _started):
        self.phase_seconds[name] = 0.0

    def should_finalize(self):
        return self.finalize_after is not None and self.model.backward_calls >= self.finalize_after

    def elapsed_seconds(self):
        return 0.0

    def estimated_remaining_seconds(self, *_args):
        return 0


def _run(monkeypatch, count, finalize_after=None, checkpoint=None):
    spec = parse_spec(BASE)
    model = Model()
    optimizer = Optimizer()
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True),
        bfloat16="bfloat16", float16="float16", float32="float32",
        optim=types.SimpleNamespace(AdamW=lambda *_a, **_k: optimizer),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("SGL_ENGINE", "vanilla")
    monkeypatch.setattr(trainer, "_set_all_seeds", lambda _seed: None)
    monkeypatch.setattr(
        trainer, "_attempt_engine",
        lambda *_a, **_k: (model, object(), list(range(count)), [],
                           {"eval_loss": 2.0, "perplexity": 7.4, "eval_tokens": 1}),
    )
    monkeypatch.setattr(trainer, "_batched", lambda *_a, **_k: [(Tensor(), Tensor())] * count)
    monkeypatch.setattr(trainer, "_evaluate", lambda *_a, **_k: {
        "eval_loss": 1.0, "perplexity": 2.7, "eval_tokens": 1})
    monkeypatch.setattr(trainer, "_save_adapter_only", lambda *_a, **_k: None)
    monkeypatch.setattr(trainer, "_save_vanilla", lambda *_a, **_k: None)
    monkeypatch.setattr(trainer, "_save_training_checkpoint", lambda *_a, **_k: "/checkpoint")

    result = trainer.run_finetune(
        spec, [], [], {"adapter_dir": "/a", "merged_dir": "/m"},
        lambda *_a, **_k: None, on_checkpoint=checkpoint,
        runtime=Runtime(model, finalize_after),
    )
    return result, model, optimizer


def test_early_finalization_waits_for_optimizer_boundary(monkeypatch):
    published = []

    class Publisher:
        def next_slot(self):
            return "a"

        def __call__(self, slot, path, step, epoch, planned):
            published.append((slot, step, epoch, planned))
            return {}

    result, model, optimizer = _run(monkeypatch, 8, finalize_after=1, checkpoint=Publisher())
    assert result["early_finalized"] is True
    assert result["steps"] == 4
    assert model.backward_calls == 4
    assert optimizer.steps == 1
    assert published == [("a", 4, 1, 8)]


def test_complete_run_flushes_one_partial_accumulation_without_extra_step(monkeypatch):
    result, model, optimizer = _run(monkeypatch, 10)
    assert result["early_finalized"] is False
    assert result["steps"] == 10
    assert model.backward_calls == 10
    assert optimizer.steps == 3
