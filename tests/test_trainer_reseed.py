"""run_finetune's engine-selection control flow (SGL_ENGINE=auto tries
unsloth, ANY failure falls back to vanilla), with a focus on the reseed
contract: the discarded unsloth attempt (model load, RNG-consuming init,
etc.) already advanced/clobbered global RNG state, so the vanilla retry must
re-seed before it starts — otherwise it is not bit-identical to a pure
SGL_ENGINE=vanilla run with the same seed.

This repo intentionally has no torch/unsloth installed (see runner/trainer.py's
module docstring: real torch is only ever imported inside functions). To
exercise run_finetune's CONTROL FLOW without either, this module:

  - injects a minimal fake `torch` module into sys.modules, just enough to
    satisfy run_finetune's own torch calls (cuda probes, dtype constants,
    optim.AdamW) — never real tensor math;
  - keeps train_rows/eval_rows/train_enc EMPTY so the training loop's
    per-batch torch calls (_batched) never actually execute their bodies;
  - monkeypatches _attempt_engine, _evaluate and _save_vanilla directly,
    since those are exactly the seams trainer.py already uses to keep the
    two engines interchangeable — this test is about whether run_finetune
    reseeds and picks the right engine on fallback, not about tensor math
    (that's the live GPU smoke step's job).
"""
from __future__ import annotations

import sys
import types

import pytest

from runner import trainer
from runner import trainer_unsloth
from runner.spec import parse_spec

RAW_SPEC = {
    "run_id": "r1", "kind": "finetune",
    "base_model": {"hf_repo": "Q/M", "revision": "sha"},
    "dataset_url": "https://d.invalid/x.jsonl", "quantize": "Q4_K_M",
    "progress_url": "https://p.invalid", "result_url": "https://res.invalid",
    "token": "t",
    "artifact_urls": {"adapter": "https://a.invalid", "gguf": "https://g.invalid",
                      "report": "https://r.invalid"},
    "recipe": {"seed": 42, "epochs": 1, "eval_split": 0.5, "batch_size": 1},
}


def _spec():
    return parse_spec(RAW_SPEC)


class _FakeCuda:
    def is_available(self):
        return True

    def is_bf16_supported(self):
        return True


class _FakeAdamW:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def step(self):
        pass

    def zero_grad(self):
        pass


class _FakeModel:
    def parameters(self):
        return iter([])

    def train(self):
        pass


def _fake_torch_module():
    mod = types.SimpleNamespace()
    mod.cuda = _FakeCuda()
    mod.bfloat16 = "bfloat16"
    mod.float16 = "float16"
    mod.float32 = "float32"
    mod.optim = types.SimpleNamespace(AdamW=_FakeAdamW)
    return mod


@pytest.fixture
def fake_torch(monkeypatch):
    mod = _fake_torch_module()
    monkeypatch.setitem(sys.modules, "torch", mod)
    return mod


@pytest.fixture(autouse=True)
def _no_sgl_engine_env(monkeypatch):
    # Default to 'auto' regardless of the ambient environment.
    monkeypatch.delenv("SGL_ENGINE", raising=False)


def test_fallback_reseeds_before_the_vanilla_retry(monkeypatch, fake_torch):
    spec = _spec()

    seed_calls = []
    monkeypatch.setattr(trainer, "_set_all_seeds", lambda seed: seed_calls.append(seed))
    monkeypatch.setattr(trainer, "_evaluate", lambda model, batches: {
        "eval_loss": 1.0, "perplexity": 2.7, "eval_tokens": 10})
    monkeypatch.setattr(trainer, "_save_vanilla", lambda model, tokenizer, out_dirs: None)

    attempt_calls = []

    def fake_attempt(load_fn, wrap_fn, spec_, r, dtype, device, train_rows, eval_rows,
                      on_progress, engine_name):
        attempt_calls.append(engine_name)
        if engine_name == "unsloth":
            raise RuntimeError("simulated unsloth failure")
        return _FakeModel(), object(), [], [], {
            "eval_loss": 2.0, "perplexity": 7.4, "eval_tokens": 10}

    monkeypatch.setattr(trainer, "_attempt_engine", fake_attempt)

    result = trainer.run_finetune(
        spec, [], [], {"adapter_dir": "/x", "merged_dir": "/y"}, lambda *a, **k: None)

    assert attempt_calls == ["unsloth", "vanilla"]
    # Seeded once at the top of run_finetune, and AGAIN as the first thing the
    # except-branch does, before the vanilla retry — not zero times (the bug
    # this fix closes) and not with a different seed.
    assert seed_calls == [spec.recipe.seed, spec.recipe.seed]
    assert result["engine"] == "vanilla"
    assert result["engine_fallback_reason"] == "RuntimeError"


def test_success_path_seeds_exactly_once(monkeypatch, fake_torch):
    spec = _spec()

    seed_calls = []
    monkeypatch.setattr(trainer, "_set_all_seeds", lambda seed: seed_calls.append(seed))
    monkeypatch.setattr(trainer, "_evaluate", lambda model, batches: {
        "eval_loss": 1.0, "perplexity": 2.7, "eval_tokens": 10})
    monkeypatch.setattr(trainer_unsloth, "save_unsloth", lambda *a, **k: None)

    def fake_attempt(load_fn, wrap_fn, spec_, r, dtype, device, train_rows, eval_rows,
                      on_progress, engine_name):
        return _FakeModel(), object(), [], [], {
            "eval_loss": 2.0, "perplexity": 7.4, "eval_tokens": 10}

    monkeypatch.setattr(trainer, "_attempt_engine", fake_attempt)

    result = trainer.run_finetune(
        spec, [], [], {"adapter_dir": "/x", "merged_dir": "/y"}, lambda *a, **k: None)

    # No fallback happened, so there is only ever the ONE seed call at the top
    # of run_finetune — a reseed here would be pointless (nothing to undo).
    assert seed_calls == [spec.recipe.seed]
    assert result["engine"] == "unsloth"
    assert result["engine_fallback_reason"] is None
