"""run_finetune's engine-selection control flow (SGL_ENGINE=auto tries
unsloth), with a focus on two contracts:

1. The reseed contract: the discarded unsloth attempt (model load, RNG-consuming
   init, etc.) already advanced/clobbered global RNG state, so the vanilla
   retry must re-seed before it starts — otherwise it is not bit-identical to
   a pure SGL_ENGINE=vanilla run with the same seed.

2. The NARROWED fallback-window contract: only a bring-up failure (import +
   load_unsloth + the tokenization it needs, via _load_and_encode) or a wrap
   failure (_wrap_unsloth) triggers the vanilla fallback. A baseline-eval
   failure sits BETWEEN those two windows and must propagate raw — it is a
   fact about the model/data that is exactly as true for vanilla as it is for
   unsloth, so falling back would just burn a doomed-to-fail-identically
   vanilla re-run first.

This repo intentionally has no torch/unsloth installed (see runner/trainer.py's
module docstring: real torch is only ever imported inside functions). To
exercise run_finetune's CONTROL FLOW without either, this module:

  - injects a minimal fake `torch` module into sys.modules, just enough to
    satisfy run_finetune's own torch calls (cuda probes, dtype constants,
    optim.AdamW) — never real tensor math;
  - keeps train_rows/eval_rows/train_enc EMPTY so the training loop's
    per-batch torch calls (_batched) never actually execute their bodies;
  - monkeypatches _load_and_encode, _baseline_eval, _wrap_unsloth, and
    _attempt_engine directly, since those are exactly the seams trainer.py
    uses to keep the two fallback windows independently testable — these
    tests are about whether run_finetune reseeds, falls back (or doesn't),
    and picks the right engine; not about tensor math (that's the live GPU
    smoke step's job).
"""
from __future__ import annotations

import sys
import types

import pytest

from runner import trainer
from runner import trainer_unsloth
from runner.errors import InfraError
from runner.spec import parse_spec

RAW_SPEC = {
    "run_id": "r1", "kind": "finetune",
    # A pinned unsloth mirror IS present here — these tests exercise the
    # bring-up/wrap exception-fallback paths, which only run once
    # unsloth_pin_ok(spec) is true (see runner/trainer.py). The no-pin path
    # (skip unsloth entirely, reason='NoUnslothPin') has its own tests below,
    # built off a spec that omits these two fields.
    "base_model": {"hf_repo": "Q/M", "revision": "sha",
                   "unsloth_repo": "unsloth/M", "unsloth_revision": "unsloth-sha"},
    "dataset_url": "https://d.invalid/x.jsonl", "quantize": "Q4_K_M",
    "progress_url": "https://p.invalid", "result_url": "https://res.invalid",
    "token": "t",
    "artifact_urls": {"adapter": "https://a.invalid", "gguf": "https://g.invalid",
                      "report": "https://r.invalid"},
    "recipe": {"seed": 42, "epochs": 1, "eval_split": 0.5, "batch_size": 1},
}

RAW_SPEC_NO_UNSLOTH_PIN = {
    **RAW_SPEC,
    "base_model": {"hf_repo": "Q/M", "revision": "sha"},
}


def _spec():
    return parse_spec(RAW_SPEC)


def _spec_no_pin():
    return parse_spec(RAW_SPEC_NO_UNSLOTH_PIN)


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


class _FakeParam:
    def __init__(self, dtype="torch.bfloat16"):
        self.dtype = dtype
        self.requires_grad = True


class _FakeModel:
    def parameters(self):
        return iter([_FakeParam()])

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


def _fake_load_and_encode(model=None):
    model = model or _FakeModel()

    def fn(load_fn, spec_, r, dtype, device, train_rows, eval_rows, engine_name):
        assert engine_name == "unsloth"
        return model, object(), [], []
    return fn


def test_bringup_failure_falls_back_to_vanilla_and_reseeds(monkeypatch, fake_torch):
    spec = _spec()

    seed_calls = []
    monkeypatch.setattr(trainer, "_set_all_seeds", lambda seed: seed_calls.append(seed))
    monkeypatch.setattr(trainer, "_evaluate", lambda model, batches: {
        "eval_loss": 1.0, "perplexity": 2.7, "eval_tokens": 10})
    monkeypatch.setattr(trainer, "_save_vanilla", lambda model, tokenizer, out_dirs: None)

    def fake_load_and_encode(load_fn, spec_, r, dtype, device, train_rows, eval_rows, engine_name):
        assert engine_name == "unsloth"
        raise RuntimeError("simulated unsloth import/load failure")
    monkeypatch.setattr(trainer, "_load_and_encode", fake_load_and_encode)

    baseline_calls = []
    monkeypatch.setattr(trainer, "_baseline_eval",
                         lambda *a, **k: baseline_calls.append(1) or (None, None))

    wrap_calls = []
    monkeypatch.setattr(trainer, "_wrap_unsloth",
                         lambda model, r: wrap_calls.append(1) or model)

    attempt_calls = []

    def fake_attempt(load_fn, wrap_fn, spec_, r, dtype, device, train_rows, eval_rows,
                      on_progress, engine_name):
        attempt_calls.append(engine_name)
        assert engine_name == "vanilla"
        return _FakeModel(), object(), [], [], {
            "eval_loss": 2.0, "perplexity": 7.4, "eval_tokens": 10}

    monkeypatch.setattr(trainer, "_attempt_engine", fake_attempt)

    result = trainer.run_finetune(
        spec, [], [], {"adapter_dir": "/x", "merged_dir": "/y"}, lambda *a, **k: None)

    # A bring-up failure never reaches baseline eval or wrap on the unsloth
    # side — it fails before either could run.
    assert baseline_calls == []
    assert wrap_calls == []
    assert attempt_calls == ["vanilla"]
    # Seeded once at the top of run_finetune, and AGAIN as the first thing the
    # fallback path does, before the vanilla retry.
    assert seed_calls == [spec.recipe.seed, spec.recipe.seed]
    assert result["engine"] == "vanilla"
    assert result["engine_fallback_reason"] == "RuntimeError"
    assert result["baseline"]["eval_loss"] == 2.0


def test_wrap_failure_falls_back_to_vanilla_and_reseeds(monkeypatch, fake_torch):
    spec = _spec()

    seed_calls = []
    monkeypatch.setattr(trainer, "_set_all_seeds", lambda seed: seed_calls.append(seed))
    monkeypatch.setattr(trainer, "_evaluate", lambda model, batches: {
        "eval_loss": 1.0, "perplexity": 2.7, "eval_tokens": 10})
    monkeypatch.setattr(trainer, "_save_vanilla", lambda model, tokenizer, out_dirs: None)

    monkeypatch.setattr(trainer, "_load_and_encode", _fake_load_and_encode())

    baseline_calls = []

    def fake_baseline_eval(tokenizer, model, eval_enc, r, device, on_progress):
        baseline_calls.append("unsloth")
        return [], {"eval_loss": 9.0, "perplexity": 99.0, "eval_tokens": 10}
    monkeypatch.setattr(trainer, "_baseline_eval", fake_baseline_eval)

    def fake_wrap_unsloth(model, r):
        raise RuntimeError("simulated wrap failure")
    monkeypatch.setattr(trainer, "_wrap_unsloth", fake_wrap_unsloth)

    attempt_calls = []

    def fake_attempt(load_fn, wrap_fn, spec_, r, dtype, device, train_rows, eval_rows,
                      on_progress, engine_name):
        attempt_calls.append(engine_name)
        assert engine_name == "vanilla"
        return _FakeModel(), object(), [], [], {
            "eval_loss": 2.0, "perplexity": 7.4, "eval_tokens": 10}

    monkeypatch.setattr(trainer, "_attempt_engine", fake_attempt)

    result = trainer.run_finetune(
        spec, [], [], {"adapter_dir": "/x", "merged_dir": "/y"}, lambda *a, **k: None)

    # The unsloth baseline DID run (it's the wrap that failed) but is
    # discarded — the full vanilla re-run computes its own.
    assert baseline_calls == ["unsloth"]
    assert attempt_calls == ["vanilla"]
    assert seed_calls == [spec.recipe.seed, spec.recipe.seed]
    assert result["engine"] == "vanilla"
    assert result["engine_fallback_reason"] == "RuntimeError"
    assert result["baseline"]["eval_loss"] == 2.0  # vanilla's own, not the discarded 9.0


def test_baseline_eval_failure_does_not_fall_back(monkeypatch, fake_torch):
    spec = _spec()

    seed_calls = []
    monkeypatch.setattr(trainer, "_set_all_seeds", lambda seed: seed_calls.append(seed))

    monkeypatch.setattr(trainer, "_load_and_encode", _fake_load_and_encode())

    def fake_baseline_eval(tokenizer, model, eval_enc, r, device, on_progress):
        raise RuntimeError("injected baseline-eval failure")
    monkeypatch.setattr(trainer, "_baseline_eval", fake_baseline_eval)

    wrap_calls = []
    monkeypatch.setattr(trainer, "_wrap_unsloth",
                         lambda model, r: wrap_calls.append(1) or model)

    def fake_attempt(*a, **k):
        raise AssertionError("must not fall back to vanilla on a baseline-eval failure")
    monkeypatch.setattr(trainer, "_attempt_engine", fake_attempt)

    with pytest.raises(RuntimeError, match="injected baseline-eval failure"):
        trainer.run_finetune(
            spec, [], [], {"adapter_dir": "/x", "merged_dir": "/y"}, lambda *a, **k: None)

    # No fallback attempted, wrap never reached, and no reseed (nothing was
    # discarded/retried) — just the ONE seed call at the top of run_finetune.
    assert wrap_calls == []
    assert seed_calls == [spec.recipe.seed]


def test_success_path_seeds_exactly_once(monkeypatch, fake_torch):
    spec = _spec()

    seed_calls = []
    monkeypatch.setattr(trainer, "_set_all_seeds", lambda seed: seed_calls.append(seed))
    monkeypatch.setattr(trainer, "_evaluate", lambda model, batches: {
        "eval_loss": 1.0, "perplexity": 2.7, "eval_tokens": 10})
    monkeypatch.setattr(trainer_unsloth, "save_unsloth", lambda *a, **k: None)

    monkeypatch.setattr(trainer, "_load_and_encode", _fake_load_and_encode())
    monkeypatch.setattr(trainer, "_baseline_eval",
                         lambda tokenizer, model, eval_enc, r, device, on_progress:
                         ([], {"eval_loss": 2.0, "perplexity": 7.4, "eval_tokens": 10}))
    monkeypatch.setattr(trainer, "_wrap_unsloth", lambda model, r: model)

    result = trainer.run_finetune(
        spec, [], [], {"adapter_dir": "/x", "merged_dir": "/y"}, lambda *a, **k: None)

    # No fallback happened, so there is only ever the ONE seed call at the top
    # of run_finetune — a reseed here would be pointless (nothing to undo).
    assert seed_calls == [spec.recipe.seed]
    assert result["engine"] == "unsloth"
    assert result["engine_fallback_reason"] is None
    assert result["dtype"] == "torch.bfloat16"
    assert result["trained_repo"] == spec.unsloth_repo
    assert result["trained_revision"] == spec.unsloth_revision


# --- no unsloth mirror pinned: never attempt unsloth at all -----------------

def test_auto_with_no_unsloth_pin_never_attempts_unsloth_and_goes_straight_to_vanilla(
        monkeypatch, fake_torch):
    spec = _spec_no_pin()
    assert spec.unsloth_repo is None and spec.unsloth_revision is None

    seed_calls = []
    monkeypatch.setattr(trainer, "_set_all_seeds", lambda seed: seed_calls.append(seed))
    monkeypatch.setattr(trainer, "_evaluate", lambda model, batches: {
        "eval_loss": 1.0, "perplexity": 2.7, "eval_tokens": 10})
    monkeypatch.setattr(trainer, "_save_vanilla", lambda model, tokenizer, out_dirs: None)

    # _load_and_encode is the unsloth bring-up seam (import + load_unsloth) —
    # if the no-pin gate is doing its job this must never be called at all.
    probe_calls = []

    def fake_load_and_encode(load_fn, spec_, r, dtype, device, train_rows, eval_rows, engine_name):
        probe_calls.append(engine_name)
        raise AssertionError("must not attempt unsloth bring-up when no mirror is pinned")
    monkeypatch.setattr(trainer, "_load_and_encode", fake_load_and_encode)

    attempt_calls = []

    def fake_attempt(load_fn, wrap_fn, spec_, r, dtype, device, train_rows, eval_rows,
                      on_progress, engine_name):
        attempt_calls.append(engine_name)
        assert engine_name == "vanilla"
        return _FakeModel(), object(), [], [], {
            "eval_loss": 2.0, "perplexity": 7.4, "eval_tokens": 10}
    monkeypatch.setattr(trainer, "_attempt_engine", fake_attempt)

    result = trainer.run_finetune(
        spec, [], [], {"adapter_dir": "/x", "merged_dir": "/y"}, lambda *a, **k: None)

    assert probe_calls == []  # the unsloth probe was NEVER called
    assert attempt_calls == ["vanilla"]
    # Nothing was discarded (unsloth was never touched), so no reseed — just
    # the one seed call at the top of run_finetune.
    assert seed_calls == [spec.recipe.seed]
    assert result["engine"] == "vanilla"
    assert result["engine_fallback_reason"] == "NoUnslothPin"
    assert result["trained_repo"] == spec.hf_repo
    assert result["trained_revision"] == spec.hf_revision


def test_forced_unsloth_with_no_pin_raises_infra_error_before_any_attempt(
        monkeypatch, fake_torch):
    monkeypatch.setenv("SGL_ENGINE", "unsloth")
    spec = _spec_no_pin()

    monkeypatch.setattr(trainer, "_set_all_seeds", lambda seed: None)

    def fake_attempt(*a, **k):
        raise AssertionError("must not attempt an unsloth load without a mirror pin")
    monkeypatch.setattr(trainer, "_attempt_engine", fake_attempt)

    with pytest.raises(InfraError, match=r"\[engine=unsloth\] no unsloth mirror pinned"):
        trainer.run_finetune(
            spec, [], [], {"adapter_dir": "/x", "merged_dir": "/y"}, lambda *a, **k: None)
