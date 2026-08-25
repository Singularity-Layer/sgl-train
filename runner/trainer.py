"""LoRA fine-tune plus held-out eval. THE ONLY torch module (plus
trainer_unsloth.py, the optional unsloth engine, which follows the same rule).

torch is imported inside functions, never at module scope, so the rest of the
package (and the whole test suite) imports on a laptop with no CUDA. The two
pure decision functions below (resolve_engine, require_engine_for_recipe) are
the exception: no imports at all, so they're unit-testable with no torch and
no unsloth installed either.
"""
from __future__ import annotations

import os
import sys

from .errors import InfraError
from .evalmath import aggregate


def resolve_engine(env_value: str, unsloth_probe_result: bool) -> str:
    """Pure decision table for SGL_ENGINE ∈ {auto (default), vanilla, unsloth}.

    - 'vanilla' -> always vanilla; the unsloth attempt is skipped entirely.
    - 'unsloth' -> always unsloth; forced, no fallback (testing mode) — the
      probe result is ignored, a failure downstream is the caller's problem.
    - anything else (including 'auto', unset, or an unrecognized value) ->
      unsloth if the probe succeeded, vanilla otherwise.

    unsloth_probe_result is not a cheap pre-check: in real use it is "did the
    actual unsloth load+wrap attempt succeed", computed by the caller before
    this function is consulted. It's injected here purely so this decision
    table is testable without torch or unsloth installed.
    """
    v = (env_value or "auto").strip().lower()
    if v == "vanilla":
        return "vanilla"
    if v == "unsloth":
        return "unsloth"
    return "unsloth" if unsloth_probe_result else "vanilla"


def require_engine_for_recipe(engine: str, load_in_4bit: bool) -> None:
    """4-bit (QLoRA) training needs unsloth's bitsandbytes integration — the
    vanilla engine has no 4-bit path at all. Raise InfraError, not SpecError:
    the recipe itself is valid, this box just lacks the capability to honor
    it, and that is our fault, never the buyer's — EXIT_INFRA, refundable.
    """
    if load_in_4bit and engine == "vanilla":
        raise InfraError(
            "4-bit training requires the unsloth engine, which is unavailable on this box")


def _set_all_seeds(seed: int) -> None:
    import random as _random

    import numpy as np
    import torch

    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _encode(tokenizer, rows, max_seq_len):
    """Chat rows → input_ids. Loss is computed on every token, which is the
    simplest honest choice; assistant-only masking is a later refinement and
    would change what eval_loss MEANS, so it must not be added silently."""
    out = []
    for r in rows:
        text = tokenizer.apply_chat_template(r.messages, tokenize=False)
        ids = tokenizer(text, truncation=True, max_length=max_seq_len,
                        return_tensors=None)["input_ids"]
        if len(ids) >= 2:
            out.append(ids)
    return out


def _evaluate(model, batches) -> dict:
    import torch

    model.eval()
    acc: list[tuple[float, int]] = []
    with torch.no_grad():
        for input_ids, attn in batches:
            labels = input_ids.clone()
            labels[attn == 0] = -100
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            # HF's causal-LM loss averages over SHIFTED labels (labels[..., 1:]) — count
            # exactly those positions so loss*n_tokens reconstructs the true sum.
            n_tokens = int((labels[..., 1:] != -100).sum().item())
            # HF returns a MEAN over supervised tokens; multiply back out so the
            # aggregation is token-weighted rather than batch-weighted.
            acc.append((float(out.loss.item()) * n_tokens, n_tokens))
    model.train()
    return aggregate(acc)


def _batched(tokenizer, encoded, batch_size, device):
    import torch

    for i in range(0, len(encoded), batch_size):
        chunk = encoded[i:i + batch_size]
        width = max(len(c) for c in chunk)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        ids = torch.full((len(chunk), width), pad, dtype=torch.long)
        attn = torch.zeros((len(chunk), width), dtype=torch.long)
        for j, c in enumerate(chunk):
            ids[j, :len(c)] = torch.tensor(c, dtype=torch.long)
            attn[j, :len(c)] = 1
        yield ids.to(device), attn.to(device)


def _load_vanilla(spec, dtype, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(spec.hf_repo, revision=spec.hf_revision)
        model = AutoModelForCausalLM.from_pretrained(
            spec.hf_repo, revision=spec.hf_revision,
            torch_dtype=dtype, device_map=None)
    except Exception as e:
        raise InfraError(f"model download failed for {spec.hf_repo}@{spec.hf_revision}: {e}") from e
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model.to(device), tokenizer


def _wrap_vanilla_lora(model, r):
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(model, LoraConfig(
        r=r.lora_rank, lora_alpha=r.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    return model


def _save_vanilla(model, tokenizer, out_dirs) -> None:
    # Training-loop failures above stay EXIT_TRAIN — they're a fact about the run.
    # A save-time OSError (disk full, read-only fs) is our infrastructure, not the
    # buyer's data or recipe, so it converts to InfraError -> EXIT_INFRA (refundable).
    try:
        os.makedirs(out_dirs["adapter_dir"], exist_ok=True)
        model.save_pretrained(out_dirs["adapter_dir"])
        tokenizer.save_pretrained(out_dirs["adapter_dir"])

        merged = model.merge_and_unload()
        os.makedirs(out_dirs["merged_dir"], exist_ok=True)
        merged.save_pretrained(out_dirs["merged_dir"], safe_serialization=True)
        tokenizer.save_pretrained(out_dirs["merged_dir"])
    except OSError as e:
        raise InfraError(f"artifact save failed (disk?): {e}") from e


def _load_unsloth(spec, dtype, device):
    from .trainer_unsloth import load_unsloth
    return load_unsloth(spec, dtype)


def _wrap_unsloth(model, r):
    from .trainer_unsloth import wrap_unsloth_lora
    return wrap_unsloth_lora(model, r)


def _attempt_engine(load_fn, wrap_fn, spec, r, dtype, device, train_rows, eval_rows, on_progress):
    """Shared load -> encode -> baseline_eval -> wrap sequence, parameterized
    by engine-specific load/wrap functions. Used for both engines so the
    baseline honesty guarantee (eval on the UNWRAPPED model, before any LoRA
    is attached) and the eval math are identical regardless of which engine
    ran — that's what keeps baseline/final numbers comparable across engines.
    """
    model, tokenizer = load_fn(spec, dtype, device)

    train_enc = _encode(tokenizer, train_rows, r.max_seq_len)
    eval_enc = _encode(tokenizer, eval_rows, r.max_seq_len)
    if not train_enc or not eval_enc:
        raise RuntimeError("dataset produced no usable sequences after tokenization")

    # BASELINE FIRST, on the untouched base model. Without this number there is
    # nothing to compare the final model to, and "it trained" is not a result.
    on_progress("baseline_eval", 0.10)
    eval_batches = list(_batched(tokenizer, eval_enc, r.batch_size, device))
    baseline = _evaluate(model, eval_batches)

    model = wrap_fn(model, r)
    return model, tokenizer, train_enc, eval_batches, baseline


def run_finetune(spec, train_rows, eval_rows, out_dirs, on_progress) -> dict:
    import torch

    r = spec.recipe
    _set_all_seeds(r.seed)
    if not torch.cuda.is_available() and os.environ.get("SGL_ALLOW_CPU") != "1":
        raise InfraError(
            "CUDA is unavailable on this box — refusing to train on CPU "
            "(set SGL_ALLOW_CPU=1 to override for local testing)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    if device == "cpu":
        dtype = torch.float32  # fp16 CPU training is broken

    engine_mode = os.environ.get("SGL_ENGINE", "auto")
    engine_fallback_reason = None

    on_progress("loading_model", 0.05, model=spec.hf_repo)

    v = (engine_mode or "auto").strip().lower()
    if v == "vanilla":
        # Skip the unsloth attempt entirely (rule: 'vanilla' never tries it).
        engine = resolve_engine(engine_mode, False)
        require_engine_for_recipe(engine, r.load_in_4bit)
        model, tokenizer, train_enc, eval_batches, baseline = _attempt_engine(
            _load_vanilla, _wrap_vanilla_lora, spec, r, dtype, device, train_rows, eval_rows, on_progress)
    elif v == "unsloth":
        # Forced: no fallback (testing mode). Any exception — import, load, an
        # empty-dataset RuntimeError, a wrap failure — propagates as-is.
        engine = resolve_engine(engine_mode, True)
        model, tokenizer, train_enc, eval_batches, baseline = _attempt_engine(
            _load_unsloth, _wrap_unsloth, spec, r, dtype, device, train_rows, eval_rows, on_progress)
    else:
        # auto: try the unsloth load path. ANY exception during import, model
        # load, or LoRA wrap falls back to the existing vanilla path unchanged.
        # Once we're past this block the engine is locked in — no switching
        # once the training loop has started.
        try:
            model, tokenizer, train_enc, eval_batches, baseline = _attempt_engine(
                _load_unsloth, _wrap_unsloth, spec, r, dtype, device, train_rows, eval_rows, on_progress)
            engine = resolve_engine(engine_mode, True)
        except Exception as e:
            engine_fallback_reason = type(e).__name__
            print(f"[sgl] unsloth unavailable ({engine_fallback_reason}), falling back to vanilla",
                  file=sys.stderr)
            engine = resolve_engine(engine_mode, False)
            require_engine_for_recipe(engine, r.load_in_4bit)
            model, tokenizer, train_enc, eval_batches, baseline = _attempt_engine(
                _load_vanilla, _wrap_vanilla_lora, spec, r, dtype, device, train_rows, eval_rows, on_progress)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=r.lr)

    steps_per_epoch = -(-len(train_enc) // r.batch_size)  # ceil — _batched yields the trailing partial batch
    total_steps = max(1, steps_per_epoch * r.epochs)
    step = 0
    model.train()
    for epoch in range(r.epochs):
        for ids, attn in _batched(tokenizer, train_enc, r.batch_size, device):
            labels = ids.clone()
            labels[attn == 0] = -100
            loss = model(input_ids=ids, attention_mask=attn, labels=labels).loss
            (loss / r.grad_accum).backward()
            step += 1
            if step % r.grad_accum == 0:
                optim.step()
                optim.zero_grad()
            if step % 10 == 0:
                on_progress("training", 0.10 + 0.75 * (step / total_steps),
                            step=step, total_steps=total_steps,
                            train_loss=float(loss.item()), epoch=epoch)
    optim.step()
    optim.zero_grad()

    on_progress("final_eval", 0.88)
    final = _evaluate(model, eval_batches)

    on_progress("saving", 0.92)
    if engine == "unsloth":
        from .trainer_unsloth import save_unsloth
        save_unsloth(model, tokenizer, out_dirs, r.load_in_4bit)
    else:
        _save_vanilla(model, tokenizer, out_dirs)

    return {"baseline": baseline, "final": final, "steps": step, "seed": r.seed,
            "device": device, "dtype": str(dtype),
            "engine": engine, "engine_fallback_reason": engine_fallback_reason}
