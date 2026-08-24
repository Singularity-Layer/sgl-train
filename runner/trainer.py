"""LoRA fine-tune plus held-out eval. THE ONLY torch module.

torch is imported inside functions, never at module scope, so the rest of the
package (and the whole test suite) imports on a laptop with no CUDA.
"""
from __future__ import annotations

import os

from .errors import InfraError
from .evalmath import aggregate


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


def run_finetune(spec, train_rows, eval_rows, out_dirs, on_progress) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    on_progress("loading_model", 0.05, model=spec.hf_repo)
    try:
        tokenizer = AutoTokenizer.from_pretrained(spec.hf_repo, revision=spec.hf_revision)
        model = AutoModelForCausalLM.from_pretrained(
            spec.hf_repo, revision=spec.hf_revision,
            torch_dtype=dtype, device_map=None)
    except Exception as e:
        raise InfraError(f"model download failed for {spec.hf_repo}@{spec.hf_revision}: {e}") from e
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device)

    train_enc = _encode(tokenizer, train_rows, r.max_seq_len)
    eval_enc = _encode(tokenizer, eval_rows, r.max_seq_len)
    if not train_enc or not eval_enc:
        raise RuntimeError("dataset produced no usable sequences after tokenization")

    # BASELINE FIRST, on the untouched base model. Without this number there is
    # nothing to compare the final model to, and "it trained" is not a result.
    on_progress("baseline_eval", 0.10)
    eval_batches = list(_batched(tokenizer, eval_enc, r.batch_size, device))
    baseline = _evaluate(model, eval_batches)

    model = get_peft_model(model, LoraConfig(
        r=r.lora_rank, lora_alpha=r.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

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
    os.makedirs(out_dirs["adapter_dir"], exist_ok=True)
    model.save_pretrained(out_dirs["adapter_dir"])
    tokenizer.save_pretrained(out_dirs["adapter_dir"])

    merged = model.merge_and_unload()
    os.makedirs(out_dirs["merged_dir"], exist_ok=True)
    merged.save_pretrained(out_dirs["merged_dir"], safe_serialization=True)
    tokenizer.save_pretrained(out_dirs["merged_dir"])

    return {"baseline": baseline, "final": final, "steps": step, "seed": r.seed,
            "device": device, "dtype": str(dtype)}
