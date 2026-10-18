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
import random
import shutil
import sys

from .errors import InfraError, attach_context
from .evalmath import aggregate


def _save_training_checkpoint(model, tokenizer, optimizer, out_dirs, slot, step, epoch, planned_steps,
                              provenance):
    """Write a complete local checkpoint directory before any remote upload."""
    import torch

    target = out_dirs[f"checkpoint_{slot}_dir"]
    temp = target + ".tmp"
    shutil.rmtree(temp, ignore_errors=True)
    os.makedirs(temp, exist_ok=True)
    model.save_pretrained(os.path.join(temp, "adapter"))
    tokenizer.save_pretrained(os.path.join(temp, "adapter"))
    state = {
        "version": 1,
        "step": step,
        "epoch": epoch,
        "planned_steps": planned_steps,
        **provenance,
        "optimizer": optimizer.state_dict(),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, os.path.join(temp, "training_state.pt"))
    shutil.rmtree(target, ignore_errors=True)
    os.replace(temp, target)
    return target


def _restore_training_checkpoint(model, optimizer, out_dirs, device, planned_steps, resume, provenance):
    """Restore adapter, optimizer, and RNG state from the verified archive."""
    import torch
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

    root = out_dirs["resume_dir"]
    state_path = os.path.join(root, "training_state.pt")
    adapter_dir = os.path.join(root, "adapter")
    if not os.path.isfile(state_path) or not os.path.isdir(adapter_dir):
        raise InfraError("resume checkpoint is missing adapter or training state")
    state = torch.load(state_path, map_location=device, weights_only=True)
    step = int(state.get("step", -1))
    epoch = int(state.get("epoch", -1))
    if step < 0 or step > planned_steps or int(state.get("planned_steps", -1)) != planned_steps:
        raise InfraError("resume checkpoint does not match this Training Recipe")
    if step != resume.step or epoch != resume.epoch:
        raise InfraError("resume checkpoint progress does not match its signed manifest")
    if any(state.get(key) != value for key, value in provenance.items()):
        raise InfraError("resume checkpoint provenance does not match this run")
    weights = load_peft_weights(adapter_dir, device=device)
    result = set_peft_model_state_dict(model, weights, adapter_name="default")
    if getattr(result, "unexpected_keys", None):
        raise InfraError("resume adapter contains unexpected weights")
    optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if device == "cuda" and state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return step, epoch


def _save_adapter_only(model, tokenizer, out_dirs, spec, engine: str) -> None:
    """Save the paid LoRA deliverable without materializing full base weights."""
    try:
        shutil.rmtree(out_dirs["adapter_dir"], ignore_errors=True)
        os.makedirs(out_dirs["adapter_dir"], exist_ok=True)
        model.save_pretrained(out_dirs["adapter_dir"])
        tokenizer.save_pretrained(out_dirs["adapter_dir"])
    except Exception as e:
        shutil.rmtree(out_dirs["adapter_dir"], ignore_errors=True)
        raise attach_context(
            InfraError(f"[engine={engine}] adapter save failed: {e}"),
            engine=engine, trained_repo=spec.hf_repo,
            trained_revision=spec.hf_revision,
        ) from e


def _is_optimizer_boundary(step: int, total_steps: int, grad_accum: int) -> bool:
    return step == total_steps or step % grad_accum == 0


def normalize_engine_mode(env_value: str) -> str:
    """Single parser for the raw SGL_ENGINE string -> {'auto', 'vanilla',
    'unsloth'}. Both resolve_engine (the pure decision table below) and
    run_finetune (which needs the normalized mode BEFORE it knows whether the
    unsloth probe succeeded, to decide which branch to even attempt) call
    this — one parser, so 'strip/lower/default-to-auto' can't drift between
    the two call sites.
    """
    return (env_value or "auto").strip().lower()


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
    v = normalize_engine_mode(env_value)
    if v == "vanilla":
        return "vanilla"
    if v == "unsloth":
        return "unsloth"
    return "unsloth" if unsloth_probe_result else "vanilla"


def unsloth_pin_for(spec, load_in_4bit: bool) -> tuple[str, str] | None:
    """The unsloth mirror pin that MUST be used for this load_in_4bit shape,
    or None if it isn't pinned. Pure, no imports (same rule as resolve_engine/
    require_engine_for_recipe) — used by both the decision layer here AND the
    loader (runner/trainer_unsloth.py's load_unsloth), so the "is unsloth
    usable" check and the actual load can never disagree about which pin pair
    applies.

    Why two separate pins exist at all: unsloth's FastLanguageModel.from_pretrained
    silently remaps a canonical repo name (e.g. "Qwen/Qwen2.5-0.5B-Instruct")
    to a mirror namespace but still applies whatever revision it was given — a
    revision that belongs to the canonical repo, not the mirror. Live GPU
    smoke (2026-08-25) proved this remap goes one level deeper under QLoRA:
    with load_in_4bit=True, unsloth resolves to a THIRD repo — a dedicated
    pre-quantized "-unsloth-bnb-4bit" mirror, distinct from the 16-bit mirror
    — and again applies whatever revision it was handed. So the 16-bit pin
    (unsloth_repo/unsloth_revision) and the 4-bit pin (unsloth_4bit_repo/
    unsloth_4bit_revision) are independent and NEITHER one substitutes for the
    other (see runner/trainer_unsloth.py's load_unsloth docstring).
    """
    repo, revision = ((spec.unsloth_4bit_repo, spec.unsloth_4bit_revision) if load_in_4bit
                      else (spec.unsloth_repo, spec.unsloth_revision))
    if repo is None or revision is None:
        return None
    return repo, revision


def unsloth_pin_ok(spec) -> bool:
    """True iff this spec carries a pinned 16-bit unsloth mirror repo+revision.

    Pure, no imports (same rule as resolve_engine/require_engine_for_recipe).
    spec.py enforces unsloth_repo/unsloth_revision as both-or-neither, so
    checking one here would be enough, but checking both documents the
    invariant this function actually depends on rather than spec.py's.

    This checks the 16-bit pin specifically (see unsloth_pin_for for why there
    are two independent pins). Callers that need to gate on the pin matching
    the recipe's actual load_in_4bit should use unsloth_pin_for(spec,
    load_in_4bit) instead — this function is kept for the non-4bit call sites
    and for backward compatibility with existing callers/tests that only ever
    dealt with the 16-bit case.
    """
    return unsloth_pin_for(spec, False) is not None


def require_engine_for_recipe(engine: str, load_in_4bit: bool) -> None:
    """4-bit (QLoRA) training needs unsloth's bitsandbytes integration — the
    vanilla engine has no 4-bit path at all. Raise InfraError, not SpecError:
    the recipe itself is valid, this box just lacks the capability to honor
    it, and that is our fault, never the buyer's — EXIT_INFRA, refundable.
    """
    if load_in_4bit and engine == "vanilla":
        raise InfraError(
            f"[engine={engine}] 4-bit training requires the unsloth engine, which is unavailable "
            f"on this box")


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


def _model_dtype(model) -> str:
    """What the model ACTUALLY loaded in, read straight off a real parameter —
    not the heuristic `dtype` computed before either engine ran. The vanilla
    loader is handed that heuristic and honors it, but the unsloth loader is
    given dtype=None (unsloth auto-picks), so the heuristic can be a flat lie
    for that engine. One shared helper, called after load for BOTH engines,
    so the reported value is never wrong for either one.
    """
    return str(next(model.parameters()).dtype)


def _load_vanilla(spec, dtype, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(spec.hf_repo, revision=spec.hf_revision)
        model = AutoModelForCausalLM.from_pretrained(
            spec.hf_repo, revision=spec.hf_revision,
            torch_dtype=dtype, device_map=None)
    except Exception as e:
        raise attach_context(
            InfraError(
                f"[engine=vanilla] model download failed for {spec.hf_repo}@{spec.hf_revision}: {e}"),
            engine="vanilla", trained_repo=spec.hf_repo, trained_revision=spec.hf_revision) from e
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model.to(device), tokenizer


def _wrap_vanilla_lora(model, r, spec):
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(model, LoraConfig(
        r=r.lora_rank, lora_alpha=r.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    return model


def _save_vanilla(model, tokenizer, out_dirs, spec) -> None:
    # Training-loop failures above stay EXIT_TRAIN — they're a fact about the run.
    # By the time we get here, training already succeeded and the compute was
    # delivered; ANY exception in this stage (disk full, a merge OOM, a
    # library bug, whatever) is our-side infrastructure, never the buyer's
    # data or recipe, so it always converts to InfraError -> EXIT_INFRA
    # (refundable) — not just OSError.
    try:
        os.makedirs(out_dirs["adapter_dir"], exist_ok=True)
        model.save_pretrained(out_dirs["adapter_dir"])
        tokenizer.save_pretrained(out_dirs["adapter_dir"])

        merged = model.merge_and_unload()
        os.makedirs(out_dirs["merged_dir"], exist_ok=True)
        merged.save_pretrained(out_dirs["merged_dir"], safe_serialization=True)
        tokenizer.save_pretrained(out_dirs["merged_dir"])
    except InfraError:
        raise
    except Exception as e:
        # Best-effort: never leave a half-written merged_dir for a retry to
        # tar up as if it were a complete artifact.
        shutil.rmtree(out_dirs["merged_dir"], ignore_errors=True)
        raise attach_context(
            InfraError(f"[engine=vanilla] artifact save failed: {e}"),
            engine="vanilla", trained_repo=spec.hf_repo, trained_revision=spec.hf_revision) from e


def _load_unsloth(spec, dtype, device):
    from .trainer_unsloth import load_unsloth
    return load_unsloth(spec, dtype)


def _wrap_unsloth(model, r, spec):
    from .trainer_unsloth import wrap_unsloth_lora
    return wrap_unsloth_lora(model, r, spec)


def _load_and_encode(load_fn, spec, r, dtype, device, train_rows, eval_rows, engine_name):
    """Model/tokenizer bring-up, plus the tokenization baseline eval needs.
    Kept together (not split further) because encoding depends on the
    tokenizer load_fn just returned, and both must finish before there is
    anything to run baseline eval ON. This whole function IS the first
    fallback-eligible window for the auto engine in run_finetune: an import
    failure, a load_fn failure, or an empty-dataset RuntimeError here all mean
    "this engine never got off the ground" and are equally true regardless of
    which engine was asked for.

    engine_name is used ONLY to label the empty-dataset exception (e.g.
    "[engine=unsloth] dataset produced no usable sequences...") — a training
    failure's error message should say which engine was actually running,
    since main.py's failed-result payload has no other way to know that if
    train() raised before the caller could record it.
    """
    model, tokenizer = load_fn(spec, dtype, device)

    train_enc = _encode(tokenizer, train_rows, r.max_seq_len)
    eval_enc = _encode(tokenizer, eval_rows, r.max_seq_len)
    if not train_enc or not eval_enc:
        raise RuntimeError(
            f"[engine={engine_name}] dataset produced no usable sequences after tokenization")
    return model, tokenizer, train_enc, eval_enc


def _baseline_eval(tokenizer, model, eval_enc, r, device, on_progress):
    """Eval on the untouched, UNWRAPPED base model — see run_finetune's auto
    branch for why this call is deliberately NOT inside any fallback-eligible
    try: an exception here (bad chat template, a real OOM on an eval batch,
    ...) is a fact about the model/data that is exactly as true for vanilla as
    it is for unsloth, so it must propagate as EXIT_TRAIN rather than
    triggering a doomed-to-fail-identically vanilla re-run first.

    BASELINE FIRST, on the untouched base model. Without this number there is
    nothing to compare the final model to, and "it trained" is not a result.
    """
    on_progress("baseline_eval", 0.10)
    eval_batches = list(_batched(tokenizer, eval_enc, r.batch_size, device))
    baseline = _evaluate(model, eval_batches)
    return eval_batches, baseline


def _attempt_engine(load_fn, wrap_fn, spec, r, dtype, device, train_rows, eval_rows, on_progress,
                     engine_name):
    """Shared load -> encode -> baseline_eval -> wrap sequence, parameterized
    by engine-specific load/wrap functions. Used for both engines (and for the
    forced SGL_ENGINE=vanilla/unsloth modes, and for the auto engine's full
    vanilla retry) so the baseline honesty guarantee and the eval math are
    identical regardless of which engine ran — that's what keeps
    baseline/final numbers comparable across engines. The auto engine's OWN
    first attempt does NOT go through this composed function — see
    run_finetune's auto branch, which calls _load_and_encode and
    _baseline_eval directly so it can narrow the fallback window to bring-up
    only.
    """
    model, tokenizer, train_enc, eval_enc = _load_and_encode(
        load_fn, spec, r, dtype, device, train_rows, eval_rows, engine_name)
    eval_batches, baseline = _baseline_eval(tokenizer, model, eval_enc, r, device, on_progress)
    model = wrap_fn(model, r, spec)
    return model, tokenizer, train_enc, eval_batches, baseline


def _fallback_to_vanilla(exc, engine_mode, spec, r, dtype, device, train_rows, eval_rows, on_progress):
    """Shared discard-and-rerun path for BOTH of the auto engine's
    fallback-eligible windows (bring-up, and wrap — see run_finetune). Reseeds
    past whatever RNG the discarded unsloth attempt consumed, then runs a
    full, fresh vanilla _attempt_engine: vanilla needs its OWN baseline,
    evaluated on its OWN loaded model, to be comparable to its own final
    number, so a wrap-time failure means any unsloth baseline already
    computed is not reusable and must be thrown away along with it.
    """
    reason = type(exc).__name__
    _set_all_seeds(r.seed)
    engine = resolve_engine(engine_mode, False)
    require_engine_for_recipe(engine, r.load_in_4bit)
    print(f"[sgl] [engine=unsloth] unsloth unavailable ({reason}), "
          f"falling back to vanilla", file=sys.stderr)
    model, tokenizer, train_enc, eval_batches, baseline = _attempt_engine(
        _load_vanilla, _wrap_vanilla_lora, spec, r, dtype, device, train_rows, eval_rows, on_progress,
        engine_name="vanilla")
    return reason, engine, model, tokenizer, train_enc, eval_batches, baseline


def run_finetune(spec, train_rows, eval_rows, out_dirs, on_progress, on_checkpoint=None,
                 runtime=None) -> dict:
    import torch
    from .budget import RuntimeClock

    r = spec.recipe
    runtime = runtime or RuntimeClock(getattr(spec, "runtime", None))
    setup_started = runtime.phase_started()
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

    v = normalize_engine_mode(engine_mode)
    if v == "vanilla":
        # Skip the unsloth attempt entirely (rule: 'vanilla' never tries it).
        engine = resolve_engine(engine_mode, False)
        require_engine_for_recipe(engine, r.load_in_4bit)
        model, tokenizer, train_enc, eval_batches, baseline = _attempt_engine(
            _load_vanilla, _wrap_vanilla_lora, spec, r, dtype, device, train_rows, eval_rows, on_progress,
            engine_name="vanilla")
    elif v == "unsloth":
        # Forced: no fallback (testing mode). Any exception — import, load, an
        # empty-dataset RuntimeError, a wrap failure — propagates as-is. But a
        # missing mirror pin FOR THE REQUESTED load_in_4bit SHAPE is checked
        # BEFORE any of that: attempting the load anyway risks the silent
        # remap-but-not-revision bug (see unsloth_pin_for), and forcing the
        # engine does not mean forcing that bug too — this is our capability
        # gap, refundable, never the buyer's.
        if unsloth_pin_for(spec, r.load_in_4bit) is None:
            if r.load_in_4bit:
                raise InfraError(
                    "[engine=unsloth] no 4-bit unsloth mirror pinned for this base model")
            raise InfraError(
                "[engine=unsloth] no unsloth mirror pinned for this base model")
        engine = resolve_engine(engine_mode, True)
        model, tokenizer, train_enc, eval_batches, baseline = _attempt_engine(
            _load_unsloth, _wrap_unsloth, spec, r, dtype, device, train_rows, eval_rows, on_progress,
            engine_name="unsloth")
    elif unsloth_pin_for(spec, r.load_in_4bit) is None:
        if r.load_in_4bit:
            # 4-bit was requested but no 4-bit mirror is pinned: unsloth
            # cannot serve this recipe (see unsloth_pin_for), and vanilla has
            # NO 4-bit path at all (require_engine_for_recipe). There is
            # nothing to silently fall back TO — falling back to vanilla
            # would just trade this loud, refundable failure for a doomed
            # attempt that fails the same way one call later. Raise
            # immediately rather than burning that attempt first.
            raise InfraError(
                "[engine=unsloth] no 4-bit unsloth mirror pinned for this base model")
        # auto, non-4bit, but no 16-bit unsloth mirror is pinned for this base
        # model: never attempt unsloth at all — no import, no probe, no
        # wasted load. This is cheaper than the bring-up fallback below AND
        # avoids the silent remap-but-not-revision bug entirely rather than
        # triggering and then recovering from it (see unsloth_pin_for).
        # Nothing here touches RNG state, so — unlike _fallback_to_vanilla —
        # there is nothing to reseed past.
        engine_fallback_reason = "NoUnslothPin"
        engine = resolve_engine(engine_mode, False)
        require_engine_for_recipe(engine, r.load_in_4bit)
        print(f"[sgl] [engine=unsloth] unsloth unavailable ({engine_fallback_reason}), "
              f"falling back to vanilla", file=sys.stderr)
        model, tokenizer, train_enc, eval_batches, baseline = _attempt_engine(
            _load_vanilla, _wrap_vanilla_lora, spec, r, dtype, device, train_rows, eval_rows, on_progress,
            engine_name="vanilla")
    else:
        # auto: the fallback try covers ONLY bring-up (import + load_unsloth +
        # the tokenization it needs) and, separately, the LoRA wrap — NOT
        # baseline eval, which sits between the two and propagates raw. Once
        # we're past this block the engine is locked in — no switching once
        # the training loop has started.
        try:
            model, tokenizer, train_enc, eval_enc = _load_and_encode(
                _load_unsloth, spec, r, dtype, device, train_rows, eval_rows,
                engine_name="unsloth")
        except Exception as e:
            # The discarded unsloth attempt (model load, RNG-consuming init,
            # etc.) already advanced/clobbered global RNG state; _fallback_to_vanilla
            # reseeds before the retry so the vanilla run is bit-identical to a
            # pure-vanilla run with the same seed.
            (engine_fallback_reason, engine, model, tokenizer, train_enc, eval_batches,
             baseline) = _fallback_to_vanilla(
                e, engine_mode, spec, r, dtype, device, train_rows, eval_rows, on_progress)
        else:
            eval_batches, baseline = _baseline_eval(tokenizer, model, eval_enc, r, device, on_progress)
            try:
                model = _wrap_unsloth(model, r, spec)
            except Exception as e:
                # A wrap failure discards this (valid) unsloth baseline too —
                # see _fallback_to_vanilla's docstring for why that full
                # discard-and-rerun is correct, not wasteful.
                (engine_fallback_reason, engine, model, tokenizer, train_enc, eval_batches,
                 baseline) = _fallback_to_vanilla(
                    e, engine_mode, spec, r, dtype, device, train_rows, eval_rows, on_progress)
            else:
                engine = resolve_engine(engine_mode, True)

    # Report what the model ACTUALLY loaded in (see _model_dtype), not the
    # heuristic `dtype` computed above — that heuristic is what vanilla was
    # asked for, but unsloth was handed dtype=None and auto-picked its own.
    actual_dtype = _model_dtype(model)

    runtime.record_phase("model_setup_and_preprocess", setup_started)
    if runtime.should_finalize():
        raise InfraError("runtime budget entered finalization reserve before training could start")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=r.lr)

    steps_per_epoch = -(-len(train_enc) // r.batch_size)  # ceil — _batched yields the trailing partial batch
    total_steps = max(1, steps_per_epoch * r.epochs)
    from .checkpoint import checkpoint_provenance, validate_resume_provenance
    provenance = checkpoint_provenance(spec, train_rows, eval_rows)
    step = 0
    if getattr(spec, "resume", None) is not None:
        try:
            validate_resume_provenance(spec.resume, provenance, total_steps)
        except ValueError as e:
            raise InfraError(str(e)) from e
        if not _is_optimizer_boundary(spec.resume.step, total_steps, r.grad_accum):
            raise InfraError("resume checkpoint is not at an optimizer boundary")
        expected_epoch = 0 if spec.resume.step == 0 else ((spec.resume.step - 1) // steps_per_epoch) + 1
        if spec.resume.epoch != expected_epoch:
            raise InfraError("resume checkpoint epoch does not match its step")
        step, _ = _restore_training_checkpoint(
            model, optim, out_dirs, device, total_steps, spec.resume, provenance)
        on_progress("training", 0.10 + 0.75 * (step / total_steps),
                    step=step, total_steps=total_steps, resumed_from_step=step)
    early_finalized = False
    last_checkpoint_step = 0
    batch_index = 0
    training_started = runtime.phase_started()
    finalize_requested = False
    model.train()
    for epoch in range(r.epochs):
        for ids, attn in _batched(tokenizer, train_enc, r.batch_size, device):
            batch_index += 1
            if batch_index <= step:
                continue
            labels = ids.clone()
            labels[attn == 0] = -100
            loss = model(input_ids=ids, attention_mask=attn, labels=labels).loss
            (loss / r.grad_accum).backward()
            step += 1
            boundary = _is_optimizer_boundary(step, total_steps, r.grad_accum)
            if boundary:
                optim.step()
                optim.zero_grad()
            if step % 10 == 0 or runtime.should_finalize():
                on_progress("training", 0.10 + 0.75 * (step / total_steps),
                            step=step, total_steps=total_steps,
                            train_loss=float(loss.item()), epoch=epoch + 1,
                            total_epochs=r.epochs,
                            elapsed_seconds=round(runtime.elapsed_seconds(), 3),
                            estimated_remaining_seconds=runtime.estimated_remaining_seconds(
                                step, total_steps, training_started))
            if runtime.should_finalize():
                finalize_requested = True
            if finalize_requested and boundary:
                early_finalized = step < total_steps
                break
        if early_finalized:
            break
        if (on_checkpoint is not None and step > last_checkpoint_step
                and _is_optimizer_boundary(step, total_steps, r.grad_accum)):
            slot = on_checkpoint.next_slot()
            checkpoint_dir = _save_training_checkpoint(
                model, tokenizer, optim, out_dirs, slot, step, epoch + 1, total_steps,
                provenance)
            remote = on_checkpoint(slot, checkpoint_dir, step, epoch + 1, total_steps)
            if remote:
                on_progress("training", 0.10 + 0.75 * (step / total_steps),
                            step=step, total_steps=total_steps, epoch=epoch + 1,
                            total_epochs=r.epochs, **remote)
            last_checkpoint_step = step
    if early_finalized and on_checkpoint is not None and step > last_checkpoint_step:
        slot = on_checkpoint.next_slot()
        checkpoint_dir = _save_training_checkpoint(
            model, tokenizer, optim, out_dirs, slot, step, min(r.epochs, epoch + 1), total_steps,
            provenance)
        remote = on_checkpoint(slot, checkpoint_dir, step, min(r.epochs, epoch + 1), total_steps)
        if remote:
            on_progress("training", 0.10 + 0.75 * (step / total_steps),
                        step=step, total_steps=total_steps, epoch=min(r.epochs, epoch + 1),
                        total_epochs=r.epochs, **remote)
    on_progress("final_eval", 0.88)
    finalization_started = runtime.phase_started()
    final = _evaluate(model, eval_batches)

    on_progress("saving", 0.92)
    if early_finalized:
        _save_adapter_only(model, tokenizer, out_dirs, spec, engine)
    elif engine == "unsloth":
        from .trainer_unsloth import save_unsloth
        save_unsloth(model, tokenizer, out_dirs, r.load_in_4bit, spec)
    else:
        _save_vanilla(model, tokenizer, out_dirs, spec)
    runtime.record_phase("final_eval_and_save", finalization_started)

    # Honesty: which repo+revision ACTUALLY trained. vanilla trains spec.hf_repo
    # @ spec.hf_revision directly; unsloth trains the pinned MIRROR instead (see
    # unsloth_pin_for / trainer_unsloth.load_unsloth) — the report must say so
    # rather than implying the canonical repo was what produced these weights.
    # unsloth_pin_for(spec, r.load_in_4bit) is the SAME call the engine
    # decision above used to confirm unsloth was even attemptable, so this can
    # never disagree with what was actually loaded: it reports the 4-BIT
    # mirror when load_in_4bit ran, since those are the weights that actually
    # trained, not the 16-bit pair.
    if engine == "unsloth":
        trained_repo, trained_revision = unsloth_pin_for(spec, r.load_in_4bit)
    else:
        trained_repo, trained_revision = spec.hf_repo, spec.hf_revision

    return {"baseline": baseline, "final": final, "steps": step,
            "planned_steps": total_steps, "early_finalized": early_finalized,
            "finalization_reason": "runtime_reserve" if early_finalized else "training_complete",
            "runtime_phase_seconds": runtime.phase_seconds,
            "seed": r.seed,
            "device": device, "dtype": actual_dtype,
            "engine": engine, "engine_fallback_reason": engine_fallback_reason,
            "trained_repo": trained_repo, "trained_revision": trained_revision}


def evaluate_rows(model_ref: str, revision: str | None, rows, recipe) -> dict:
    """Token-weighted eval of a saved model (a local merged dir, or a pinned
    hub repo@revision) on `rows`. Used by auto-tune to score the winner and
    the untouched base on the TEST split, which no trial ever selected on —
    so the number in the report is not inflated by the selection itself.
    Plain transformers load: this is measurement, not training, and must not
    depend on which engine trained the weights.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (
        torch.float16 if device == "cuda" else torch.float32)
    kw = {"revision": revision} if revision else {}
    tokenizer = AutoTokenizer.from_pretrained(model_ref, **kw)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_ref, torch_dtype=dtype, device_map=None, **kw).to(device)
    try:
        enc = _encode(tokenizer, rows, recipe.max_seq_len)
        if not enc:
            raise RuntimeError("test split produced no usable sequences after tokenization")
        batches = list(_batched(tokenizer, enc, recipe.batch_size, device))
        return _evaluate(model, batches)
    finally:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
