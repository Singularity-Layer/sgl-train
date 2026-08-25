"""Optional unsloth engine. Imports are ALL function-local, same rule as
trainer.py: this module must be importable (though never actually called) on a
laptop with no CUDA and no unsloth installed.

trainer.py drives the shared load -> baseline_eval -> wrap -> train -> eval ->
save sequence; this module only supplies the unsloth-specific load/wrap/save
steps so the two engines produce numbers and artifacts that are directly
comparable.
"""
from __future__ import annotations

import os
import shutil

from .errors import InfraError, attach_context


def load_unsloth(spec, dtype_hint):
    """FastLanguageModel.from_pretrained -> (model, tokenizer), UNWRAPPED.

    No LoRA here on purpose: the caller runs baseline eval on exactly this
    object before calling wrap_unsloth_lora, so the baseline reflects the
    untouched base model — the same honesty guarantee the vanilla engine
    makes. dtype_hint is accepted for symmetry with the vanilla loader but
    unsloth is left to auto-select (dtype=None is its own "autodetect" value,
    matching what the vanilla loader would have picked on this box anyway).

    model_name/revision are spec.unsloth_repo/spec.unsloth_revision — NEVER
    spec.hf_repo/spec.hf_revision. Live GPU smoke (2026-08-25) proved
    FastLanguageModel.from_pretrained SILENTLY remaps a canonical repo name
    (e.g. "Qwen/Qwen2.5-0.5B-Instruct") to unsloth's own mirror namespace
    ("unsloth/Qwen2.5-0.5B-Instruct") but still applies whatever revision it
    was handed — and that sha is the CANONICAL repo's, which does not exist in
    the mirror repo, so the load failed with "AutoConfig error: Unrecognized
    model in unsloth/Qwen2.5-0.5B-Instruct". Passing the mirror's OWN
    repo+revision (validated as a matched pair in spec.py, and required by
    trainer.py's unsloth_pin_ok before this function is ever called) makes
    unsloth's internal remap a no-op: the name it resolves to is the name we
    already gave it, and the revision is that mirror's own sha — never a
    surprise substitution.
    """
    from unsloth import FastLanguageModel

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=spec.unsloth_repo,
            revision=spec.unsloth_revision,
            max_seq_length=spec.recipe.max_seq_len,
            load_in_4bit=spec.recipe.load_in_4bit,
            dtype=None,
        )
    except Exception as e:
        raise attach_context(
            e,
            engine="unsloth", trained_repo=spec.unsloth_repo, trained_revision=spec.unsloth_revision) from e
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def wrap_unsloth_lora(model, recipe, spec):
    """Attach a LoRA adapter via unsloth's own PEFT wrapper.

    Kept as a distinct function (not inlined) so a wrap-time failure has a
    single, obvious call site — see trainer.py's comment on why a wrap
    failure does not retroactively switch engines.
    """
    from unsloth import FastLanguageModel

    try:
        return FastLanguageModel.get_peft_model(
            model,
            r=recipe.lora_rank,
            lora_alpha=recipe.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            random_state=recipe.seed,
        )
    except Exception as e:
        raise attach_context(
            e,
            engine="unsloth", trained_repo=spec.unsloth_repo, trained_revision=spec.unsloth_revision) from e


def save_unsloth(model, tokenizer, out_dirs, load_in_4bit, spec) -> None:
    """Write the SAME on-disk layout the vanilla path writes: an adapter_dir
    (the LoRA adapter alone) and a merged_dir (full merged weights). main.py's
    tar/convert stages read both of these by directory name and don't know or
    care which engine produced them.

    4-bit base weights can't be folded back with plain merge_and_unload — the
    base tensors are quantized, so there is nothing dense to add the LoRA
    delta into. unsloth's save_pretrained_merged(..., save_method="merged_16bit")
    dequantizes the base, merges, and writes the result in one call; that path
    is used only when load_in_4bit was actually true.
    """
    try:
        os.makedirs(out_dirs["adapter_dir"], exist_ok=True)
        model.save_pretrained(out_dirs["adapter_dir"])
        tokenizer.save_pretrained(out_dirs["adapter_dir"])

        os.makedirs(out_dirs["merged_dir"], exist_ok=True)
        if load_in_4bit:
            model.save_pretrained_merged(
                out_dirs["merged_dir"], tokenizer, save_method="merged_16bit")
        else:
            merged = model.merge_and_unload()
            merged.save_pretrained(out_dirs["merged_dir"], safe_serialization=True)
            tokenizer.save_pretrained(out_dirs["merged_dir"])
    except InfraError:
        raise
    except Exception as e:
        # Training already succeeded and delivered the compute by the time we
        # get here — ANY save-stage exception (disk, a dequant/merge failure,
        # a library bug) is our-side infrastructure, not the buyer's data or
        # recipe. Never leave a half-written merged_dir for a retry to tar up
        # as if it were a complete artifact.
        shutil.rmtree(out_dirs["merged_dir"], ignore_errors=True)
        raise attach_context(
            InfraError(f"[engine=unsloth] artifact save failed: {e}"),
            engine="unsloth", trained_repo=spec.unsloth_repo, trained_revision=spec.unsloth_revision) from e
