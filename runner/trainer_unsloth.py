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

from .errors import InfraError


def load_unsloth(spec, dtype_hint):
    """FastLanguageModel.from_pretrained -> (model, tokenizer), UNWRAPPED.

    No LoRA here on purpose: the caller runs baseline eval on exactly this
    object before calling wrap_unsloth_lora, so the baseline reflects the
    untouched base model — the same honesty guarantee the vanilla engine
    makes. dtype_hint is accepted for symmetry with the vanilla loader but
    unsloth is left to auto-select (dtype=None is its own "autodetect" value,
    matching what the vanilla loader would have picked on this box anyway).
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=spec.hf_repo,
        revision=spec.hf_revision,
        max_seq_length=spec.recipe.max_seq_len,
        load_in_4bit=spec.recipe.load_in_4bit,
        dtype=None,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def wrap_unsloth_lora(model, recipe):
    """Attach a LoRA adapter via unsloth's own PEFT wrapper.

    Kept as a distinct function (not inlined) so a wrap-time failure has a
    single, obvious call site — see trainer.py's comment on why a wrap
    failure does not retroactively switch engines.
    """
    from unsloth import FastLanguageModel

    return FastLanguageModel.get_peft_model(
        model,
        r=recipe.lora_rank,
        lora_alpha=recipe.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        random_state=recipe.seed,
    )


def save_unsloth(model, tokenizer, out_dirs, load_in_4bit) -> None:
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
    except OSError as e:
        raise InfraError(f"[engine=unsloth] artifact save failed (disk?): {e}") from e
