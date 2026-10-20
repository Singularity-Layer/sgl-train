# Decision fine-tuning provenance and release gate

`decision_finetune` is dark until a provider GPU canary passes. Its runtime is
implemented with permissive components and does not import, vendor, or depend
on Unsloth's AGPL decision-training modules.

## Provenance

The runner uses standard PyTorch, Transformers, PEFT/LoRA, bitsandbytes, and
safetensors components. Its split policy, label mapping, loss, calibration,
metrics, training loop, and artifact staging are implemented in this repository.

The record encoder, joint schema head, release loader, and System One response
helpers are vendored unchanged from `Cloudflare/clef` revision
`0b331204bb13fbd2ca93a64df1956f5b55478ce5`, licensed Apache-2.0. The upstream
license and a source notice are stored beside the vendored file under
`runner/_vendor/cloudflare_clef/` and are copied into delivered model bundles.

Do not copy implementation details from Unsloth's decision modules into this
lane. The existing Unsloth integration for ordinary SFT/QLoRA remains separate
and unchanged.

## Training and evaluation contract

- Input rows are strict `state`, `questions`, and `gold` records.
- A seeded connected-component split creates training, calibration holdout,
  and untouched test sets. `group_key`/`group_id` keeps provider or service
  families together; canonical state+question hashes join duplicates even when
  labels or supplied family keys differ. Exact duplicates are removed.
- The backbone receives PEFT LoRA adapters and the Clef joint head trains with
  option cross-entropy plus a Brier calibration term.
- Scalar temperature calibration fits only the holdout set.
- Baseline and final accuracy, ECE, negative log loss, and record accuracy are
  reported on the untouched test set.
- The artifact contains a System One-compatible Clef head and the fitted
  temperature in `sgl_decision_config.json`.
- Every saved adapter and optional merged bundle is manifest-checked, released
  from the training process, clean-loaded, and required to answer a bounded
  System One probe before its presigned upload can begin.

## Runtime gate

The active `v0.6.0` volume predates this decision lane. Do not mutate that
immutable runtime in place. Candidate `v0.7.0` isolates the stack pinned in
`scripts/decision-runtime-constraints.txt` from the ordinary LoRA/QLoRA
environment. It must pass an NVIDIA/CUDA canary covering model load, NF4
LoRA, a backward step through the head and adapters, calibration, adapter
reload, merged reload, and System One inference before backend enablement.

The production runner remains NVIDIA/CUDA based. A local MLX showcase is a
separate execution target and does not establish hosted-runtime compatibility.

## Artifact boundary

The required artifact contains the PEFT adapter and Clef head
(`joint_head.safetensors` plus its config). The optional merged artifact is a
16-bit safetensors bundle containing the same head and calibration config. This
lane does not create or advertise GGUF. Decision GGUF serving requires its own
verified path and llama.cpp b11443 or newer; a plain language-model GGUF would
silently omit the purchased decision behavior.
