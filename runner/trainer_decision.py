"""Permissive Clef-compatible decision fine-tuning.

The orchestration is backend-injected for GPU-free tests. The production
backend uses PyTorch, Transformers and PEFT plus Cloudflare's Apache-2.0 Clef
record encoder/head. It does not import Unsloth decision code.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import DataError, InfraError

_HEAD_FILES = ("joint_head.safetensors", "joint_head_config.json")
_ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")
_COMMON_FILES = _HEAD_FILES + (
    "sgl_decision_config.json",
    "joint_schema_model.py",
    "sgl_systemone.py",
    "LICENSE.cloudflare-clef",
    "NOTICE.cloudflare-clef",
)


@dataclass(frozen=True)
class DecisionItem:
    encoded: Any
    labels: tuple[int, ...]


@dataclass
class DecisionSession:
    model: Any
    processor: Any
    tokenizer: Any
    device: Any
    dtype: Any
    head_config: dict[str, int | float]


def _trained_pin(spec) -> tuple[str, str]:
    pair = (spec.decision_repo, spec.decision_revision)
    if not all(pair):
        raise InfraError("[engine=clef-peft] a pinned base model is required")
    return pair  # type: ignore[return-value]


def _label_for(question: dict, answer: object, option_ids: tuple[str, ...]) -> int:
    if question["type"] == "noul":
        key = "true" if answer is True else "false"
    else:
        key = str(answer)
    try:
        return option_ids.index(key)
    except ValueError:
        raise ValueError("gold answer is absent from encoded options") from None


def _softmax(values: list[float], temperature: float = 1.0) -> list[float]:
    scaled = [v / temperature for v in values]
    peak = max(scaled)
    exps = [math.exp(v - peak) for v in scaled]
    total = sum(exps)
    return [v / total for v in exps]


def decision_metrics(
    logits: list[list[float]], labels: list[int], rows: list[int], temperature: float = 1.0,
) -> dict[str, float]:
    if not logits or len(logits) != len(labels) or len(rows) != len(labels):
        raise ValueError("decision metrics need aligned non-empty logits, labels, and rows")
    correct, confidences, losses = [], [], []
    record_correct: dict[int, bool] = {}
    for values, label, row in zip(logits, labels, rows):
        probs = _softmax(values, temperature)
        prediction = max(range(len(probs)), key=probs.__getitem__)
        hit = prediction == label
        correct.append(float(hit))
        confidences.append(max(probs))
        losses.append(-math.log(max(probs[label], 1e-12)))
        record_correct[row] = record_correct.get(row, True) and hit
    ece = 0.0
    for index in range(10):
        lo, hi = index / 10, (index + 1) / 10
        members = [i for i, confidence in enumerate(confidences)
                   if lo <= confidence < hi or (index == 9 and confidence == 1.0)]
        if members:
            confidence = sum(confidences[i] for i in members) / len(members)
            accuracy = sum(correct[i] for i in members) / len(members)
            ece += len(members) / len(labels) * abs(confidence - accuracy)
    return {
        "accuracy": sum(correct) / len(correct),
        "ece": ece,
        "loss": sum(losses) / len(losses),
        "record_accuracy": sum(record_correct.values()) / len(record_correct),
    }


def fit_temperature(logits: list[list[float]], labels: list[int], rows: list[int]) -> tuple[float, dict]:
    """Deterministic held-out scalar calibration; no test labels are fitted."""
    candidates = [math.exp(math.log(0.25) + i * (math.log(4.0 / 0.25) / 100))
                  for i in range(101)]
    best = min(candidates, key=lambda value: decision_metrics(logits, labels, rows, value)["loss"])
    return best, decision_metrics(logits, labels, rows, best)


class TorchDecisionBackend:
    """Standard PyTorch/PEFT implementation around Cloudflare's Apache Clef head."""

    def __init__(self) -> None:
        try:
            import torch
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from safetensors.torch import save_file
            from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
            from ._vendor.cloudflare_clef import joint_schema_model as clef
        except Exception as exc:  # noqa: BLE001
            raise InfraError("[engine=clef-peft] decision runtime is unavailable") from exc
        self.torch = torch
        self.LoraConfig = LoraConfig
        self.get_peft_model = get_peft_model
        self.prepare_model_for_kbit_training = prepare_model_for_kbit_training
        self.save_file = save_file
        self.AutoModelForMultimodalLM = AutoModelForMultimodalLM
        self.AutoProcessor = AutoProcessor
        self.BitsAndBytesConfig = BitsAndBytesConfig
        self.clef = clef

    def cuda_available(self) -> bool:
        return bool(self.torch.cuda.is_available())

    def _target_modules(self, backbone) -> list[str]:
        """Select language-backbone linear layers without touching vision or lm_head."""
        text_model = backbone.model
        if hasattr(text_model, "language_model"):
            text_model = text_model.language_model
        prefix = next(
            (name for name, module in backbone.named_modules() if module is text_model), None)
        if prefix is None:
            raise InfraError("[engine=clef-peft] cannot locate the language backbone")
        names = []
        for name, module in backbone.named_modules():
            if not name.startswith(prefix + "."):
                continue
            if name.endswith(("lm_head", "embed_tokens")):
                continue
            if isinstance(module, self.torch.nn.Linear) or module.__class__.__name__ in {
                "Linear4bit", "Linear8bitLt"
            }:
                names.append(name)
        if not names:
            raise InfraError("[engine=clef-peft] language backbone has no LoRA target layers")
        return names

    def load(self, spec) -> DecisionSession:
        torch = self.torch
        repo, revision = _trained_pin(spec)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
                 else torch.float16 if device.type == "cuda" else torch.float32)
        processor = self.AutoProcessor.from_pretrained(repo, revision=revision)
        tokenizer = processor.tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        kwargs: dict[str, Any] = {"revision": revision, "torch_dtype": dtype}
        if spec.recipe.load_in_4bit:
            kwargs["quantization_config"] = self.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
            kwargs["device_map"] = {"": 0}
        backbone = self.AutoModelForMultimodalLM.from_pretrained(repo, **kwargs)
        if not spec.recipe.load_in_4bit:
            backbone.to(device)
        else:
            backbone = self.prepare_model_for_kbit_training(
                backbone, use_gradient_checkpointing=True)
        backbone.config.use_cache = False
        lora = self.LoraConfig(
            r=spec.recipe.lora_rank,
            lora_alpha=spec.recipe.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            target_modules=self._target_modules(backbone),
        )
        backbone = self.get_peft_model(backbone, lora)
        if hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()

        config = getattr(backbone.get_base_model(), "config", backbone.config)
        hidden = getattr(config, "hidden_size", None)
        if hidden is None and getattr(config, "text_config", None) is not None:
            hidden = config.text_config.hidden_size
        if not isinstance(hidden, int) or hidden <= 0:
            raise InfraError("[engine=clef-peft] base model does not expose a hidden size")
        width = min(512, hidden)
        heads = next((n for n in (8, 4, 2, 1) if width % n == 0), 1)
        head_config = {
            "hidden_size": hidden,
            "width": width,
            "routing_layers": 1,
            "layers": 2,
            "heads": heads,
            "feedforward": width * 4,
            "dropout": 0.0,
        }
        torch.manual_seed(spec.recipe.seed)
        head = self.clef.JointSchemaHead(**head_config).to(device=device, dtype=torch.float32)
        model = self.clef.ClefModel(backbone, head)
        return DecisionSession(model, processor, tokenizer, device, dtype, head_config)

    def build_items(self, rows: list[dict], session: DecisionSession, max_length: int) -> list[DecisionItem]:
        items = []
        for row_index, row in enumerate(rows):
            record = {**row, "id": str(row_index)}
            encoded = self.clef.encode_record(session.tokenizer, record, max_length=max_length)
            labels = tuple(
                _label_for(row["questions"][q.question_id], row["gold"][q.question_id], q.option_ids)
                for q in encoded.questions
            )
            items.append(DecisionItem(encoded, labels))
        return items

    def _collect_logits(self, session: DecisionSession, items: list[DecisionItem], batch_size: int):
        torch = self.torch
        session.model.eval()
        logits, labels, rows = [], [], []
        with torch.no_grad():
            for start in range(0, len(items), batch_size):
                chunk = items[start:start + batch_size]
                batch = self.clef.collate_records(
                    [item.encoded for item in chunk], session.tokenizer.pad_token_id, session.device)
                with torch.autocast(
                    device_type=session.device.type,
                    dtype=session.dtype,
                    enabled=session.device.type == "cuda",
                ):
                    output = session.model(batch)
                for offset, (record_logits, item) in enumerate(zip(output, chunk)):
                    for values, label in zip(record_logits, item.labels):
                        logits.append([float(v) for v in values.detach().float().cpu().tolist()])
                        labels.append(label)
                        rows.append(start + offset)
        return logits, labels, rows

    def evaluate(
        self, session: DecisionSession, items: list[DecisionItem], batch_size: int,
        temperature: float = 1.0,
    ) -> dict[str, float]:
        return decision_metrics(*self._collect_logits(session, items, batch_size), temperature)

    def calibrate(self, session: DecisionSession, items: list[DecisionItem], batch_size: int):
        return fit_temperature(*self._collect_logits(session, items, batch_size))

    def _loss(self, output, items: list[DecisionItem]):
        torch = self.torch
        losses = []
        for record_logits, item in zip(output, items):
            for logits, label in zip(record_logits, item.labels):
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                classes = logits.numel()
                target = torch.full_like(log_probs, 0.05 / classes)
                target[label] += 0.95
                cross_entropy = -(target * log_probs).sum()
                one_hot = torch.zeros_like(log_probs)
                one_hot[label] = 1.0
                brier = ((log_probs.exp() - one_hot) ** 2).sum()
                losses.append(cross_entropy + brier)
        return torch.stack(losses).mean()

    def train(self, session, items, recipe, on_progress) -> int:
        torch = self.torch
        language = [p for p in session.model.language_model.parameters() if p.requires_grad]
        head = [p for p in session.model.head.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([
            {"params": language, "lr": recipe.lr},
            {"params": head, "lr": 1e-4},
        ], weight_decay=0.01)
        use_scaler = session.device.type == "cuda" and session.dtype == torch.float16
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
        optimizer.zero_grad(set_to_none=True)
        batches_per_epoch = math.ceil(len(items) / recipe.batch_size)
        micro_steps = batches_per_epoch * recipe.epochs
        optimizer_steps = 0
        accumulated = 0
        session.model.train()
        for epoch in range(recipe.epochs):
            order = list(range(len(items)))
            random.Random(recipe.seed + epoch).shuffle(order)
            for start in range(0, len(order), recipe.batch_size):
                chunk = [items[index] for index in order[start:start + recipe.batch_size]]
                batch = self.clef.collate_records(
                    [item.encoded for item in chunk], session.tokenizer.pad_token_id, session.device)
                with torch.autocast(
                    device_type=session.device.type,
                    dtype=session.dtype,
                    enabled=session.device.type == "cuda",
                ):
                    loss = self._loss(session.model(batch), chunk) / recipe.grad_accum
                scaler.scale(loss).backward()
                accumulated += 1
                is_last = start + recipe.batch_size >= len(order) and epoch + 1 == recipe.epochs
                if accumulated % recipe.grad_accum == 0 or is_last:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(language + head, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                done = epoch * batches_per_epoch + start // recipe.batch_size + 1
                on_progress(
                    "training", 0.20 + 0.62 * done / micro_steps,
                    step=done, total_steps=micro_steps,
                    train_loss=float(loss.detach().float().cpu()) * recipe.grad_accum,
                    epoch=epoch,
                )
        return optimizer_steps

    def _write_shared(self, session, directory: Path, temperature: float, spec) -> None:
        self.save_file(
            {name: value.detach().cpu().contiguous()
             for name, value in session.model.head.state_dict().items()},
            str(directory / _HEAD_FILES[0]),
        )
        (directory / _HEAD_FILES[1]).write_text(
            json.dumps(session.head_config, indent=2) + "\n", encoding="utf-8")
        (directory / "sgl_decision_config.json").write_text(json.dumps({
            "schema": 1,
            "architecture": "cloudflare-clef-joint-schema",
            "base_model": spec.decision_repo,
            "base_revision": spec.decision_revision,
            "catalog_model": spec.hf_repo,
            "catalog_revision": spec.hf_revision,
            "temperature": temperature,
            "question_types": ["choice", "noul", "score"],
        }, indent=2) + "\n", encoding="utf-8")
        source = Path(self.clef.__file__).resolve()
        shutil.copyfile(source, directory / "joint_schema_model.py")
        serving = Path(__file__).with_name("decision_systemone.py")
        shutil.copyfile(serving, directory / "sgl_systemone.py")
        vendor = source.parent
        shutil.copyfile(vendor / "LICENSE", directory / "LICENSE.cloudflare-clef")
        shutil.copyfile(vendor / "NOTICE", directory / "NOTICE.cloudflare-clef")

    def save_adapter(self, session, directory: str, temperature: float, spec) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        session.model.language_model.save_pretrained(str(target), safe_serialization=True)
        session.processor.save_pretrained(str(target))
        self._write_shared(session, target, temperature, spec)

    def save_merged(self, session, directory: str, temperature: float, spec) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        merged = session.model.language_model.merge_and_unload(safe_merge=True)
        merged.save_pretrained(str(target), safe_serialization=True)
        session.processor.save_pretrained(str(target))
        self._write_shared(session, target, temperature, spec)

    @staticmethod
    def describe(session: DecisionSession) -> tuple[str, str]:
        return str(session.device), str(session.dtype)


def _replace_dir(stage: Path, final: Path) -> None:
    if final.exists():
        shutil.rmtree(final)
    stage.replace(final)


def _stage_save(final_text: str, save, verify) -> None:
    final = Path(final_text)
    stage = final.with_name(final.name + ".staging")
    shutil.rmtree(stage, ignore_errors=True)
    stage.parent.mkdir(parents=True, exist_ok=True)
    try:
        save(str(stage))
        complete = all((stage / name).is_file() for name in _COMMON_FILES) and verify(stage)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    if not complete:
        shutil.rmtree(stage, ignore_errors=True)
        raise InfraError("[engine=clef-peft] decision artifact save was incomplete")
    _replace_dir(stage, final)


def run_decision_finetune(
    spec,
    train_rows,
    holdout_rows,
    test_rows,
    out_dirs,
    on_progress,
    backend=None,
) -> dict:
    backend = backend or TorchDecisionBackend()
    if not backend.cuda_available() and os.environ.get("SGL_ALLOW_CPU") != "1":
        raise InfraError("[engine=clef-peft] CUDA is unavailable on this box")
    repo, revision = _trained_pin(spec)
    r = spec.recipe
    on_progress("loading_model", 0.06)
    try:
        session = backend.load(spec)
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, InfraError):
            raise
        raise InfraError("[engine=clef-peft] model or LoRA/head setup failed") from exc
    on_progress("encoding_dataset", 0.10)
    try:
        train_items = backend.build_items(train_rows, session, r.max_seq_len)
        holdout_items = backend.build_items(holdout_rows, session, r.max_seq_len)
        test_items = backend.build_items(test_rows, session, r.max_seq_len)
    except ValueError as exc:
        raise DataError("decision dataset cannot be encoded within the selected model context") from exc
    if not train_items or not holdout_items or not test_items:
        raise DataError("decision encoder produced an empty split")

    on_progress("baseline_eval", 0.14)
    baseline_holdout = backend.evaluate(session, holdout_items, r.batch_size)
    baseline = backend.evaluate(session, test_items, r.batch_size)
    steps = backend.train(session, train_items, r, on_progress)
    on_progress("calibrating", 0.86)
    temperature, calibration = backend.calibrate(session, holdout_items, r.batch_size)
    final = backend.evaluate(session, test_items, r.batch_size, temperature)

    on_progress("saving", 0.92)
    _stage_save(
        out_dirs["adapter_dir"],
        lambda directory: backend.save_adapter(session, directory, temperature, spec),
        lambda directory: any((directory / name).is_file() for name in _ADAPTER_WEIGHTS),
    )

    merged_ok, merged_error = False, None
    if "merged" in spec.artifact_urls:
        try:
            _stage_save(
                out_dirs["merged_dir"],
                lambda directory: backend.save_merged(session, directory, temperature, spec),
                lambda directory: any(directory.glob("model*.safetensors")),
            )
            merged_ok = True
        except Exception as exc:  # noqa: BLE001 - adapter remains the paid deliverable
            merged_error = str(exc)[:300]

    device, dtype = backend.describe(session)
    return {
        "seed": r.seed,
        "steps": steps,
        "baseline": baseline,
        "baseline_holdout": baseline_holdout,
        "final": final,
        "calibration": calibration,
        "calibration_temperature": temperature,
        "engine": "clef-peft",
        "trained_repo": repo,
        "trained_revision": revision,
        "device": device,
        "dtype": dtype,
        "merged_ok": merged_ok,
        "merged_error": merged_error,
        "decision_recipe": {
            "label_smoothing": 0.05,
            "brier_weight": 1.0,
            "head_learning_rate": 1e-4,
        },
    }
