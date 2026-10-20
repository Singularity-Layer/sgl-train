"""System One response adapter for SGL Clef artifacts, including calibration."""
from __future__ import annotations

from typing import Any


def load_artifact(path: str, device: str = "cuda", dtype=None):
    """Load either an SGL adapter/head folder or a merged Clef release folder."""
    import json
    from pathlib import Path

    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    try:
        from joint_schema_model import ClefModel, JointSchemaHead, load_release_model
    except ImportError:
        from ._vendor.cloudflare_clef.joint_schema_model import (
            ClefModel,
            JointSchemaHead,
            load_release_model,
        )

    root = Path(path)
    runtime = json.loads((root / "sgl_decision_config.json").read_text(encoding="utf-8"))
    dtype = dtype or torch.bfloat16
    if (root / "adapter_config.json").is_file():
        from peft import PeftModel

        processor = AutoProcessor.from_pretrained(root)
        base = AutoModelForMultimodalLM.from_pretrained(
            runtime["base_model"],
            revision=runtime["base_revision"],
            dtype=dtype,
            device_map={"": device},
        )
        backbone = PeftModel.from_pretrained(base, root).eval()
        head_config = json.loads((root / "joint_head_config.json").read_text(encoding="utf-8"))
        head = JointSchemaHead(**head_config)
        head.load_state_dict(load_file(str(root / "joint_head.safetensors")), strict=True)
        model = ClefModel(backbone, head.to(device=device, dtype=dtype)).eval()
    else:
        if str(device) == "mps":
            # Accelerate's direct MPS device-map load can crash before the first
            # tensor is materialized. Load on CPU, then transfer explicitly.
            from transformers import Qwen3_5ForConditionalGeneration

            processor = AutoProcessor.from_pretrained(root)
            backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
                root, dtype=dtype)
            backbone = backbone.to(device)
            backbone.config.use_cache = False
            head_config = json.loads(
                (root / "joint_head_config.json").read_text(encoding="utf-8"))
            head = JointSchemaHead(**head_config)
            head.load_state_dict(
                load_file(str(root / "joint_head.safetensors")), strict=True)
            model = ClefModel(
                backbone, head.to(device=device, dtype=dtype)).eval()
        else:
            model, processor = load_release_model(root, device=device, dtype=dtype)
    temperature = runtime.get("temperature_by_type", runtime.get("temperature", 1.0))
    return model, processor, temperature


def _temperature_for(question: dict, temperature: Any) -> float:
    if isinstance(temperature, dict):
        value = temperature.get(question.get("type"), 1.0)
    elif isinstance(temperature, (list, tuple)):
        order = {"choice": 0, "score": 1, "noul": 2}
        index = order.get(question.get("type"))
        value = temperature[index] if index is not None and index < len(temperature) else 1.0
    else:
        value = temperature
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError("temperature must be positive")
    return float(value)


def systemone(model: Any, processor: Any, request: dict, temperature: Any = 1.0,
              max_length: int = 16384) -> dict:
    import torch
    try:  # artifact folder
        from joint_schema_model import collate_records, encode_record, systemone_answer
    except ImportError:  # repository package
        from ._vendor.cloudflare_clef.joint_schema_model import (
            collate_records,
            encode_record,
            systemone_answer,
        )

    questions = request.get("questions")
    if not isinstance(request.get("model"), str) or "state" not in request:
        raise ValueError("model and state are required")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("at least one question is required")
    encoded = encode_record(
        processor.tokenizer, request, max_length=max_length, processor=processor)
    device = next(model.parameters()).device
    with torch.inference_mode():
        logits = model(collate_records(
            [encoded], processor.tokenizer.pad_token_id, device))[0]
    answers = {}
    for question, values in zip(encoded.questions, logits):
        question_temperature = _temperature_for(
            questions[question.question_id], temperature)
        probabilities = (values.float() / question_temperature).softmax(-1).tolist()
        answers[question.question_id] = systemone_answer(
            questions[question.question_id], dict(zip(question.option_ids, probabilities)))
    return {
        "model": request["model"],
        "answers": answers,
        "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0},
    }
