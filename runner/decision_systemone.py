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
        model, processor = load_release_model(root, device=device, dtype=dtype)
    return model, processor, float(runtime.get("temperature", 1.0))


def systemone(model: Any, processor: Any, request: dict, temperature: float = 1.0,
              max_length: int = 16384) -> dict:
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
    if not isinstance(temperature, (int, float)) or temperature <= 0:
        raise ValueError("temperature must be positive")

    encoded = encode_record(
        processor.tokenizer, request, max_length=max_length, processor=processor)
    device = next(model.parameters()).device
    logits = model(collate_records(
        [encoded], processor.tokenizer.pad_token_id, device))[0]
    answers = {}
    for question, values in zip(encoded.questions, logits):
        probabilities = (values.float() / float(temperature)).softmax(-1).tolist()
        answers[question.question_id] = systemone_answer(
            questions[question.question_id], dict(zip(question.option_ids, probabilities)))
    return {
        "model": request["model"],
        "answers": answers,
        "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0},
    }
