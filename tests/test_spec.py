import pytest
from runner.spec import parse_spec, SpecError

BASE = {
    "run_id": "11111111-1111-1111-1111-111111111111",
    "kind": "finetune",
    "base_model": {"hf_repo": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "abc123"},
    "dataset_url": "https://example.invalid/data.jsonl",
    "quantize": "Q4_K_M",
    "progress_url": "https://example.invalid/progress",
    "result_url": "https://example.invalid/result",
    "token": "tok",
    "artifact_urls": {"adapter": "https://example.invalid/a", "gguf": "https://example.invalid/g", "report": "https://example.invalid/r"},
    "recipe": {"lora_rank": 16, "lora_alpha": 32, "lr": 2e-4, "epochs": 3,
               "max_seq_len": 1024, "batch_size": 4, "grad_accum": 4,
               "eval_split": 0.1, "seed": 1234},
}


def test_parses_a_valid_spec():
    s = parse_spec(BASE)
    assert s.run_id == BASE["run_id"]
    assert s.hf_repo == "Qwen/Qwen2.5-0.5B-Instruct"
    assert s.hf_revision == "abc123"
    assert s.recipe.seed == 1234


def test_rejects_missing_revision():
    raw = {**BASE, "base_model": {"hf_repo": "Qwen/Qwen2.5-0.5B-Instruct"}}
    with pytest.raises(SpecError, match="revision"):
        parse_spec(raw)


def test_rejects_missing_seed():
    recipe = {k: v for k, v in BASE["recipe"].items() if k != "seed"}
    with pytest.raises(SpecError, match="seed"):
        parse_spec({**BASE, "recipe": recipe})


def test_clamps_out_of_range_values():
    recipe = {**BASE["recipe"], "lora_rank": 999, "lr": 5.0, "epochs": 0, "eval_split": 0.9}
    s = parse_spec({**BASE, "recipe": recipe})
    assert s.recipe.lora_rank == 64
    assert s.recipe.lr == 1e-3
    assert s.recipe.epochs == 1
    assert s.recipe.eval_split == 0.3


def test_rejects_unknown_kind():
    with pytest.raises(SpecError, match="kind"):
        parse_spec({**BASE, "kind": "mine-bitcoin"})


def test_rejects_unknown_quantize():
    with pytest.raises(SpecError, match="quantize"):
        parse_spec({**BASE, "quantize": "Q99_ULTRA"})
