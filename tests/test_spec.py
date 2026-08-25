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


def test_rejects_a_path_traversal_run_id():
    with pytest.raises(SpecError, match="run_id"):
        parse_spec({**BASE, "run_id": "../etc"})


def test_accepts_a_uuid_style_run_id():
    s = parse_spec({**BASE, "run_id": "11111111-1111-1111-1111-111111111111"})
    assert s.run_id == "11111111-1111-1111-1111-111111111111"


def test_rejects_a_file_scheme_dataset_url():
    with pytest.raises(SpecError, match="https"):
        parse_spec({**BASE, "dataset_url": "file:///etc/passwd"})


def test_accepts_http_to_localhost_for_local_testing(monkeypatch):
    monkeypatch.setenv("SGL_ALLOW_LOCAL_URLS", "1")
    s = parse_spec({**BASE, "dataset_url": "http://localhost:8080/x"})
    assert s.dataset_url == "http://localhost:8080/x"


def test_rejects_plain_http_to_a_real_host():
    with pytest.raises(SpecError, match="https"):
        parse_spec({**BASE, "dataset_url": "http://evil.example/x"})


def test_rejects_https_to_loopback_ip():
    with pytest.raises(SpecError, match="private or local"):
        parse_spec({**BASE, "dataset_url": "https://127.0.0.1/x"})


def test_rejects_https_to_cloud_metadata_ip():
    with pytest.raises(SpecError, match="private or local"):
        parse_spec({**BASE, "dataset_url": "https://169.254.169.254/x"})


def test_rejects_https_with_empty_host():
    with pytest.raises(SpecError, match="no host"):
        parse_spec({**BASE, "dataset_url": "https:///x"})


def test_rejects_https_to_localhost():
    with pytest.raises(SpecError, match="private or local"):
        parse_spec({**BASE, "dataset_url": "https://localhost/x"})


def test_rejects_https_to_ipv6_loopback():
    with pytest.raises(SpecError, match="private or local"):
        parse_spec({**BASE, "dataset_url": "https://[::1]/x"})


def test_accepts_https_to_loopback_ip_with_local_urls_allowed(monkeypatch):
    monkeypatch.setenv("SGL_ALLOW_LOCAL_URLS", "1")
    s = parse_spec({**BASE, "dataset_url": "https://127.0.0.1/x"})
    assert s.dataset_url == "https://127.0.0.1/x"


def test_load_in_4bit_defaults_to_false_when_absent():
    s = parse_spec(BASE)
    assert s.recipe.load_in_4bit is False


def test_load_in_4bit_accepts_explicit_true():
    s = parse_spec({**BASE, "recipe": {**BASE["recipe"], "load_in_4bit": True}})
    assert s.recipe.load_in_4bit is True


def test_load_in_4bit_accepts_explicit_false():
    s = parse_spec({**BASE, "recipe": {**BASE["recipe"], "load_in_4bit": False}})
    assert s.recipe.load_in_4bit is False


@pytest.mark.parametrize("garbage", ["true", "false", 1, 0, "1", None, [], {}])
def test_load_in_4bit_rejects_non_boolean_values(garbage):
    with pytest.raises(SpecError, match="load_in_4bit"):
        parse_spec({**BASE, "recipe": {**BASE["recipe"], "load_in_4bit": garbage}})
