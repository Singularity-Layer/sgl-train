import pytest
from runner.spec import parse_spec, SpecError
from runner.checkpoint import canonical_sha256

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


def test_parses_a_runtime_budget_for_plain_runs():
    raw = {**BASE, "runtime": {
        "budget_seconds": 7_200,
        "finalization_reserve_seconds": 1_200,
        "estimated_p95_seconds": 5_400,
        "profile_version": "2026-10-16-p95-v1",
    }}
    s = parse_spec(raw)
    assert s.runtime is not None
    assert s.runtime.budget_seconds == 7_200
    assert s.runtime.finalization_reserve_seconds == 1_200


def test_parses_private_checkpoint_upload_slots():
    raw = {**BASE, "checkpoint_urls": {
        "a": "https://example.invalid/checkpoint-a",
        "b": "https://example.invalid/checkpoint-b",
        "manifest": "https://example.invalid/checkpoint-manifest",
    }}
    spec = parse_spec(raw)
    assert spec.checkpoint_urls["a"].endswith("checkpoint-a")


def test_rejects_partial_checkpoint_upload_slots():
    with pytest.raises(SpecError, match="checkpoint_urls"):
        parse_spec({**BASE, "checkpoint_urls": {"a": "https://example.invalid/a"}})


def test_parses_a_pinned_resume_checkpoint():
    raw = {**BASE, "checkpoint_urls": {
        "a": "https://example.invalid/a", "b": "https://example.invalid/b",
        "manifest": "https://example.invalid/manifest",
    }, "resume": {
        "source_run_id": "source-1", "archive_url": "https://example.invalid/checkpoint",
        "archive_sha256": "ab" * 32, "archive_bytes": 1234, "step": 20, "epoch": 1,
        "planned_steps": 30, "slot": "a",
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct@abc123",
        "dataset_sha256": "cd" * 32,
        "recipe_sha256": canonical_sha256(parse_spec(BASE).recipe),
    }}
    spec = parse_spec(raw)
    assert spec.resume is not None
    assert spec.resume.step == 20


def test_rejects_a_resume_checkpoint_without_integrity_metadata():
    with pytest.raises(SpecError, match="resume"):
        parse_spec({**BASE, "resume": {"source_run_id": "source-1", "archive_url": "https://example.invalid/x"}})


@pytest.mark.parametrize("extra", ["resume", "checkpoint_urls", "runtime"])
def test_rejects_autotune_reliability_modes_that_cannot_share_one_budget(extra):
    raw = {**BASE, "autotune": {
        "trials": [{"lr": 2e-4, "lora_rank": 16, "epochs": 1}],
        "budget_seconds": 3600, "trial_estimate_seconds": 600,
        "reserve_seconds": 600, "experiment_url": "https://example.invalid/experiment",
    }}
    if extra == "runtime":
        raw[extra] = {"budget_seconds": 3600, "finalization_reserve_seconds": 600,
                      "estimated_p95_seconds": 1800, "profile_version": "v1"}
    elif extra == "checkpoint_urls":
        raw[extra] = {"a": "https://example.invalid/a", "b": "https://example.invalid/b",
                      "manifest": "https://example.invalid/manifest"}
    else:
        raw["checkpoint_urls"] = {"a": "https://example.invalid/a", "b": "https://example.invalid/b",
                                  "manifest": "https://example.invalid/manifest"}
        raw[extra] = {
            "source_run_id": "source-1", "archive_url": "https://example.invalid/checkpoint",
            "archive_sha256": "ab" * 32, "archive_bytes": 10, "step": 1, "epoch": 1,
            "planned_steps": 10, "slot": "a",
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct@abc123",
            "dataset_sha256": "cd" * 32,
            "recipe_sha256": canonical_sha256(parse_spec(BASE).recipe),
        }
    with pytest.raises(SpecError, match="autotune cannot"):
        parse_spec(raw)


@pytest.mark.parametrize("runtime", [
    {"budget_seconds": 300, "finalization_reserve_seconds": 100, "estimated_p95_seconds": 1, "profile_version": "v1"},
    {"budget_seconds": 3_600, "finalization_reserve_seconds": 3_600, "estimated_p95_seconds": 1, "profile_version": "v1"},
    {"budget_seconds": 3_600, "finalization_reserve_seconds": 900, "estimated_p95_seconds": 1, "profile_version": "bad version!"},
])
def test_rejects_an_unsafe_runtime_budget(runtime):
    with pytest.raises(SpecError, match="runtime"):
        parse_spec({**BASE, "runtime": runtime})


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


# --- unsloth_repo/unsloth_revision: optional both-or-neither mirror pin -----
# See runner/trainer_unsloth.py: unsloth's from_pretrained silently remaps a
# canonical repo name to its own mirror namespace but keeps whatever revision
# it was given, which belongs to the canonical repo, not the mirror. These
# two fields let a caller pin the MIRROR's own repo+revision so trainer.py's
# unsloth_pin_ok can gate the unsloth engine on that pin actually existing.

def test_unsloth_mirror_pin_absent_by_default():
    s = parse_spec(BASE)
    assert s.unsloth_repo is None
    assert s.unsloth_revision is None


def test_unsloth_mirror_pin_parses_when_both_present():
    raw = {**BASE, "base_model": {
        **BASE["base_model"],
        "unsloth_repo": "unsloth/Qwen2.5-0.5B-Instruct",
        "unsloth_revision": "ae616882a38b36759fc46ac3fd6769498833b913",
    }}
    s = parse_spec(raw)
    assert s.unsloth_repo == "unsloth/Qwen2.5-0.5B-Instruct"
    assert s.unsloth_revision == "ae616882a38b36759fc46ac3fd6769498833b913"


def test_unsloth_mirror_pin_rejects_repo_without_revision():
    raw = {**BASE, "base_model": {
        **BASE["base_model"], "unsloth_repo": "unsloth/Qwen2.5-0.5B-Instruct"}}
    with pytest.raises(SpecError, match="must both be present or both be absent"):
        parse_spec(raw)


def test_unsloth_mirror_pin_rejects_revision_without_repo():
    raw = {**BASE, "base_model": {
        **BASE["base_model"], "unsloth_revision": "ae616882a38b36759fc46ac3fd6769498833b913"}}
    with pytest.raises(SpecError, match="must both be present or both be absent"):
        parse_spec(raw)


@pytest.mark.parametrize("empty", ["", "   "])
def test_unsloth_mirror_pin_rejects_empty_string(empty):
    """An empty-string pin is a MALFORMED pin, not an absent one.

    Reading it as "absent" would turn a backend bug into a silent engine
    downgrade (unsloth skipped, vanilla run) instead of a loud failure.
    Explicit null / a missing key are the only ways to mean "no pin".
    """
    raw = {**BASE, "base_model": {
        **BASE["base_model"], "unsloth_repo": empty, "unsloth_revision": empty}}
    with pytest.raises(SpecError, match="must be a non-empty string when present"):
        parse_spec(raw)


def test_unsloth_mirror_pin_absent_via_null_or_missing():
    """Only a missing key or an explicit null means 'no pin'."""
    for bm in ({**BASE["base_model"]},
               {**BASE["base_model"], "unsloth_repo": None, "unsloth_revision": None}):
        s = parse_spec({**BASE, "base_model": bm})
        assert s.unsloth_repo is None and s.unsloth_revision is None


# --- Strict pin validation: both fields must be non-empty strings -----

def test_unsloth_repo_with_whitespace_only_raises():
    """Whitespace-only strings must raise SpecError."""
    raw = {**BASE, "base_model": {
        **BASE["base_model"],
        "unsloth_repo": "   ",
        "unsloth_revision": "ae616882a38b36759fc46ac3fd6769498833b913",
    }}
    with pytest.raises(SpecError, match="must be a non-empty string when present"):
        parse_spec(raw)


def test_unsloth_revision_with_whitespace_only_raises():
    """Whitespace-only revision must raise SpecError."""
    raw = {**BASE, "base_model": {
        **BASE["base_model"],
        "unsloth_repo": "unsloth/Qwen2.5-0.5B-Instruct",
        "unsloth_revision": "  \t  ",
    }}
    with pytest.raises(SpecError, match="must be a non-empty string when present"):
        parse_spec(raw)


@pytest.mark.parametrize("non_string", [0, 1, 0.0, [], {}, True, False])
def test_unsloth_repo_non_string_raises(non_string):
    """Non-string types must raise SpecError."""
    raw = {**BASE, "base_model": {
        **BASE["base_model"],
        "unsloth_repo": non_string,
        "unsloth_revision": "ae616882a38b36759fc46ac3fd6769498833b913",
    }}
    with pytest.raises(SpecError, match="must be a non-empty string when present"):
        parse_spec(raw)


@pytest.mark.parametrize("non_string", [0, 1, 0.0, [], {}, True, False])
def test_unsloth_revision_non_string_raises(non_string):
    """Non-string revision must raise SpecError."""
    raw = {**BASE, "base_model": {
        **BASE["base_model"],
        "unsloth_repo": "unsloth/Qwen2.5-0.5B-Instruct",
        "unsloth_revision": non_string,
    }}
    with pytest.raises(SpecError, match="must be a non-empty string when present"):
        parse_spec(raw)


def test_unsloth_mirror_pin_strips_whitespace():
    """Valid strings with surrounding whitespace are stripped and stored."""
    raw = {**BASE, "base_model": {
        **BASE["base_model"],
        "unsloth_repo": "  unsloth/Qwen2.5-0.5B-Instruct  ",
        "unsloth_revision": "\n ae616882a38b36759fc46ac3fd6769498833b913 \t",
    }}
    s = parse_spec(raw)
    assert s.unsloth_repo == "unsloth/Qwen2.5-0.5B-Instruct"
    assert s.unsloth_revision == "ae616882a38b36759fc46ac3fd6769498833b913"


# --- unsloth_4bit_repo/unsloth_4bit_revision: the QLoRA mirror pin ----------
# Live GPU smoke (2026-08-25) proved unsloth's remap goes one level deeper
# under load_in_4bit=True: it resolves to a THIRD repo — a dedicated
# pre-quantized "-unsloth-bnb-4bit" mirror, distinct from both the canonical
# repo and the 16-bit unsloth mirror above — and again applies whatever
# revision it was given. This pair pins that mirror, independently, with the
# same strict both-or-neither validation as unsloth_repo/unsloth_revision.

def test_unsloth_4bit_mirror_pin_absent_by_default():
    s = parse_spec(BASE)
    assert s.unsloth_4bit_repo is None
    assert s.unsloth_4bit_revision is None


def test_unsloth_4bit_mirror_pin_parses_when_both_present():
    raw = {**BASE, "base_model": {
        **BASE["base_model"],
        "unsloth_4bit_repo": "unsloth/Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit",
        "unsloth_4bit_revision": "10413c288cb9629acdf60b3e0229f3ba75efe413",
    }}
    s = parse_spec(raw)
    assert s.unsloth_4bit_repo == "unsloth/Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit"
    assert s.unsloth_4bit_revision == "10413c288cb9629acdf60b3e0229f3ba75efe413"


def test_unsloth_4bit_mirror_pin_independent_of_16bit_pin():
    """Both pins can be present at once and needn't agree — they pin two
    different mirror repos entirely."""
    raw = {**BASE, "base_model": {
        **BASE["base_model"],
        "unsloth_repo": "unsloth/Qwen2.5-0.5B-Instruct",
        "unsloth_revision": "ae616882a38b36759fc46ac3fd6769498833b913",
        "unsloth_4bit_repo": "unsloth/Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit",
        "unsloth_4bit_revision": "10413c288cb9629acdf60b3e0229f3ba75efe413",
    }}
    s = parse_spec(raw)
    assert s.unsloth_repo == "unsloth/Qwen2.5-0.5B-Instruct"
    assert s.unsloth_4bit_repo == "unsloth/Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit"
    assert s.unsloth_repo != s.unsloth_4bit_repo
    assert s.unsloth_revision != s.unsloth_4bit_revision


def test_unsloth_4bit_mirror_pin_rejects_repo_without_revision():
    raw = {**BASE, "base_model": {
        **BASE["base_model"],
        "unsloth_4bit_repo": "unsloth/Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit"}}
    with pytest.raises(SpecError, match="must both be present or both be absent"):
        parse_spec(raw)


def test_unsloth_4bit_mirror_pin_rejects_revision_without_repo():
    raw = {**BASE, "base_model": {
        **BASE["base_model"],
        "unsloth_4bit_revision": "10413c288cb9629acdf60b3e0229f3ba75efe413"}}
    with pytest.raises(SpecError, match="must both be present or both be absent"):
        parse_spec(raw)


@pytest.mark.parametrize("empty", ["", "   "])
def test_unsloth_4bit_mirror_pin_rejects_empty_string(empty):
    raw = {**BASE, "base_model": {
        **BASE["base_model"], "unsloth_4bit_repo": empty, "unsloth_4bit_revision": empty}}
    with pytest.raises(SpecError, match="must be a non-empty string when present"):
        parse_spec(raw)


@pytest.mark.parametrize("non_string", [0, 1, 0.0, [], {}, True, False])
def test_unsloth_4bit_repo_non_string_raises(non_string):
    raw = {**BASE, "base_model": {
        **BASE["base_model"],
        "unsloth_4bit_repo": non_string,
        "unsloth_4bit_revision": "10413c288cb9629acdf60b3e0229f3ba75efe413",
    }}
    with pytest.raises(SpecError, match="must be a non-empty string when present"):
        parse_spec(raw)


@pytest.mark.parametrize("non_string", [0, 1, 0.0, [], {}, True, False])
def test_unsloth_4bit_revision_non_string_raises(non_string):
    raw = {**BASE, "base_model": {
        **BASE["base_model"],
        "unsloth_4bit_repo": "unsloth/Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit",
        "unsloth_4bit_revision": non_string,
    }}
    with pytest.raises(SpecError, match="must be a non-empty string when present"):
        parse_spec(raw)


def test_unsloth_4bit_mirror_pin_absent_via_null_or_missing():
    """Only a missing key or an explicit null means 'no pin'."""
    for bm in ({**BASE["base_model"]},
               {**BASE["base_model"], "unsloth_4bit_repo": None, "unsloth_4bit_revision": None}):
        s = parse_spec({**BASE, "base_model": bm})
        assert s.unsloth_4bit_repo is None and s.unsloth_4bit_revision is None


def test_unsloth_4bit_mirror_pin_strips_whitespace():
    raw = {**BASE, "base_model": {
        **BASE["base_model"],
        "unsloth_4bit_repo": "  unsloth/Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit  ",
        "unsloth_4bit_revision": "\n 10413c288cb9629acdf60b3e0229f3ba75efe413 \t",
    }}
    s = parse_spec(raw)
    assert s.unsloth_4bit_repo == "unsloth/Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit"
    assert s.unsloth_4bit_revision == "10413c288cb9629acdf60b3e0229f3ba75efe413"
