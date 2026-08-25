"""Pure-logic tests for engine selection and the QLoRA/vanilla capability gate.
No torch, no unsloth — resolve_engine, require_engine_for_recipe and
normalize_engine_mode do not import either, by design (see runner/trainer.py).
"""
import types

import pytest

from runner.errors import InfraError
from runner.trainer import (_attempt_engine, _model_dtype, normalize_engine_mode,
                             require_engine_for_recipe, resolve_engine)


@pytest.mark.parametrize("probe", [True, False])
def test_vanilla_env_always_picks_vanilla_regardless_of_probe(probe):
    assert resolve_engine("vanilla", probe) == "vanilla"


@pytest.mark.parametrize("probe", [True, False])
def test_unsloth_env_always_picks_unsloth_regardless_of_probe(probe):
    assert resolve_engine("unsloth", probe) == "unsloth"


def test_auto_picks_unsloth_when_probe_succeeds():
    assert resolve_engine("auto", True) == "unsloth"


def test_auto_falls_back_to_vanilla_when_probe_fails():
    assert resolve_engine("auto", False) == "vanilla"


def test_unset_env_behaves_like_auto():
    assert resolve_engine(None, True) == "unsloth"
    assert resolve_engine(None, False) == "vanilla"
    assert resolve_engine("", True) == "unsloth"


def test_unrecognized_env_value_falls_back_to_auto_behavior():
    assert resolve_engine("bogus", True) == "unsloth"
    assert resolve_engine("bogus", False) == "vanilla"


def test_env_value_is_case_and_whitespace_insensitive():
    assert resolve_engine(" VANILLA ", True) == "vanilla"
    assert resolve_engine("Unsloth", False) == "unsloth"


def test_4bit_on_unsloth_is_fine():
    require_engine_for_recipe("unsloth", True)  # must not raise


def test_4bit_on_vanilla_raises_infra_error():
    with pytest.raises(InfraError, match="4-bit training requires the unsloth engine"):
        require_engine_for_recipe("vanilla", True)


def test_no_4bit_on_vanilla_is_fine():
    require_engine_for_recipe("vanilla", False)  # must not raise


def test_no_4bit_on_unsloth_is_fine():
    require_engine_for_recipe("unsloth", False)  # must not raise


def test_4bit_on_vanilla_error_names_the_engine_that_failed():
    # main.py's failed-result payload has no other way to learn which engine
    # was running when train() raised (see trainer.py's raise sites) — the
    # engine label has to live in the message string itself.
    with pytest.raises(InfraError, match=r"\[engine=vanilla\]"):
        require_engine_for_recipe("vanilla", True)


# --- normalize_engine_mode: the single SGL_ENGINE parser -------------------
# resolve_engine and run_finetune both call this instead of each re-parsing
# the raw env string; these cases pin its exact behavior so that contract
# can't silently drift.

@pytest.mark.parametrize("raw, expected", [
    (None, "auto"),
    ("", "auto"),
    ("auto", "auto"),
    ("vanilla", "vanilla"),
    ("unsloth", "unsloth"),
    (" VANILLA ", "vanilla"),
    ("Unsloth", "unsloth"),
    ("bogus", "bogus"),
])
def test_normalize_engine_mode(raw, expected):
    assert normalize_engine_mode(raw) == expected


def test_resolve_engine_delegates_normalization_to_normalize_engine_mode():
    # Any raw value normalize_engine_mode maps to "vanilla" must make
    # resolve_engine pick vanilla regardless of the probe result — if
    # resolve_engine ever re-implemented its own parsing, this is the kind of
    # case (whitespace/case variants) where the two could quietly disagree.
    for raw in (" VANILLA ", "Vanilla", "vanilla"):
        assert normalize_engine_mode(raw) == "vanilla"
        assert resolve_engine(raw, True) == "vanilla"
        assert resolve_engine(raw, False) == "vanilla"


# --- _attempt_engine: exception messages carry the engine label ------------

def test_attempt_engine_empty_dataset_error_names_the_engine():
    r = types.SimpleNamespace(max_seq_len=128)
    with pytest.raises(RuntimeError, match=r"\[engine=unsloth\] dataset produced no usable sequences"):
        _attempt_engine(
            load_fn=lambda spec, dtype, device: (object(), object()),
            wrap_fn=lambda model, rec: model,
            spec=None, r=r, dtype=None, device="cpu",
            train_rows=[], eval_rows=[], on_progress=lambda *a, **k: None,
            engine_name="unsloth",
        )


def test_attempt_engine_empty_dataset_error_names_vanilla_too():
    r = types.SimpleNamespace(max_seq_len=128)
    with pytest.raises(RuntimeError, match=r"\[engine=vanilla\] dataset produced no usable sequences"):
        _attempt_engine(
            load_fn=lambda spec, dtype, device: (object(), object()),
            wrap_fn=lambda model, rec: model,
            spec=None, r=r, dtype=None, device="cpu",
            train_rows=[], eval_rows=[], on_progress=lambda *a, **k: None,
            engine_name="vanilla",
        )


# --- _model_dtype: report the model's REAL dtype, not the vanilla heuristic ---
# The unsloth loader is handed dtype=None and auto-picks its own, so the
# heuristic `dtype` computed in run_finetune before either engine ran can be
# flat wrong for that engine. _model_dtype reads the truth straight off a real
# parameter instead, for both engines.

def test_model_dtype_reads_off_first_parameter():
    class _Param:
        def __init__(self, dtype):
            self.dtype = dtype

    class _FakeModel:
        def __init__(self, dtypes):
            self._dtypes = dtypes

        def parameters(self):
            return iter(_Param(d) for d in self._dtypes)

    model = _FakeModel(["torch.bfloat16", "torch.float32"])
    assert _model_dtype(model) == "torch.bfloat16"


def test_model_dtype_uses_str_of_the_dtype_object_not_just_first_field():
    # str(dtype) is what gets reported, not repr or some other stringification
    # — pin that _model_dtype doesn't do anything fancier than str().
    class _WeirdDtype:
        def __str__(self):
            return "torch.float16"

    class _Param:
        dtype = _WeirdDtype()

    class _FakeModel:
        def parameters(self):
            return iter([_Param()])

    assert _model_dtype(_FakeModel()) == "torch.float16"
