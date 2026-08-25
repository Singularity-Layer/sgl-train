"""Pure-logic tests for engine selection and the QLoRA/vanilla capability gate.
No torch, no unsloth — resolve_engine and require_engine_for_recipe do not
import either, by design (see runner/trainer.py).
"""
import pytest

from runner.errors import InfraError
from runner.trainer import require_engine_for_recipe, resolve_engine


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
