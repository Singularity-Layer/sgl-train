"""Save-stage failures are infra, not training, on BOTH engines: by the time
save runs, training already succeeded and the compute was delivered, so a
crash while writing artifacts is our-side infrastructure, never the buyer's
data or recipe. That conversion must hold for ANY exception (not just
OSError), and a failed merge must never leave a half-written merged_dir behind
for a retry to tar up as if it were a complete artifact.

_save_vanilla and trainer_unsloth.save_unsloth are both torch-free (they only
call os/shutil plus methods on the model/tokenizer objects handed to them), so
they're directly testable here with plain fakes and a real tmp_path — no torch,
no unsloth needed.
"""
from __future__ import annotations

import os

import pytest

from runner import trainer, trainer_unsloth
from runner.errors import InfraError


class _FakeTokenizer:
    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)


def _out_dirs(tmp_path):
    return {"adapter_dir": str(tmp_path / "adapter"), "merged_dir": str(tmp_path / "merged")}


class _FakeSpec:
    """Minimal spec for test usage."""
    def __init__(self):
        self.hf_repo = "Q/M"
        self.hf_revision = "sha"
        self.unsloth_repo = "unsloth/M"
        self.unsloth_revision = "unsloth-sha"
        self.unsloth_4bit_repo = "unsloth/M-unsloth-bnb-4bit"
        self.unsloth_4bit_revision = "unsloth-4bit-sha"


def _spec():
    return _FakeSpec()


# --- _save_vanilla -----------------------------------------------------------

class _AdapterSaveRaises:
    def __init__(self, exc):
        self._exc = exc

    def save_pretrained(self, path):
        raise self._exc


class _MergeRaises:
    def __init__(self, exc):
        self._exc = exc

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)

    def merge_and_unload(self):
        raise self._exc


class _MergedSaveWritesPartialThenRaises:
    def save_pretrained(self, path, safe_serialization=True):
        with open(os.path.join(path, "partial.safetensors"), "w") as fh:
            fh.write("half-written")
        raise RuntimeError("cuda oom mid-write")


class _MergeSucceedsButSaveIsPartial:
    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)

    def merge_and_unload(self):
        return _MergedSaveWritesPartialThenRaises()


@pytest.mark.parametrize("exc", [OSError("disk full"), RuntimeError("cuda oom during merge")])
def test_save_vanilla_adapter_save_failure_is_infra_error_any_exception_type(tmp_path, exc):
    with pytest.raises(InfraError, match=r"\[engine=vanilla\] artifact save failed"):
        trainer._save_vanilla(_AdapterSaveRaises(exc), _FakeTokenizer(), _out_dirs(tmp_path), _spec())


@pytest.mark.parametrize("exc", [OSError("disk full"), RuntimeError("cuda oom during merge")])
def test_save_vanilla_merge_failure_is_infra_error_any_exception_type(tmp_path, exc):
    with pytest.raises(InfraError, match=r"\[engine=vanilla\] artifact save failed"):
        trainer._save_vanilla(_MergeRaises(exc), _FakeTokenizer(), _out_dirs(tmp_path), _spec())


def test_save_vanilla_cleans_partial_merged_dir_on_failure(tmp_path):
    out_dirs = _out_dirs(tmp_path)

    with pytest.raises(InfraError):
        trainer._save_vanilla(_MergeSucceedsButSaveIsPartial(), _FakeTokenizer(), out_dirs, _spec())

    assert not os.path.exists(out_dirs["merged_dir"])


def test_save_vanilla_infra_error_is_not_double_wrapped(tmp_path):
    class _RaisesInfraDirectly:
        def save_pretrained(self, path):
            raise InfraError("[engine=vanilla] some inner infra failure")

    with pytest.raises(InfraError, match="some inner infra failure"):
        trainer._save_vanilla(_RaisesInfraDirectly(), _FakeTokenizer(), _out_dirs(tmp_path), _spec())


def test_save_vanilla_success_leaves_both_dirs_populated(tmp_path):
    class _Merged:
        def save_pretrained(self, path, safe_serialization=True):
            with open(os.path.join(path, "model.safetensors"), "w") as fh:
                fh.write("weights")

    class _Model:
        def save_pretrained(self, path):
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "adapter.bin"), "w") as fh:
                fh.write("adapter")

        def merge_and_unload(self):
            return _Merged()

    out_dirs = _out_dirs(tmp_path)
    trainer._save_vanilla(_Model(), _FakeTokenizer(), out_dirs, _spec())

    assert os.path.exists(os.path.join(out_dirs["adapter_dir"], "adapter.bin"))
    assert os.path.exists(os.path.join(out_dirs["merged_dir"], "model.safetensors"))


def test_early_save_writes_only_adapter_and_never_merges_base_model(tmp_path):
    class _Model:
        def __init__(self):
            self.merged = False

        def save_pretrained(self, path):
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "adapter.bin"), "w") as fh:
                fh.write("adapter")

        def merge_and_unload(self):
            self.merged = True
            raise AssertionError("early finalization must not merge base weights")

    model = _Model()
    out_dirs = _out_dirs(tmp_path)
    trainer._save_adapter_only(model, _FakeTokenizer(), out_dirs, _spec(), "vanilla")
    assert os.path.exists(os.path.join(out_dirs["adapter_dir"], "adapter.bin"))
    assert not os.path.exists(out_dirs["merged_dir"])
    assert model.merged is False


@pytest.mark.parametrize("step,total,accum,expected", [
    (1, 10, 4, False), (4, 10, 4, True), (8, 10, 4, True), (10, 10, 4, True),
])
def test_optimizer_boundary_matches_gradient_accumulation(step, total, accum, expected):
    assert trainer._is_optimizer_boundary(step, total, accum) is expected


# --- trainer_unsloth.save_unsloth --------------------------------------------

class _UnslothMergeRaises:
    def __init__(self, exc):
        self._exc = exc

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)

    def merge_and_unload(self):
        raise self._exc


class _Unsloth4bitMergeRaises:
    def __init__(self, exc):
        self._exc = exc

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)

    def save_pretrained_merged(self, path, tokenizer, save_method):
        raise self._exc


@pytest.mark.parametrize("exc", [OSError("disk full"), RuntimeError("cuda oom during merge")])
def test_save_unsloth_16bit_merge_failure_is_infra_error_any_exception_type(tmp_path, exc):
    with pytest.raises(InfraError, match=r"\[engine=unsloth\] artifact save failed"):
        trainer_unsloth.save_unsloth(
            _UnslothMergeRaises(exc), _FakeTokenizer(), _out_dirs(tmp_path), load_in_4bit=False, spec=_spec())


@pytest.mark.parametrize("exc", [OSError("disk full"), RuntimeError("dequant failed")])
def test_save_unsloth_4bit_merge_failure_is_infra_error_any_exception_type(tmp_path, exc):
    with pytest.raises(InfraError, match=r"\[engine=unsloth\] artifact save failed"):
        trainer_unsloth.save_unsloth(
            _Unsloth4bitMergeRaises(exc), _FakeTokenizer(), _out_dirs(tmp_path), load_in_4bit=True, spec=_spec())


def test_save_unsloth_cleans_partial_merged_dir_on_failure(tmp_path):
    out_dirs = _out_dirs(tmp_path)

    class _PartialMerged:
        def save_pretrained(self, path, safe_serialization=True):
            with open(os.path.join(path, "partial.safetensors"), "w") as fh:
                fh.write("half-written")
            raise RuntimeError("cuda oom mid-write")

    class _Model:
        def save_pretrained(self, path):
            os.makedirs(path, exist_ok=True)

        def merge_and_unload(self):
            return _PartialMerged()

    with pytest.raises(InfraError):
        trainer_unsloth.save_unsloth(_Model(), _FakeTokenizer(), out_dirs, load_in_4bit=False, spec=_spec())

    assert not os.path.exists(out_dirs["merged_dir"])


def test_save_unsloth_infra_error_is_not_double_wrapped(tmp_path):
    class _RaisesInfraDirectly:
        def save_pretrained(self, path):
            raise InfraError("[engine=unsloth] some inner infra failure")

    with pytest.raises(InfraError, match="some inner infra failure"):
        trainer_unsloth.save_unsloth(
            _RaisesInfraDirectly(), _FakeTokenizer(), _out_dirs(tmp_path), load_in_4bit=False, spec=_spec())
