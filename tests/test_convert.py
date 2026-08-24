import pytest
from runner.convert import convert_commands, to_gguf, ConvertError, CONVERT_PY


def test_builds_a_two_step_pipeline():
    cmds = convert_commands("/w/merged", "/w/model.gguf", "Q4_K_M")
    assert len(cmds) == 2
    assert cmds[0][0] == CONVERT_PY
    assert "convert_hf_to_gguf.py" in cmds[0][1]
    assert "/w/merged" in cmds[0]
    assert cmds[1][-1] == "Q4_K_M"
    assert cmds[1][0].endswith("llama-quantize")


def test_f16_intermediate_is_removed_from_the_final_path():
    cmds = convert_commands("/w/merged", "/w/model.gguf", "Q4_K_M")
    f16 = cmds[0][cmds[0].index("--outfile") + 1]
    assert f16 != "/w/model.gguf"
    assert cmds[1][1] == f16
    assert cmds[1][2] == "/w/model.gguf"


def test_to_gguf_raises_on_failure():
    def fake_run(cmd):
        return 1 if "quantize" in cmd[0] else 0

    with pytest.raises(ConvertError, match="quantize"):
        to_gguf("/w/merged", "/w/model.gguf", "Q4_K_M", run=fake_run)


def test_to_gguf_succeeds_when_both_steps_exit_zero():
    to_gguf("/w/merged", "/w/model.gguf", "Q4_K_M", run=lambda cmd: 0)


def test_to_gguf_includes_stderr_tail_when_run_returns_a_tuple():
    def fake_run(cmd):
        if "quantize" in cmd[0]:
            return 1, "disk is full\n"
        return 0, ""

    with pytest.raises(ConvertError, match="disk is full"):
        to_gguf("/w/merged", "/w/model.gguf", "Q4_K_M", run=fake_run)
