import math
import pytest
from runner.evalmath import aggregate, improved


def test_token_weighted_not_batch_averaged():
    # batch A: 10 tokens at loss 1.0; batch B: 90 tokens at loss 2.0
    out = aggregate([(10.0, 10), (180.0, 90)])
    assert out["eval_loss"] == pytest.approx(1.9)   # not 1.5
    assert out["eval_tokens"] == 100


def test_perplexity_is_exp_of_loss():
    out = aggregate([(2.0, 1)])
    assert out["perplexity"] == pytest.approx(math.exp(2.0))


def test_zero_tokens_raises():
    with pytest.raises(ValueError, match="no tokens"):
        aggregate([(0.0, 0)])


def test_empty_raises():
    with pytest.raises(ValueError, match="no tokens"):
        aggregate([])


def test_improved_requires_beating_tolerance():
    assert improved(2.00, 1.90) is True     # 5% better
    assert improved(2.00, 1.995) is False   # 0.25% — inside the noise band
    assert improved(2.00, 2.10) is False
