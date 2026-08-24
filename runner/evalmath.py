"""Eval aggregation. Torch-free so the arithmetic is testable without a GPU.

Kept separate from trainer.py precisely because this is where a wrong number
becomes a false claim of improvement.
"""
from __future__ import annotations

import math


def aggregate(batches: list[tuple[float, int]]) -> dict:
    """Token-weighted mean cross-entropy and its perplexity.

    `batches` is [(sum_of_loss_over_tokens, n_tokens), ...]. Weighting by tokens
    rather than by batch is what makes runs with different batch sizes comparable.
    """
    total_loss = sum(b[0] for b in batches)
    total_tokens = sum(b[1] for b in batches)
    if total_tokens <= 0:
        raise ValueError("cannot evaluate: no tokens in the eval set")
    loss = total_loss / total_tokens
    return {"eval_loss": loss, "perplexity": math.exp(loss), "eval_tokens": total_tokens}


def improved(before: float, after: float, rel_tol: float = 0.01) -> bool:
    """True only when `after` beats `before` by more than rel_tol.

    The tolerance exists because run-to-run noise routinely moves loss by a
    fraction of a percent, and calling that an improvement is how an
    optimisation loop convinces itself it is working when it is not.
    """
    if before <= 0:
        return False
    return (before - after) / before > rel_tol
