"""Shared error taxonomy. Torch-free — imported from both main.py and trainer.py,
and trainer.py must stay importable on a laptop with no CUDA.
"""
from __future__ import annotations


class InfraError(RuntimeError):
    """Our-side infrastructure failure — refundable; never the customer's fault."""


class DataError(ValueError):
    """Buyer-owned input cannot be trained; content must never enter the message."""


def attach_context(exc: Exception, **ctx: object) -> Exception:
    """Attach execution context (engine, trained_repo, etc.) to an exception.

    Context is stored in exc.sgl_context as a dict, additive to any message.
    Returns the exception to allow one-liner raises: raise attach_context(...).
    """
    exc.sgl_context = dict(ctx)  # type: ignore
    return exc
