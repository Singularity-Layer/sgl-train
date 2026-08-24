"""Shared error taxonomy. Torch-free — imported from both main.py and trainer.py,
and trainer.py must stay importable on a laptop with no CUDA.
"""
from __future__ import annotations


class InfraError(RuntimeError):
    """Our-side infrastructure failure — refundable; never the customer's fault."""
