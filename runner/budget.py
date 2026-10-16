"""Monotonic runtime-budget decisions shared by training engines."""
from __future__ import annotations

import time
from collections.abc import Callable

from .spec import RuntimeBudget


class RuntimeClock:
    def __init__(self, budget: RuntimeBudget | None, clock: Callable[[], float] = time.monotonic):
        self.budget = budget
        self.clock = clock
        self.started = clock()

    def elapsed_seconds(self) -> float:
        return max(0.0, self.clock() - self.started)

    def remaining_seconds(self) -> int | None:
        if self.budget is None:
            return None
        return max(0, round(self.budget.budget_seconds - self.elapsed_seconds()))

    def should_finalize(self) -> bool:
        remaining = self.remaining_seconds()
        return remaining is not None and remaining <= self.budget.finalization_reserve_seconds

    def estimated_remaining_seconds(self, completed: int, total: int) -> int | None:
        if self.budget is None:
            return None
        if total <= 0:
            return self.budget.estimated_p95_seconds
        if completed >= total:
            return 0
        if completed <= 0:
            return self.budget.estimated_p95_seconds
        elapsed = self.elapsed_seconds()
        return max(0, round((elapsed / completed) * (total - completed)))
