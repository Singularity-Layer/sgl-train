from runner.budget import RuntimeClock
from runner.spec import RuntimeBudget


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


def budget():
    return RuntimeBudget(
        budget_seconds=3_600,
        finalization_reserve_seconds=900,
        estimated_p95_seconds=3_000,
        profile_version="v1",
    )


def test_runtime_clock_enters_finalization_reserve_before_deadline():
    now = Clock()
    runtime = RuntimeClock(budget(), clock=now)
    now.value = 2_699
    assert runtime.should_finalize() is False
    now.value = 2_700
    assert runtime.should_finalize() is True
    assert runtime.remaining_seconds() == 900


def test_eta_uses_measured_progress_and_never_returns_negative():
    now = Clock()
    runtime = RuntimeClock(budget(), clock=now)
    now.value = 600
    assert runtime.estimated_remaining_seconds(20, 100) == 2_400
    assert runtime.estimated_remaining_seconds(100, 100) == 0
    assert runtime.estimated_remaining_seconds(0, 100) == 3_000


def test_runtime_clock_is_disabled_for_legacy_specs():
    now = Clock()
    runtime = RuntimeClock(None, clock=now)
    now.value = 999_999
    assert runtime.should_finalize() is False
    assert runtime.remaining_seconds() is None
    assert runtime.estimated_remaining_seconds(1, 10) is None
