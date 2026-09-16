import pytest

from predictive_maintenance.streaming.reorder import EventTimeReorderBuffer, ReorderStatus


def test_reorders_events_within_lateness_window() -> None:
    buffer = EventTimeReorderBuffer[str](allowed_lateness=1)

    assert buffer.observe(1, 1, "cycle-1").released == ()
    assert buffer.observe(1, 3, "cycle-3").released == ("cycle-1",)
    assert buffer.observe(1, 2, "cycle-2").released == ("cycle-2",)
    assert buffer.flush(1) == ("cycle-3",)


def test_late_event_is_not_released() -> None:
    buffer = EventTimeReorderBuffer[str](allowed_lateness=1)
    buffer.observe(1, 1, "cycle-1")
    buffer.observe(1, 3, "cycle-3")

    result = buffer.observe(1, 1, "late-cycle-1")

    assert result.status == ReorderStatus.LATE
    assert result.released == ()


def test_duplicate_pending_cycle_is_safe() -> None:
    buffer = EventTimeReorderBuffer[str](allowed_lateness=2)
    buffer.observe(1, 5, "first")

    result = buffer.observe(1, 5, "retry")

    assert result.status == ReorderStatus.DUPLICATE


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="allowed_lateness"):
        EventTimeReorderBuffer[str](allowed_lateness=-1)
