"""Per-engine event-time reordering with a bounded lateness window."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Generic, TypeVar

T = TypeVar("T")


class ReorderStatus(StrEnum):
    """Outcome of observing one event."""

    BUFFERED = "buffered"
    LATE = "late"
    DUPLICATE = "duplicate"


@dataclass(frozen=True)
class ReorderResult(Generic[T]):
    """Events released after observing one event."""

    status: ReorderStatus
    released: tuple[T, ...] = ()


@dataclass
class _EngineBuffer(Generic[T]):
    max_cycle: int | None = None
    pending: dict[int, T] = field(default_factory=dict)


class EventTimeReorderBuffer(Generic[T]):
    """Buffer events independently per engine and release them by cycle order."""

    def __init__(self, allowed_lateness: int = 2) -> None:
        if allowed_lateness < 0:
            raise ValueError("allowed_lateness must be non-negative")
        self._allowed_lateness = allowed_lateness
        self._engines: dict[int, _EngineBuffer[T]] = {}

    def observe(self, engine_id: int, cycle: int, event: T) -> ReorderResult[T]:
        """Observe an event and release cycles behind the event-time watermark."""

        if engine_id <= 0 or cycle <= 0:
            raise ValueError("engine_id and cycle must be positive")
        state = self._engines.setdefault(engine_id, _EngineBuffer())
        if cycle in state.pending:
            return ReorderResult(status=ReorderStatus.DUPLICATE)

        if state.max_cycle is not None:
            watermark = state.max_cycle - self._allowed_lateness
            if cycle < watermark:
                return ReorderResult(status=ReorderStatus.LATE)

        state.pending[cycle] = event
        state.max_cycle = max(state.max_cycle or cycle, cycle)
        watermark = state.max_cycle - self._allowed_lateness
        released = self._release_through(state, watermark)
        return ReorderResult(status=ReorderStatus.BUFFERED, released=tuple(released))

    def flush(self, engine_id: int) -> tuple[T, ...]:
        """Release all buffered events for an engine in cycle order."""

        state = self._engines.get(engine_id)
        if state is None:
            return ()
        released = tuple(state.pending[cycle] for cycle in sorted(state.pending))
        state.pending.clear()
        return released

    @staticmethod
    def _release_through(state: _EngineBuffer[T], watermark: int) -> list[T]:
        cycles = sorted(cycle for cycle in state.pending if cycle <= watermark)
        return [state.pending.pop(cycle) for cycle in cycles]
