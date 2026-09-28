"""The event bus.

Requests accumulate events on their state and publish them once, when they have finished — so a
request that fails midway publishes what it actually did rather than half a story, and nothing on
the request path waits for a consumer.

In-memory and bounded. Everything on the bus is a side effect — telemetry, evaluation samples,
security events for the audit store — so a full buffer drops the oldest rather than blocking a
request. Redis Streams replaces this behind the same ``publish``/``subscribe`` surface when events
have to leave the process.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from prag.core.models.events import DomainEvent

__all__ = ["InMemoryEventBus"]


class InMemoryEventBus:
    def __init__(self, *, capacity: int = 10_000) -> None:
        self._buffer: deque[DomainEvent] = deque(maxlen=capacity)
        self._subscribers: list[Callable[[DomainEvent], None]] = []
        self.handler_errors = 0

    def subscribe(self, handler: Callable[[DomainEvent], None]) -> None:
        self._subscribers.append(handler)

    def publish(self, events: Iterable[DomainEvent]) -> None:
        for event in events:
            self._buffer.append(event)
            for handler in self._subscribers:
                try:
                    handler(event)
                except Exception:
                    # A broken consumer must not fail the request that published to it. The
                    # count is the signal; the event itself stays in the buffer for replay.
                    self.handler_errors += 1

    def drain(self) -> list[DomainEvent]:
        """Take everything buffered, oldest first."""
        events = list(self._buffer)
        self._buffer.clear()
        return events

    def __len__(self) -> int:
        return len(self._buffer)
