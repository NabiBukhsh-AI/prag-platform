"""Tracing, metrics, and the event bus.

Span attribute and metric names are declared constants, never string literals at call sites. A
dashboard keyed on a name breaks silently when one call site misspells it — nothing fails, the
series simply stops appearing, which is the worst way for observability to break because it looks
like the problem went away.
"""

from prag.observability.events import InMemoryEventBus
from prag.observability.metrics import METRICS, MetricDef, Metrics, record_request
from prag.observability.tracing import SpanAttr, SpanRecord, TraceRecorder, request_spans

__all__ = [
    "METRICS",
    "InMemoryEventBus",
    "MetricDef",
    "Metrics",
    "SpanAttr",
    "SpanRecord",
    "TraceRecorder",
    "record_request",
    "request_spans",
]
