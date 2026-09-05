"""Tracing, metrics, logging, and the event bus.

Span attribute names are constants, never string literals at call sites. A dashboard keyed on an
attribute breaks silently when one call site misspells it — nothing fails, the metric simply
stops appearing, which is the worst way for observability to break because it looks like the
problem went away.
"""

from prag.observability.tracing import SpanAttr, SpanRecord, TraceRecorder

__all__ = ["SpanAttr", "SpanRecord", "TraceRecorder"]
