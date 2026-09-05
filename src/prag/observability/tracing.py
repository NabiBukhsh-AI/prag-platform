"""Span attributes and the event bus.

Attribute names are **constants, never string literals at call sites.** A dashboard keyed on
``prag.retrieval.leg.timed_out`` breaks silently when one call site writes
``prag.retrieval.leg.timeout``, and nothing fails — the metric simply stops appearing, which is
the worst way for observability to break because it looks like the problem went away.

OpenTelemetry is an optional dependency here. The platform must run and be testable without a
collector, so this module degrades to a no-op recorder rather than requiring one. What it never
degrades on is the *schema*: the attribute names are asserted by tests whether or not anything
is exporting them.

Redaction is a property of the recorder, not of its callers. Query text and evidence text carry
short retention windows, and a span attribute is the easiest place to leak them by accident.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["SpanAttr", "SpanRecord", "TraceRecorder"]


class SpanAttr:
    """The span attribute schema.

    A class of constants rather than an enum, because these are written into a vendor-neutral
    attribute map and read by dashboards that know them as strings.
    """

    REQUEST_ID: Final = "prag.request.id"
    TRACE_ID: Final = "prag.trace.id"
    TENANT_ID: Final = "prag.tenant.id"
    SLA_TIER: Final = "prag.request.sla_tier"

    NODE_ID: Final = "prag.node.id"
    NODE_STATUS: Final = "prag.node.status"
    NODE_ELAPSED_MS: Final = "prag.node.elapsed_ms"

    CLASSIFIER_TIER: Final = "prag.intelligence.classifier_tier"
    ROUTER_UNCERTAINTY: Final = "prag.intelligence.router_uncertainty"
    STRATEGY: Final = "prag.routing.strategy"

    PLAN_ID: Final = "prag.retrieval.plan_id"
    LEG_ID: Final = "prag.retrieval.leg.id"
    LEG_STATUS: Final = "prag.retrieval.leg.status"
    LEG_CANDIDATES: Final = "prag.retrieval.leg.candidates"
    CANDIDATES_TOTAL: Final = "prag.retrieval.candidates_total"

    EVIDENCE_GROUPS: Final = "prag.evidence.groups"
    EVIDENCE_INDEPENDENT: Final = "prag.evidence.independent_groups"
    RERANK_SKIPPED_REASON: Final = "prag.evidence.rerank_skipped_reason"

    CONTEXT_BUNDLE_ID: Final = "prag.context.bundle_id"
    CONTEXT_EVIDENCE_TOKENS: Final = "prag.context.evidence_tokens"
    CONTEXT_COMPRESSION_LEVEL: Final = "prag.context.compression_level"
    CONTEXT_COVERAGE_WARNING: Final = "prag.context.coverage_warning"
    CONTEXT_DROPPED_GROUPS: Final = "prag.context.dropped_groups"

    MODEL_ID: Final = "prag.generation.model_id"
    MODEL_VERSION: Final = "prag.generation.model_version"
    PROVIDER_ID: Final = "prag.generation.provider_id"
    TTFT_MS: Final = "prag.generation.ttft_ms"
    TOKENS_IN: Final = "prag.generation.tokens_in"
    TOKENS_OUT: Final = "prag.generation.tokens_out"

    GROUNDEDNESS: Final = "prag.grounding.groundedness"
    CLAIMS_TOTAL: Final = "prag.grounding.claims_total"
    CLAIMS_UNSOURCED: Final = "prag.grounding.claims_unsourced"

    BUDGET_DEGRADATION_LEVEL: Final = "prag.budget.degradation_level"
    BUDGET_WALL_REMAINING_MS: Final = "prag.budget.wall_ms_remaining"
    USD_COST: Final = "prag.cost.usd"

    ABSTAINED: Final = "prag.answer.abstained"
    ABSTENTION_REASON: Final = "prag.answer.abstention_reason"
    CACHE_HIT: Final = "prag.cache.hit"
    ERROR_REASON_CODE: Final = "prag.error.reason_code"

    @classmethod
    def all_names(cls) -> frozenset[str]:
        return frozenset(
            value
            for name, value in vars(cls).items()
            if not name.startswith("_") and isinstance(value, str)
        )


#: Attributes that must never carry free text from a request. Enforced by the recorder rather
#: than by convention, because "do not put the query in a span" is a rule that holds until the
#: first debugging session where putting the query in a span would have been convenient.
_FORBIDDEN_SUBSTRINGS: Final = ("query_text", "evidence_text", "answer_text", "prompt")


@dataclass(slots=True)
class SpanRecord:
    """One recorded span."""

    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    started_at_ms: int = 0
    elapsed_ms: int = 0
    error: str | None = None


class TraceRecorder:
    """Collects spans in memory, and optionally forwards them to OpenTelemetry.

    In-memory by default so that tests can assert on the span schema without a collector — and
    they should, because a span emitted with the wrong attribute name fails nothing at runtime.
    """

    def __init__(self, *, otel_tracer: Any = None, redact: bool = True) -> None:
        self._spans: list[SpanRecord] = []
        self._otel = otel_tracer
        self._redact = redact

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[SpanRecord]:
        record = SpanRecord(
            name=name,
            attributes=self._clean(attributes),
            started_at_ms=int(time.time() * 1000),
        )
        started = time.monotonic()
        try:
            yield record
        except Exception as exc:
            record.error = getattr(exc, "reason_code", type(exc).__name__)
            raise
        finally:
            record.elapsed_ms = int((time.monotonic() - started) * 1000)
            self._spans.append(record)
            self._export(record)

    def _clean(self, attributes: dict[str, Any]) -> dict[str, Any]:
        if not self._redact:
            return dict(attributes)
        return {
            key: value
            for key, value in attributes.items()
            if not any(forbidden in key for forbidden in _FORBIDDEN_SUBSTRINGS)
        }

    def _export(self, record: SpanRecord) -> None:
        if self._otel is None:
            return
        with self._otel.start_as_current_span(record.name) as span:  # pragma: no cover
            for key, value in record.attributes.items():
                span.set_attribute(key, value)

    @property
    def spans(self) -> tuple[SpanRecord, ...]:
        return tuple(self._spans)

    def named(self, name: str) -> tuple[SpanRecord, ...]:
        return tuple(s for s in self._spans if s.name == name)

    def clear(self) -> None:
        self._spans.clear()
