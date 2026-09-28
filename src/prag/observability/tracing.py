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
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from prag.core.models.state import RequestState

__all__ = ["SpanAttr", "SpanRecord", "TraceRecorder", "request_spans"]


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

    # Root span, §16.1. The user is hashed: the trace skeleton outlives query content by months.
    USER_ID_HASH: Final = "prag.request.user_id_hash"
    OUTCOME: Final = "prag.request.outcome"
    ROUTE_CLASS: Final = "prag.request.route_class"
    TOTAL_LATENCY_MS: Final = "prag.request.total_latency_ms"
    CONFIDENCE: Final = "prag.answer.confidence_score"

    INTENT: Final = "prag.intelligence.intent"
    DOMAIN: Final = "prag.intelligence.domain"
    COMPLEXITY: Final = "prag.intelligence.complexity"
    ELIMINATED: Final = "prag.routing.eliminated"
    HEDGED: Final = "prag.routing.hedged"

    LEGS: Final = "prag.retrieval.legs"
    SOURCE_ID: Final = "prag.retrieval.leg.source_id"
    LEG_LATENCY_MS: Final = "prag.retrieval.leg.latency_ms"
    POOL_SIZE: Final = "prag.evidence.pool_size"
    ORDERING_MODE: Final = "prag.context.ordering_mode"

    GUARDRAIL_HITS: Final = "prag.guardrail.hits"
    GUARDRAIL_BLOCKED: Final = "prag.guardrail.blocked"
    INJECTION_HITS: Final = "prag.guardrail.injection_hits"
    ACL_DROPS: Final = "prag.guardrail.acl_drops"
    QUARANTINE_DROPS: Final = "prag.guardrail.quarantine_drops"

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

    def __init__(
        self, *, otel_tracer: Any = None, redact: bool = True, capacity: int = 10_000
    ) -> None:
        # Bounded: the in-memory copy is for tests and a local debug view, and an unbounded one
        # is a slow memory leak in a server that runs for weeks.
        self._spans: deque[SpanRecord] = deque(maxlen=capacity)
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

    def record_trace(self, spans: Sequence[SpanRecord]) -> None:
        """Record one request's trace: the first span is the root, the rest its children.

        Exported as one trace with real start and end times, so a trace viewer shows the node
        waterfall rather than a pile of unrelated zero-length spans.
        """
        cleaned = [
            SpanRecord(
                name=s.name,
                attributes=self._clean(s.attributes),
                started_at_ms=s.started_at_ms,
                elapsed_ms=s.elapsed_ms,
                error=s.error,
            )
            for s in spans
        ]
        self._spans.extend(cleaned)
        if self._otel is not None and cleaned:
            _export_trace(self._otel, cleaned)

    def _export(self, record: SpanRecord) -> None:
        if self._otel is not None:
            _export_trace(self._otel, [record])

    @property
    def spans(self) -> tuple[SpanRecord, ...]:
        return tuple(self._spans)

    def named(self, name: str) -> tuple[SpanRecord, ...]:
        return tuple(s for s in self._spans if s.name == name)

    def clear(self) -> None:
        self._spans.clear()


def _export_trace(tracer: Any, spans: Sequence[SpanRecord]) -> None:
    """Emit spans through an OpenTelemetry tracer, children parented on the first span.

    Imported lazily: OpenTelemetry is the optional ``otel`` extra, and a platform without it
    records in memory and exports nothing.
    """
    from opentelemetry import trace

    def ns(ms: int) -> int:
        return ms * 1_000_000

    def start(record: SpanRecord, context: Any = None) -> Any:
        span = tracer.start_span(
            record.name,
            context=context,
            start_time=ns(record.started_at_ms),
            attributes={k: v for k, v in record.attributes.items() if v is not None},
        )
        if record.error:
            span.set_status(trace.Status(trace.StatusCode.ERROR, record.error))
        return span

    root, *children = spans
    root_span = start(root)
    context = trace.set_span_in_context(root_span)
    for child in children:
        start(child, context).end(end_time=ns(child.started_at_ms + child.elapsed_ms))
    root_span.end(end_time=ns(root.started_at_ms + root.elapsed_ms))


#: Graph nodes to their §16.1 span names. A node absent here gets ``prag.node.<id>``.
_NODE_SPANS: Final = {
    "analyze": "prag.query.understand",
    "retrieve": "prag.retrieval.plan",
    "build_context": "prag.context.build",
    "generate": "prag.generation",
    "abstain": "prag.abstain",
}


def request_spans(
    state: RequestState, *, outcome: str, started_at_ms: int, elapsed_ms: int
) -> list[SpanRecord]:
    """The trace for one finished request, root first, built from its final state.

    Built after the fact rather than instrumented inline, so the graph engine and the nodes stay
    free of telemetry calls and a request that ended by exception still gets a complete trace.
    Node spans are laid out sequentially from their recorded timings; sub-spans that have no
    timing of their own (the strategy decision, each retrieval leg) sit inside their node.
    """
    from prag.core.ids import short_hash

    result = state.result
    root = SpanRecord(
        name="prag.request",
        started_at_ms=started_at_ms,
        elapsed_ms=elapsed_ms,
        attributes={
            SpanAttr.REQUEST_ID: state.request_id,
            SpanAttr.TRACE_ID: state.trace_id,
            SpanAttr.TENANT_ID: state.principal.tenant_id,
            SpanAttr.USER_ID_HASH: short_hash(state.principal.user_id),
            SpanAttr.SLA_TIER: str(state.principal.sla_tier),
            SpanAttr.OUTCOME: outcome,
            SpanAttr.ABSTAINED: outcome != "answered",
            SpanAttr.TOTAL_LATENCY_MS: elapsed_ms,
            SpanAttr.BUDGET_DEGRADATION_LEVEL: state.budget.degradation_level,
            SpanAttr.USD_COST: state.budget.usd_spent,
        },
    )
    if result is not None:
        root.attributes.update(
            {
                SpanAttr.ROUTE_CLASS: result.diagnostics.route_class,
                SpanAttr.CONFIDENCE: result.confidence.score,
            }
        )
        if result.diagnostics.ttft_ms is not None:
            root.attributes[SpanAttr.TTFT_MS] = result.diagnostics.ttft_ms

    spans = [root]
    cursor = started_at_ms

    def child(name: str, duration: int = 0, **attributes: Any) -> SpanRecord:
        span = SpanRecord(
            name=name, started_at_ms=cursor, elapsed_ms=duration, attributes=attributes
        )
        spans.append(span)
        return span

    child("prag.guardrail.input", **_guardrail_attrs(state, "input"))
    for node, ms in state.node_timings.items():
        span = child(_NODE_SPANS.get(node, f"prag.node.{node}"), ms, **{SpanAttr.NODE_ID: node})
        span.attributes.update(_node_attrs(state, node))
        if node == "analyze" and state.strategy is not None:
            child(
                "prag.route.strategy",
                **{
                    SpanAttr.STRATEGY: str(state.strategy.strategy),
                    SpanAttr.ELIMINATED: sorted(str(k) for k in state.strategy.eliminated),
                    SpanAttr.HEDGED: state.strategy.hedged,
                },
            )
        if node == "retrieve":
            for leg in state.pool.leg_results if state.pool else ():
                child(
                    "prag.retrieval.leg",
                    leg.latency_ms,
                    **{
                        SpanAttr.LEG_ID: leg.leg_id,
                        SpanAttr.SOURCE_ID: leg.source_id,
                        SpanAttr.LEG_STATUS: str(leg.status),
                        SpanAttr.LEG_CANDIDATES: len(leg.candidates),
                    },
                )
            child("prag.guardrail.retrieval", **_screen_attrs(state))
        cursor += ms
    child("prag.guardrail.output", **_guardrail_attrs(state, "output"))
    return spans


def _node_attrs(state: RequestState, node: str) -> dict[str, Any]:
    if node == "analyze" and state.analysis is not None:
        analysis = state.analysis
        return {
            SpanAttr.CLASSIFIER_TIER: analysis.classifier_tier_used,
            SpanAttr.ROUTER_UNCERTAINTY: analysis.router_uncertainty,
            SpanAttr.INTENT: str(analysis.intent.value),
            SpanAttr.DOMAIN: str(analysis.domain.value),
            SpanAttr.COMPLEXITY: str(analysis.complexity.value),
        }
    if node == "retrieve" and state.plan is not None:
        return {
            SpanAttr.PLAN_ID: state.plan.plan_id,
            SpanAttr.LEGS: len(state.plan.legs),
            SpanAttr.POOL_SIZE: len(state.pool.candidates) if state.pool else 0,
            SpanAttr.EVIDENCE_GROUPS: len(state.evidence),
            SpanAttr.EVIDENCE_INDEPENDENT: sum(1 for g in state.evidence if g.independent),
        }
    if node == "build_context" and state.bundle is not None:
        bundle = state.bundle
        return {
            SpanAttr.CONTEXT_BUNDLE_ID: bundle.bundle_id,
            SpanAttr.CONTEXT_EVIDENCE_TOKENS: sum(
                r.used_tokens for r in bundle.regions if str(r.name) == "EVIDENCE"
            ),
            SpanAttr.CONTEXT_COMPRESSION_LEVEL: bundle.compression_level,
            SpanAttr.CONTEXT_DROPPED_GROUPS: len(bundle.dropped_group_ids),
            SpanAttr.CONTEXT_COVERAGE_WARNING: bundle.coverage_warning,
            SpanAttr.ORDERING_MODE: str(bundle.ordering_mode),
        }
    if node == "generate" and state.result is not None:
        diagnostics = state.result.diagnostics
        return {
            SpanAttr.MODEL_ID: diagnostics.model_id,
            SpanAttr.MODEL_VERSION: diagnostics.model_version,
            SpanAttr.TOKENS_IN: diagnostics.usage.tokens_in if diagnostics.usage else 0,
            SpanAttr.TOKENS_OUT: diagnostics.usage.tokens_out if diagnostics.usage else 0,
            SpanAttr.GROUNDEDNESS: state.result.grounding.groundedness,
            SpanAttr.CLAIMS_TOTAL: state.result.grounding.claims_total,
            SpanAttr.CLAIMS_UNSOURCED: state.result.grounding.claims_unsourced,
        }
    return {}


def _guardrail_attrs(state: RequestState, phase: str) -> dict[str, Any]:
    verdicts = [v for v in state.guardrail_verdicts if str(v.phase) == phase]
    return {
        SpanAttr.GUARDRAIL_HITS: sorted(
            {v.reason_code for v in verdicts if v.reason_code is not None}
        ),
        SpanAttr.GUARDRAIL_BLOCKED: any(v.blocked for v in verdicts),
    }


def _screen_attrs(state: RequestState) -> dict[str, Any]:
    reasons = [
        v.reason_code
        for v in state.guardrail_verdicts
        if str(v.phase) == "retrieval" and v.blocked
    ]
    return {
        SpanAttr.INJECTION_HITS: reasons.count("document_injection"),
        SpanAttr.ACL_DROPS: reasons.count("acl_recheck_mismatch"),
        SpanAttr.QUARANTINE_DROPS: reasons.count("source_quarantined"),
    }
