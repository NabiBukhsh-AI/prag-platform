"""Metrics, in Prometheus exposition format.

Every metric is declared here, with its label set, before anything records it. Recording an
undeclared metric or the wrong labels raises: a misspelled metric name fails nothing at runtime —
the series just stops appearing, and a dashboard or alert keyed on it goes quiet, which reads as
the problem having gone away. The alert rules and dashboard in ``deploy/observability`` are
tested against these declarations for the same reason.

In-process and dependency-free. Prometheus scrapes ``/metrics``; nothing here needs a client
library to produce text a scraper can read.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from prag.core.models.events import EventKind

if TYPE_CHECKING:
    from prag.core.models.state import RequestState

__all__ = ["METRICS", "MetricDef", "Metrics", "record_request"]

#: Milliseconds. Spans the interactive tier's 650 ms TTFT target and the batch tier's minutes.
_MS_BUCKETS: tuple[float, ...] = (25, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 30_000)


@dataclass(frozen=True, slots=True)
class MetricDef:
    name: str
    kind: Literal["counter", "histogram"]
    help: str
    labels: tuple[str, ...] = ()


METRICS: dict[str, MetricDef] = {
    m.name: m
    for m in (
        MetricDef(
            "prag_requests_total",
            "counter",
            "Requests by tenant and outcome.",
            ("tenant", "outcome"),
        ),
        MetricDef(
            "prag_request_duration_ms",
            "histogram",
            "End-to-end request latency.",
            ("outcome",),
        ),
        MetricDef("prag_node_duration_ms", "histogram", "Latency per graph node.", ("node",)),
        MetricDef(
            "prag_guardrail_verdicts_total",
            "counter",
            "Guardrail verdicts other than a quiet allow.",
            ("phase", "guardrail", "action", "reason"),
        ),
        MetricDef(
            "prag_security_events_total",
            "counter",
            "Security events and isolation alerts published.",
            ("kind",),
        ),
        MetricDef("prag_cost_usd_total", "counter", "Spend attributed per tenant.", ("tenant",)),
        MetricDef(
            "prag_degradation_total",
            "counter",
            "Requests by the degradation level they finished at.",
            ("level",),
        ),
    )
}

Labels = tuple[tuple[str, str], ...]


class Metrics:
    def __init__(self) -> None:
        self._counters: dict[tuple[str, Labels], float] = defaultdict(float)
        self._histograms: dict[tuple[str, Labels], list[float]] = {}

    def _key(self, name: str, kind: str, labels: dict[str, str]) -> tuple[str, Labels]:
        definition = METRICS.get(name)
        if definition is None or definition.kind != kind:
            raise KeyError(f"undeclared {kind} {name!r}")
        if set(labels) != set(definition.labels):
            raise KeyError(f"{name} takes labels {definition.labels}, got {tuple(labels)}")
        return name, tuple(sorted(labels.items()))

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        self._counters[self._key(name, "counter", labels)] += value

    def observe(self, name: str, value: float, **labels: str) -> None:
        key = self._key(name, "histogram", labels)
        # One slot per bucket, then sum, then count.
        slots = self._histograms.setdefault(key, [0.0] * (len(_MS_BUCKETS) + 2))
        for i, bound in enumerate(_MS_BUCKETS):
            if value <= bound:
                slots[i] += 1
        slots[-2] += value
        slots[-1] += 1

    def value(self, name: str, **labels: str) -> float:
        """A counter's current value, or a histogram's count. For tests and health checks."""
        key = (name, tuple(sorted(labels.items())))
        if key in self._histograms:
            return self._histograms[key][-1]
        return self._counters.get(key, 0.0)

    def render(self) -> str:
        """The Prometheus text exposition format, version 0.0.4."""
        lines: list[str] = []
        for definition in METRICS.values():
            lines.append(f"# HELP {definition.name} {definition.help}")
            lines.append(f"# TYPE {definition.name} {definition.kind}")
            if definition.kind == "counter":
                for (name, labels), value in sorted(self._counters.items()):
                    if name == definition.name:
                        lines.append(f"{name}{_fmt(labels)} {value:g}")
                continue
            for (name, labels), slots in sorted(self._histograms.items()):
                if name != definition.name:
                    continue
                for bound, count in zip(_MS_BUCKETS, slots, strict=False):
                    lines.append(f"{name}_bucket{_fmt(labels, le=f'{bound:g}')} {count:g}")
                lines.append(f"{name}_bucket{_fmt(labels, le='+Inf')} {slots[-1]:g}")
                lines.append(f"{name}_sum{_fmt(labels)} {slots[-2]:g}")
                lines.append(f"{name}_count{_fmt(labels)} {slots[-1]:g}")
        return "\n".join(lines) + "\n"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _fmt(labels: Labels, **extra: str) -> str:
    pairs = [*labels, *extra.items()]
    if not pairs:
        return ""
    return "{" + ",".join(f'{k}="{_escape(v)}"' for k, v in pairs) + "}"


_SECURITY_KINDS = frozenset({EventKind.SECURITY_EVENT, EventKind.ISOLATION_ALERT})


def record_request(
    metrics: Metrics, state: RequestState, *, outcome: str, elapsed_ms: int
) -> None:
    """Record everything a finished request contributes, from its final state.

    Derived from the state rather than recorded along the way, so a request that ends by
    exception is counted exactly like one that completed — the engine hands its state back on
    the exception for this reason.
    """
    tenant = state.principal.tenant_id
    metrics.inc("prag_requests_total", tenant=tenant, outcome=outcome)
    metrics.observe("prag_request_duration_ms", elapsed_ms, outcome=outcome)
    for node, ms in state.node_timings.items():
        metrics.observe("prag_node_duration_ms", ms, node=node)
    for verdict in state.guardrail_verdicts:
        if verdict.action == "allow" and verdict.reason_code is None:
            continue
        metrics.inc(
            "prag_guardrail_verdicts_total",
            phase=str(verdict.phase),
            guardrail=verdict.guardrail,
            action=str(verdict.action),
            reason=verdict.reason_code or "none",
        )
    for event in state.events:
        if event.kind in _SECURITY_KINDS:
            metrics.inc("prag_security_events_total", kind=str(event.kind))
    if state.budget.usd_spent:
        metrics.inc("prag_cost_usd_total", state.budget.usd_spent, tenant=tenant)
    metrics.inc("prag_degradation_total", level=str(state.budget.degradation_level))
