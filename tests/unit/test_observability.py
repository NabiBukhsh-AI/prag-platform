"""The event bus, the metrics registry, and the per-request trace."""

from __future__ import annotations

from itertools import pairwise

import pytest

from prag.core.models.events import DomainEvent, EventKind
from prag.observability import METRICS, InMemoryEventBus, Metrics, SpanAttr, request_spans
from tests.e2e.test_standard_answer import a_state, build_engine


def an_event(n: int = 0) -> DomainEvent:
    return DomainEvent(
        event_id=f"evt_{n}",
        kind=EventKind.SECURITY_EVENT,
        request_id="req_1",
        tenant_id="t",
        occurred_at_ms=0,
    )


class TestEventBus:
    def test_subscribers_see_every_event(self) -> None:
        bus, seen = InMemoryEventBus(), []
        bus.subscribe(seen.append)
        bus.publish([an_event(1), an_event(2)])
        assert [e.event_id for e in seen] == ["evt_1", "evt_2"]

    def test_a_broken_consumer_does_not_fail_the_publisher(self) -> None:
        bus = InMemoryEventBus()
        bus.subscribe(lambda e: 1 / 0)
        bus.publish([an_event()])

        assert bus.handler_errors == 1
        assert len(bus) == 1, "the event stays buffered for replay"

    def test_a_full_buffer_drops_the_oldest(self) -> None:
        bus = InMemoryEventBus(capacity=2)
        bus.publish([an_event(1), an_event(2), an_event(3)])
        assert [e.event_id for e in bus.drain()] == ["evt_2", "evt_3"]
        assert len(bus) == 0


class TestMetrics:
    def test_an_undeclared_metric_raises(self) -> None:
        """A misspelled metric fails here instead of silently vanishing from a dashboard."""
        with pytest.raises(KeyError):
            Metrics().inc("prag_request_total", tenant="t", outcome="answered")

    def test_the_wrong_labels_raise(self) -> None:
        with pytest.raises(KeyError):
            Metrics().inc("prag_requests_total", tenant="t")

    def test_a_counter_renders(self) -> None:
        metrics = Metrics()
        metrics.inc("prag_requests_total", tenant="t", outcome="answered")
        metrics.inc("prag_requests_total", tenant="t", outcome="answered")

        assert 'prag_requests_total{outcome="answered",tenant="t"} 2' in metrics.render()

    def test_histogram_buckets_are_cumulative(self) -> None:
        metrics = Metrics()
        for ms in (10, 60, 60, 20_000):
            metrics.observe("prag_request_duration_ms", ms, outcome="answered")
        text = metrics.render()

        assert 'prag_request_duration_ms_bucket{outcome="answered",le="25"} 1' in text
        assert 'prag_request_duration_ms_bucket{outcome="answered",le="100"} 3' in text
        assert 'prag_request_duration_ms_bucket{outcome="answered",le="+Inf"} 4' in text
        assert 'prag_request_duration_ms_count{outcome="answered"} 4' in text
        assert 'prag_request_duration_ms_sum{outcome="answered"} 20130' in text

    def test_label_values_are_escaped(self) -> None:
        metrics = Metrics()
        metrics.inc("prag_cost_usd_total", 1.0, tenant='a"b\\c')
        assert 'tenant="a\\"b\\\\c"' in metrics.render()

    def test_every_declared_metric_has_a_type_line(self) -> None:
        text = Metrics().render()
        for name, definition in METRICS.items():
            assert f"# TYPE {name} {definition.kind}" in text


class TestRequestTrace:
    async def test_the_trace_follows_the_node_order(self) -> None:
        engine = await build_engine()
        run = await engine.run(a_state("how long are incident records retained"))
        spans = request_spans(run.state, outcome="answered", started_at_ms=1_000, elapsed_ms=50)

        names = [s.name for s in spans]
        assert names[0] == "prag.request"
        assert names.index("prag.query.understand") < names.index("prag.retrieval.plan")
        assert names.index("prag.retrieval.plan") < names.index("prag.generation")

    async def test_node_spans_are_laid_out_sequentially(self) -> None:
        engine = await build_engine()
        run = await engine.run(a_state("how long are incident records retained"))
        spans = request_spans(run.state, outcome="answered", started_at_ms=1_000, elapsed_ms=50)

        node_spans = [s for s in spans if SpanAttr.NODE_ID in s.attributes]
        for earlier, later in pairwise(node_spans):
            assert later.started_at_ms == earlier.started_at_ms + earlier.elapsed_ms

    async def test_export_is_one_trace_with_children_parented_on_the_root(self) -> None:
        pytest.importorskip("opentelemetry.sdk")
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from prag.observability import TraceRecorder

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        recorder = TraceRecorder(otel_tracer=provider.get_tracer("test"))

        engine = await build_engine()
        run = await engine.run(a_state("how long are incident records retained"))
        recorder.record_trace(
            request_spans(run.state, outcome="answered", started_at_ms=1_000, elapsed_ms=50)
        )

        exported = exporter.get_finished_spans()
        root = next(s for s in exported if s.name == "prag.request")
        children = [s for s in exported if s is not root]
        assert children
        assert {s.context.trace_id for s in exported} == {root.context.trace_id}
        root_id = root.context.span_id
        assert all(s.parent is not None and s.parent.span_id == root_id for s in children)
        assert root.start_time == 1_000 * 1_000_000
        assert root.end_time == 1_050 * 1_000_000

    def test_an_endpoint_without_the_extra_fails_at_startup(self, monkeypatch) -> None:
        """Starting up and silently exporting nothing would look exactly like a quiet system."""
        import builtins

        from prag.api.composition import _otel_tracer
        from prag.core.errors import ConfigurationError

        real_import = builtins.__import__

        def no_otel(name, *args, **kwargs):
            if name.startswith("opentelemetry"):
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_otel)
        with pytest.raises(ConfigurationError):
            _otel_tracer("http://collector:4318/v1/traces")

    def test_no_endpoint_means_no_exporter(self) -> None:
        from prag.api.composition import _otel_tracer

        assert _otel_tracer(None) is None

    async def test_every_attribute_is_in_the_schema(self) -> None:
        engine = await build_engine()
        run = await engine.run(a_state("how long are incident records retained"))
        for span in request_spans(run.state, outcome="answered", started_at_ms=0, elapsed_ms=1):
            assert set(span.attributes) <= SpanAttr.all_names(), span.name
