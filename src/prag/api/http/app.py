"""The FastAPI application.

This module imports FastAPI at the top and deliberately does **not** use
``from __future__ import annotations``. Both choices are forced by how FastAPI resolves
endpoint signatures: it inspects annotations at decoration time, so deferred (string)
annotations referring to names local to a factory function cannot be resolved, and every
parameter silently degrades into a query parameter. The symptom is a 422 on a perfectly good
request body, which points nowhere near the cause.

``prag.api`` does not import this module, so the optional ``api`` extra stays optional: a worker
that never serves HTTP does not need a web framework on its path.

The endpoint worth reading is ``/v1/answer``. It returns an ``AnswerEnvelope`` on success *and*
on abstention, because an abstention is an outcome of the request rather than a failure to
process it, and a client should parse one shape either way.
"""

import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from prag.api.composition import Platform
from prag.api.middleware import resolve_principal, to_response
from prag.api.sse import SseEvent, error_frame, format_sse
from prag.core.errors import AbstentionRequired, PragError
from prag.core.ids import new_request_id, new_trace_id
from prag.core.models.identity import Budget
from prag.core.models.state import RequestState
from prag.observability.tracing import SpanAttr

__all__ = ["AnswerRequest", "IngestRequest", "build_app"]

#: Wall-clock budget per SLA tier. Interactive is tight because the design targets a 650 ms time
#: to first token; batch is generous because nobody is waiting on it.
_WALL_MS_BY_TIER: dict[str, int] = {
    "interactive": 3_000,
    "standard": 10_000,
    "high_stakes": 30_000,
    "batch": 120_000,
}
_USD_BY_TIER: dict[str, float] = {
    "interactive": 0.02,
    "standard": 0.05,
    "high_stakes": 0.50,
    "batch": 0.20,
}


class AnswerRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8_000)
    stream: bool = False


class IngestRequest(BaseModel):
    """An ingest request.

    Deliberately carries no tenant field. The document is written under the *caller's* tenant,
    because trusting a body field here would let any caller write into any tenant's corpus —
    the write-side version of the leak the read path guards against.
    """

    document_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source_id: str = "kb.seed"
    acl_hash: str = "public"
    authority: float = Field(default=0.8, ge=0.0, le=1.0)


def _state_for(platform: Platform, query: str, headers: dict[str, str]) -> RequestState:
    from prag.config import resolve_tenant_policy

    principal = resolve_principal(headers)
    tier = str(principal.sla_tier)
    wall_ms = _WALL_MS_BY_TIER.get(tier, 10_000)

    return RequestState(
        request_id=new_request_id(),
        trace_id=new_trace_id(),
        principal=principal,
        policy=resolve_tenant_policy(platform.settings, principal.tenant_id),
        budget=Budget(
            wall_ms_total=wall_ms,
            wall_ms_remaining=wall_ms,
            usd_total=_USD_BY_TIER.get(tier, 0.05),
            max_tokens_in=platform.settings.context.max_evidence_tokens,
            max_tokens_out=2_048,
        ),
        raw_query=query,
    )


def build_app(platform: Platform) -> FastAPI:
    """Construct the ASGI app over an already-wired platform.

    The platform is built once at startup and passed in, so the graph is validated before the
    first request rather than on it — a broken graph fails the process instead of failing
    whichever caller happens to arrive first.
    """
    app = FastAPI(
        title="prag-platform",
        version="0.1.0",
        description=(
            "Parametric RAG platform: routes each query between weight-resident, "
            "adapter-retrieved and text-retrieved knowledge."
        ),
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Liveness plus corpus state.

        Reports the indexed vector count, because an empty index answers every query with
        nothing and that is indistinguishable from a corpus with no match — which is exactly
        what makes a failed seed hard to notice.
        """
        return {
            "status": "ok",
            "config_version": platform.settings.config_version,
            "indexed_vectors": platform.store.count(platform.collection),
            "parametric_enabled": platform.settings.parametric.enabled,
        }

    @app.post("/v1/answer")
    async def answer(body: AnswerRequest, request: Request) -> Any:
        headers = {k.lower(): v for k, v in request.headers.items()}

        try:
            state = _state_for(platform, body.query, headers)
        except PragError as exc:
            response = to_response(exc)
            return JSONResponse(status_code=response.status, content=response.as_dict())

        if body.stream:
            return StreamingResponse(_stream(platform, state), media_type="text/event-stream")

        with platform.tracer.span(
            "http.answer",
            **{
                SpanAttr.REQUEST_ID: state.request_id,
                SpanAttr.TRACE_ID: state.trace_id,
                SpanAttr.TENANT_ID: state.principal.tenant_id,
                SpanAttr.SLA_TIER: str(state.principal.sla_tier),
            },
        ) as span:
            try:
                run = await platform.engine.run(state)
            except AbstentionRequired as exc:
                # An abstention is a 200 carrying an envelope. A 4xx would make the abstention
                # rate indistinguishable from client error in every dashboard that groups by
                # status, and abstention rate is a metric with both an upper and a lower alert.
                span.attributes[SpanAttr.ABSTAINED] = True
                span.attributes[SpanAttr.ABSTENTION_REASON] = exc.abstention_code
                return JSONResponse(status_code=200, content=_abstention_envelope(state, exc))
            except PragError as exc:
                span.attributes[SpanAttr.ERROR_REASON_CODE] = exc.reason_code
                response = to_response(exc)
                return JSONResponse(status_code=response.status, content=response.as_dict())

            envelope = run.state.result
            if envelope is None:  # pragma: no cover - the graph guarantees a terminal write
                response = to_response(RuntimeError("graph produced no envelope"))
                return JSONResponse(status_code=response.status, content=response.as_dict())

            span.attributes.update(
                {
                    SpanAttr.MODEL_ID: envelope.diagnostics.model_id,
                    SpanAttr.GROUNDEDNESS: envelope.grounding.groundedness,
                    SpanAttr.CLAIMS_UNSOURCED: envelope.grounding.claims_unsourced,
                    SpanAttr.BUDGET_DEGRADATION_LEVEL: envelope.diagnostics.degradation_level,
                    SpanAttr.USD_COST: envelope.diagnostics.usd_cost,
                    SpanAttr.ABSTAINED: envelope.abstained,
                }
            )
            return JSONResponse(status_code=200, content=envelope.model_dump(mode="json"))

    @app.post("/v1/ingest")
    async def ingest(body: IngestRequest, request: Request) -> Any:
        headers = {k.lower(): v for k, v in request.headers.items()}
        try:
            principal = resolve_principal(headers)
        except PragError as exc:
            response = to_response(exc)
            return JSONResponse(status_code=response.status, content=response.as_dict())

        indexed = await platform.ingest_markdown(
            body.content,
            document_id=body.document_id,
            tenant_id=principal.tenant_id,
            source_id=body.source_id,
            acl_hash=body.acl_hash,
            authority=body.authority,
        )
        return JSONResponse(
            status_code=201,
            content={"document_id": body.document_id, "chunks_indexed": indexed},
        )

    return app


async def _stream(platform: Platform, state: RequestState):
    """Run the graph and render it as SSE.

    Phase 1 runs the graph to completion and then emits the answer, because grounding
    verification needs the whole answer and the local provider produces it in one step. The
    sentence-buffered path in ``generation.streaming`` is what token-level streaming will use,
    and the event protocol is already shaped for it — including corrections.
    """
    started = time.monotonic()
    try:
        run = await platform.engine.run(state)
    except AbstentionRequired as exc:
        yield format_sse(SseEvent.DONE, _abstention_envelope(state, exc))
        return
    except PragError as exc:
        yield error_frame(exc.reason_code, exc.detail, retryable=exc.retryable)
        return

    envelope = run.state.result
    if envelope is None:  # pragma: no cover
        yield error_frame("internal_error", "no envelope produced", retryable=False)
        return

    yield format_sse(
        SseEvent.TOKEN,
        {"text": envelope.answer, "ttft_ms": int((time.monotonic() - started) * 1000)},
    )
    yield format_sse(SseEvent.DONE, envelope.model_dump(mode="json"))


def _abstention_envelope(state: RequestState, exc: AbstentionRequired) -> dict[str, Any]:
    """Render an abstention as a complete envelope.

    Every abstention carries a machine-readable reason and, where possible, a suggested action.
    One that does not say why is a failure of the abstention path rather than a use of it.
    """
    from prag.core.models.fusion import (
        Abstention,
        AbstentionCode,
        ConfidenceBand,
        ConfidenceBlock,
        KnowledgeBasis,
    )
    from prag.core.models.generation import AnswerEnvelope, Diagnostics, GroundingReport

    try:
        code = AbstentionCode(exc.abstention_code)
    except ValueError:
        code = AbstentionCode.KNOWLEDGE_BELOW_FLOOR

    envelope = AnswerEnvelope(
        request_id=state.request_id,
        answer="",
        confidence=ConfidenceBlock(
            score=0.0, band=ConfidenceBand.LOW, basis=KnowledgeBasis.ABSTAIN
        ),
        grounding=GroundingReport(claims_total=0, claims_cited=0, claims_unsourced=0),
        abstention=Abstention(
            reason_code=code,
            explanation=exc.detail,
            suggested_action=exc.suggested_action,
        ),
        diagnostics=Diagnostics(
            route_class="abstain",
            strategy="NON_PARAMETRIC",
            model_id="none",
            model_version="none",
            total_ms=state.total_elapsed_ms,
            degradation_level=state.budget.degradation_level,
        ),
    )
    return envelope.model_dump(mode="json")
