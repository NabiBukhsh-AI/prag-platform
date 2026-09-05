"""The HTTP surface: identity, answers, abstention, ingest, and error mapping."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from prag.api import build_platform
from prag.api.http import build_app
from prag.api.middleware import status_for, to_response
from prag.config import PragSettings
from prag.core.errors import (
    AbstentionRequired,
    AclRecheckMismatch,
    DeadlineExceeded,
    GuardrailBlocked,
    ProviderUnavailable,
    SourceUnavailable,
)
from prag.observability import SpanAttr

pytestmark = pytest.mark.integration

CORPUS = """# Incident Response

## Escalation

For a sev-1 incident the on-call lead must be paged within 15 minutes of detection. If the page
is unacknowledged after 5 minutes, escalation moves to the engineering manager, and after a
further 10 minutes to the director of engineering.

## Data retention

Incident records are retained for 30 days and then archived to cold storage, where they remain
queryable for a further 12 months before permanent deletion. Archived records stay searchable by
incident number throughout that window.
"""

TENANT = {"x-tenant-id": "tenant-a", "x-user-id": "u1"}


@pytest.fixture
async def client() -> TestClient:
    platform = build_platform(PragSettings())
    await platform.ingest_markdown(CORPUS, document_id="runbook-1", tenant_id="tenant-a")
    return TestClient(build_app(platform))


class TestHealth:
    def test_reports_corpus_state(self, client: TestClient) -> None:
        """An empty index answers everything with nothing, which hides a failed seed."""
        body = client.get("/health").json()

        assert body["status"] == "ok"
        assert body["indexed_vectors"] > 0
        assert body["parametric_enabled"] is False
        assert body["config_version"]


class TestIdentity:
    def test_a_missing_tenant_is_refused(self, client: TestClient) -> None:
        """Defaulting to a tenant is how a misconfigured client reads another tenant's corpus.

        The failure would look like the system working, which is why it is refused outright.
        """
        response = client.post("/v1/answer", json={"query": "anything"})

        assert response.status_code == 400
        assert response.json()["error"]["reason_code"] == "guardrail_blocked"

    def test_an_unknown_sla_tier_falls_back(self, client: TestClient) -> None:
        """A client typo should get a served request at a sane tier, not an outage."""
        response = client.post(
            "/v1/answer",
            json={"query": "how quickly must a sev-1 be escalated"},
            headers={**TENANT, "x-sla-tier": "not-a-tier"},
        )
        assert response.status_code == 200


class TestAnswer:
    def test_returns_an_envelope(self, client: TestClient) -> None:
        response = client.post(
            "/v1/answer",
            json={"query": "how quickly must a sev-1 be escalated"},
            headers=TENANT,
        )
        body = response.json()

        assert response.status_code == 200
        assert body["schema_version"] == "answer_envelope.v2"
        assert body["answer"]
        assert body["diagnostics"]["model_id"] == "mid.instruct"

    def test_the_envelope_carries_structured_confidence(self, client: TestClient) -> None:
        """Never prose hedging alone: a client that gates on a threshold needs the number."""
        body = client.post(
            "/v1/answer",
            json={"query": "how long are incident records retained"},
            headers=TENANT,
        ).json()

        assert 0.0 <= body["confidence"]["score"] <= 1.0
        assert body["confidence"]["band"] in {"low", "medium", "high"}
        assert body["confidence"]["basis"] == "retrieved_evidence"

    def test_citations_resolve_to_evidence(self, client: TestClient) -> None:
        body = client.post(
            "/v1/answer",
            json={"query": "how long are incident records retained"},
            headers=TENANT,
        ).json()

        assert body["citations"]
        for citation in body["citations"]:
            assert citation["document_id"] == "runbook-1"
            assert citation["entailment_score"] > 0.0

    def test_another_tenant_abstains_with_a_two_hundred(self, client: TestClient) -> None:
        """An abstention is an outcome of the request, not a failure to process it.

        A 4xx would make the abstention rate indistinguishable from client error in every
        dashboard that groups by status.
        """
        response = client.post(
            "/v1/answer",
            json={"query": "how quickly must a sev-1 be escalated"},
            headers={"x-tenant-id": "tenant-b", "x-user-id": "u2"},
        )
        body = response.json()

        assert response.status_code == 200
        assert body["abstention"] is not None
        assert body["abstention"]["reason_code"] == "private_query_no_evidence"
        assert body["abstention"]["suggested_action"], "an abstention must say what to do next"
        assert body["answer"] == ""

    def test_an_empty_query_is_rejected_by_validation(self, client: TestClient) -> None:
        assert client.post("/v1/answer", json={"query": ""}, headers=TENANT).status_code == 422


class TestStreaming:
    def test_emits_sse_frames_ending_with_the_envelope(self, client: TestClient) -> None:
        response = client.post(
            "/v1/answer",
            json={"query": "how quickly must a sev-1 be escalated", "stream": True},
            headers=TENANT,
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        frames = [f for f in response.text.split("\n\n") if f.strip()]
        assert any(f.startswith("event: token") for f in frames)

        done = next(f for f in frames if f.startswith("event: done"))
        envelope = json.loads(done.split("data: ", 1)[1])
        assert envelope["schema_version"] == "answer_envelope.v2"

    def test_an_abstention_still_ends_with_an_envelope(self, client: TestClient) -> None:
        """A client that only reads `done` gets a complete, accurate answer either way."""
        response = client.post(
            "/v1/answer",
            json={"query": "anything at all", "stream": True},
            headers={"x-tenant-id": "tenant-zzz", "x-user-id": "u"},
        )
        done = next(f for f in response.text.split("\n\n") if f.startswith("event: done"))
        envelope = json.loads(done.split("data: ", 1)[1])

        assert envelope["abstention"] is not None


class TestIngest:
    def test_indexes_a_document(self, client: TestClient) -> None:
        response = client.post(
            "/v1/ingest",
            json={"document_id": "policy-1", "content": "# Policy\n\n" + "Body text. " * 60},
            headers=TENANT,
        )
        assert response.status_code == 201
        assert response.json()["chunks_indexed"] > 0

    def test_writes_under_the_callers_tenant(self, client: TestClient) -> None:
        """Trusting a tenant field in the body would let any caller write into any corpus.

        That is the write-side version of the leak the read path guards against.
        """
        client.post(
            "/v1/ingest",
            json={
                "document_id": "sneaky",
                "content": "# Secret\n\n" + "Confidential body text. " * 40,
            },
            headers={"x-tenant-id": "tenant-writer", "x-user-id": "u"},
        )

        # The writer can see it.
        mine = client.post(
            "/v1/answer",
            json={"query": "confidential body text"},
            headers={"x-tenant-id": "tenant-writer", "x-user-id": "u"},
        ).json()
        assert mine["abstention"] is None

        # The original tenant cannot.
        theirs = client.post(
            "/v1/answer", json={"query": "confidential body text"}, headers=TENANT
        ).json()
        assert "Confidential body text" not in theirs.get("answer", "")

    def test_ingest_requires_a_tenant(self, client: TestClient) -> None:
        response = client.post(
            "/v1/ingest", json={"document_id": "d", "content": "some content here"}
        )
        assert response.status_code == 400


class TestErrorMapping:
    """One place decides status, so a subsystem adding a failure never touches transport."""

    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (GuardrailBlocked("x", guardrail="g", phase="input"), 400),
            (DeadlineExceeded("x"), 504),
            (SourceUnavailable("x"), 503),
            (ProviderUnavailable("x"), 502),
            (AclRecheckMismatch("x"), 500),
        ],
    )
    def test_status_mapping(self, error: Exception, expected: int) -> None:
        assert status_for(error) == expected

    def test_an_isolation_violation_is_ours_not_the_callers(self) -> None:
        """A 403 would confirm that something exists which the caller may not see.

        A violation means the platform's own boundary failed, which is the platform's fault.
        """
        assert status_for(AclRecheckMismatch("canary")) == 500

    def test_an_abstention_is_not_an_error_status(self) -> None:
        assert status_for(AbstentionRequired("x", abstention_code="y")) == 200

    def test_an_untyped_exception_leaks_nothing(self) -> None:
        """An unexpected exception's text is the likeliest place for an internal detail."""
        response = to_response(RuntimeError("connection to 10.0.3.14:5432 refused"))

        assert response.status == 500
        assert response.reason_code == "internal_error"
        assert "10.0.3.14" not in response.detail

    def test_a_typed_error_keeps_its_reason_code(self) -> None:
        response = to_response(SourceUnavailable("down", source_id="vector.primary"))

        assert response.reason_code == "source_unavailable"
        assert response.retryable is True
        assert "vector.primary" not in response.detail, "context stays out of the client body"


class TestTracing:
    def test_a_request_records_a_span_with_the_schema(self, client: TestClient) -> None:
        """A span emitted with the wrong attribute name fails nothing at runtime."""
        platform = build_platform(PragSettings())
        app_client = TestClient(build_app(platform))
        app_client.post("/v1/answer", json={"query": "anything"}, headers=TENANT)

        spans = platform.tracer.named("http.answer")
        assert spans, "the request path must record a span"

        recorded = set(spans[0].attributes)
        assert SpanAttr.REQUEST_ID in recorded
        assert SpanAttr.TENANT_ID in recorded
        assert recorded <= SpanAttr.all_names(), "every attribute comes from the schema"

    def test_query_text_never_reaches_a_span(self) -> None:
        """Redaction is a property of the recorder, not of its callers."""
        from prag.observability import TraceRecorder

        tracer = TraceRecorder()
        with tracer.span("x", query_text="a secret question", tenant_id="t"):
            pass

        attributes = tracer.spans[0].attributes
        assert "query_text" not in attributes
        assert attributes["tenant_id"] == "t"
