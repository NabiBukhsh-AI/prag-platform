"""Conformance suite for ``KnowledgeSource``.

Every implementation runs this. The assertions are the obligations stated on the protocol, and
they are the ones a new backend most often gets wrong: honouring the deadline, filtering by ACL
at the source, and reporting truncation distinctly from success.
"""

from __future__ import annotations

from typing import Any

import pytest

from prag.core.errors import SourceUnavailable
from prag.core.models.common import Deadline, HealthState
from prag.core.models.identity import Principal
from prag.core.models.retrieval import LegStatus, RetrievalLeg
from prag.core.protocols import KnowledgeSource
from tests.fakes.sources import make_candidate

pytestmark = pytest.mark.contract


def a_leg(top_k: int = 10, timeout_ms: int = 250) -> RetrievalLeg:
    return RetrievalLeg(
        leg_id="leg-1",
        source_id="fake.memory",
        query_variant="raw",
        top_k=top_k,
        timeout_ms=timeout_ms,
    )


@pytest.fixture
def corpus() -> list[Any]:
    """A small corpus spanning two ACL scopes and two lineage roots."""
    return [
        make_candidate("escalation policy for sev-1", candidate_id="c1", acl_hash="public"),
        make_candidate(
            "internal runbook detail",
            candidate_id="c2",
            document_id="doc-2",
            acl_hash="acl-engineering",
        ),
        make_candidate(
            "finance-only quarterly figures",
            candidate_id="c3",
            document_id="doc-3",
            acl_hash="acl-finance",
        ),
    ]


def test_satisfies_protocol(knowledge_source_factory: Any) -> None:
    assert isinstance(knowledge_source_factory(), KnowledgeSource)


def test_declares_capabilities(knowledge_source_factory: Any) -> None:
    """Capabilities must be real, so the planner can read them instead of branching on id."""
    source = knowledge_source_factory()
    assert source.source_id
    assert source.capabilities.max_top_k > 0
    assert any(
        (
            source.capabilities.supports_vectors,
            source.capabilities.supports_text,
        )
    ), "a source that supports neither vectors nor text cannot serve any leg"


async def test_filters_by_acl_at_the_source(
    knowledge_source_factory: Any, corpus: list[Any], principal: Principal, deadline: Deadline
) -> None:
    """The caller must never receive a candidate its ACL set does not cover.

    Checked at the source rather than downstream. The independent recheck is defence in depth,
    not a substitute: data fetched is data that can leak through a log line or a cache entry
    even when it never reaches the response.
    """
    source = knowledge_source_factory(corpus)
    result = await source.retrieve(a_leg(), principal, deadline)

    allowed = set(principal.acl_hashes) | {"public"}
    returned = {c.metadata.acl_hash for c in result.candidates}
    assert returned <= allowed, f"leaked candidates with ACL {returned - allowed}"


async def test_different_principals_see_different_candidates(
    knowledge_source_factory: Any,
    corpus: list[Any],
    principal: Principal,
    other_principal: Principal,
    deadline: Deadline,
) -> None:
    """The private document of one tenant must not reach another.

    The single failure this system most needs to prevent, asserted at the lowest layer that can
    prevent it.
    """
    source = knowledge_source_factory(corpus)
    mine = await source.retrieve(a_leg(), principal, deadline)
    theirs = await source.retrieve(a_leg(), other_principal, deadline)

    my_ids = {c.candidate_id for c in mine.candidates}
    their_ids = {c.candidate_id for c in theirs.candidates}
    assert "c2" in my_ids, "engineering principal should see the engineering document"
    assert "c2" not in their_ids, "finance principal must not see the engineering document"
    assert "c3" in their_ids, "finance principal should see the finance document"
    assert "c3" not in my_ids, "engineering principal must not see the finance document"


async def test_respects_top_k(
    knowledge_source_factory: Any, corpus: list[Any], principal: Principal, deadline: Deadline
) -> None:
    source = knowledge_source_factory(corpus)
    result = await source.retrieve(a_leg(top_k=1), principal, deadline)
    assert len(result.candidates) <= 1


async def test_returns_partial_rather_than_overrunning_the_deadline(
    knowledge_source_factory: Any, corpus: list[Any], principal: Principal
) -> None:
    """A source that cannot finish in time truncates and says so.

    Overrunning by 200 ms is not being slightly late: it spends budget belonging to a later
    stage, and the request pays for it at the reranker or the model. PARTIAL must be
    distinguishable from OK, or coverage warnings cannot be attributed.
    """
    source = knowledge_source_factory(corpus, latency_ms=5_000)
    result = await source.retrieve(a_leg(), principal, Deadline.in_ms(10, label="tight"))

    assert result.status is LegStatus.PARTIAL
    assert result.usable, "partial results are still usable; the caller decides what to do"


async def test_reports_latency(
    knowledge_source_factory: Any, corpus: list[Any], principal: Principal, deadline: Deadline
) -> None:
    source = knowledge_source_factory(corpus)
    result = await source.retrieve(a_leg(), principal, deadline)
    assert result.latency_ms >= 0
    assert result.leg_id == "leg-1"


async def test_failure_raises_a_typed_error(
    knowledge_source_factory: Any, principal: Principal, deadline: Deadline
) -> None:
    """Failures are typed, so the orchestrator can decide by class rather than by message."""
    source = knowledge_source_factory([], fail=True)
    with pytest.raises(SourceUnavailable) as excinfo:
        await source.retrieve(a_leg(), principal, deadline)
    assert excinfo.value.reason_code == "source_unavailable"
    assert excinfo.value.retryable is True


async def test_health_is_reported(knowledge_source_factory: Any) -> None:
    source = knowledge_source_factory([])
    health = await source.health()
    assert health.state in tuple(HealthState)
    assert health.usable is (health.state is not HealthState.UNAVAILABLE)


async def test_candidates_carry_lineage_and_authority(
    knowledge_source_factory: Any, corpus: list[Any], principal: Principal, deadline: Deadline
) -> None:
    """Metadata must arrive with the candidate.

    Fusion needs authority, freshness, and the lineage root. Fetching them later would put a
    database round trip inside a loop that runs fifty times per request, and the independence
    correction would be skipped the first time someone noticed the latency.
    """
    source = knowledge_source_factory(corpus)
    result = await source.retrieve(a_leg(), principal, deadline)
    assert result.candidates, "corpus should yield at least one visible candidate"

    for candidate in result.candidates:
        assert candidate.metadata.lineage_root
        assert 0.0 <= candidate.metadata.authority <= 1.0
        assert candidate.metadata.acl_hash
        assert candidate.effective_score == candidate.raw_score
