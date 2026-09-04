"""The first end-to-end slice: normalize, chunk, index, retrieve.

Every piece is real except the embedder, which is the deterministic fake. That combination is
the point — it exercises the actual wiring between ingestion and retrieval while keeping the
assertions about *behaviour* rather than about a particular model's opinion of similarity.
"""

from __future__ import annotations

import pytest

from prag.core.errors import SourceUnavailable
from prag.core.models.common import Deadline
from prag.core.models.identity import Principal
from prag.core.models.retrieval import LegStatus, RetrievalLeg
from prag.core.protocols import KnowledgeSource
from prag.ingestion import index_chunks, normalize_markdown, validate_chunks
from prag.ingestion.chunking import StructureAwareChunker
from prag.retrieval import VectorKnowledgeSource
from prag.storage.vectorstore import InMemoryVectorStore, acl_filter
from prag.storage.vectorstore.filters import FilterError, matches
from tests.fakes.providers import DeterministicEmbeddingProvider

RUNBOOK = """# Incident Response

## Severity levels

Sev-1 means a total outage affecting every tenant of the platform. It pages the
on-call lead immediately and opens a bridge call for the duration.

Sev-2 means degraded service for a subset of tenants, paged in business hours.

## Escalation

For a sev-1 incident the on-call lead must be paged within 15 minutes of
detection. Unacknowledged after 5 minutes, escalation moves to the engineering
manager, and after a further 10 minutes to the director.

## Data retention

Incident records are retained for 30 days and then archived to cold storage,
where they remain queryable for a further 12 months before deletion.
"""


@pytest.fixture
def principal() -> Principal:
    return Principal(tenant_id="tenant-a", user_id="u1", acl_hashes=("acl-eng",))


async def seeded_source(
    *,
    acl_hash: str = "public",
    tenant_id: str = "tenant-a",
    store: InMemoryVectorStore | None = None,
) -> tuple[VectorKnowledgeSource, InMemoryVectorStore, int]:
    """Run the real pipeline and return a source over the result."""
    store = store or InMemoryVectorStore()
    embedder = DeterministicEmbeddingProvider(dimensions=32)

    doc = normalize_markdown(
        RUNBOOK,
        document_id="runbook-1",
        tenant_id=tenant_id,
        source_id="kb.runbooks",
        acl_hash=acl_hash,
        authority=0.9,
    )
    chunks = validate_chunks(
        StructureAwareChunker(min_section_tokens=20, target_child_tokens=60).chunk(doc)
    ).kept
    written = await index_chunks(
        chunks,
        store=store,
        embedder=embedder,
        collection="chunks",
        deadline=Deadline.in_ms(5_000, label="seed"),
    )
    source = VectorKnowledgeSource(
        source_id="vector.primary", store=store, embedder=embedder, collection="chunks"
    )
    return source, store, written


def a_leg(text: str, *, top_k: int = 5, **overrides: object) -> RetrievalLeg:
    base: dict[str, object] = {
        "leg_id": "leg-1",
        "source_id": "vector.primary",
        "query_variant": "raw",
        "query_text": text,
        "top_k": top_k,
        "timeout_ms": 500,
    }
    return RetrievalLeg(**{**base, **overrides})  # type: ignore[arg-type]


class TestFilterDialect:
    def test_absent_filter_matches_everything(self) -> None:
        assert matches({"a": 1}, None)
        assert matches({"a": 1}, {})

    def test_equality_and_membership(self) -> None:
        assert matches({"tenant_id": "t"}, {"tenant_id": "t"})
        assert not matches({"tenant_id": "t"}, {"tenant_id": "other"})
        assert matches({"acl_hash": "x"}, {"acl_hash": {"$in": ["x", "y"]}})

    def test_ranges_treat_a_missing_field_as_no_match(self) -> None:
        """Absence is not zero.

        Treating it as zero would sweep every undated document into a "newer than" filter.
        """
        assert not matches({}, {"updated_at_ms": {"$gt": 0}})
        assert matches({"updated_at_ms": 5}, {"updated_at_ms": {"$gt": 0}})

    def test_boolean_composition(self) -> None:
        payload = {"tenant_id": "t", "authority": 0.9}
        assert matches(payload, {"$and": [{"tenant_id": "t"}, {"authority": {"$gte": 0.5}}]})
        assert matches(payload, {"$or": [{"tenant_id": "other"}, {"authority": {"$gte": 0.5}}]})
        assert matches(payload, {"$not": {"tenant_id": "other"}})

    def test_unknown_operator_raises(self) -> None:
        """A filter matching nothing because it was misspelled looks like an empty corpus."""
        with pytest.raises(FilterError, match="unknown filter operator"):
            matches({"a": 1}, {"a": {"$regex": ".*"}})

    def test_acl_filter_always_includes_public(self) -> None:
        """Otherwise an unprivileged caller can see nothing at all."""
        built = acl_filter("tenant-a", ("acl-eng",))
        assert built["tenant_id"] == "tenant-a"
        assert "public" in built["acl_hash"]["$in"]


class TestIngestToRetrieve:
    async def test_the_slice_works(self, principal: Principal) -> None:
        source, _, written = await seeded_source()
        assert written > 0

        result = await source.retrieve(
            a_leg("how quickly must a sev-1 be escalated"),
            principal,
            Deadline.in_ms(2_000, label="test"),
        )
        assert result.status is LegStatus.OK
        assert result.candidates

    async def test_retrieved_candidate_carries_the_parent(self, principal: Principal) -> None:
        """The whole point of parent-child: match the child, read the parent."""
        source, _, _ = await seeded_source()
        result = await source.retrieve(
            a_leg("escalation timing"), principal, Deadline.in_ms(2_000, label="t")
        )

        with_parent = [c for c in result.candidates if c.parent_text]
        assert with_parent, "sections larger than one child must carry a parent"
        for candidate in with_parent:
            assert candidate.text in candidate.parent_text
            assert candidate.context_text == candidate.parent_text

    async def test_metadata_survives_the_round_trip(self, principal: Principal) -> None:
        """Fusion needs authority, lineage and the ACL hash without a second fetch."""
        source, _, _ = await seeded_source()
        result = await source.retrieve(
            a_leg("retention"), principal, Deadline.in_ms(2_000, label="t")
        )

        candidate = result.candidates[0]
        assert candidate.metadata.authority == pytest.approx(0.9)
        assert candidate.metadata.lineage_root == "runbook-1"
        assert candidate.metadata.embedding_version
        assert candidate.document_id == "runbook-1"
        assert candidate.source_id == "vector.primary"

    async def test_ranks_are_recorded_per_leg(self, principal: Principal) -> None:
        """Rank-based fusion needs the ranks, and per-leg ranks show which source found it."""
        source, _, _ = await seeded_source()
        result = await source.retrieve(
            a_leg("severity levels"), principal, Deadline.in_ms(2_000, label="t")
        )
        assert [c.rank_by_leg["leg-1"] for c in result.candidates] == list(
            range(len(result.candidates))
        )

    async def test_top_k_is_honoured(self, principal: Principal) -> None:
        source, _, _ = await seeded_source()
        result = await source.retrieve(
            a_leg("incident", top_k=2), principal, Deadline.in_ms(2_000, label="t")
        )
        assert len(result.candidates) <= 2

    async def test_reindexing_unchanged_content_does_not_duplicate(self) -> None:
        """Chunk ids are derived, so a re-run overwrites in place."""
        store = InMemoryVectorStore()
        _, _, first = await seeded_source(store=store)
        count_after_first = store.count("chunks")
        _, _, second = await seeded_source(store=store)

        assert first == second
        assert store.count("chunks") == count_after_first


class TestIsolation:
    async def test_another_tenant_sees_nothing(self) -> None:
        """The filter is built from the principal and applied at the store."""
        source, _, _ = await seeded_source(tenant_id="tenant-a")
        intruder = Principal(tenant_id="tenant-b", user_id="u2", acl_hashes=("acl-eng",))

        result = await source.retrieve(
            a_leg("escalation"), intruder, Deadline.in_ms(2_000, label="t")
        )
        assert result.candidates == ()

    async def test_acl_restricted_content_is_not_fetched(self) -> None:
        """Content the caller may not see must never be retrieved, not merely not returned.

        Data that has been fetched can leak through a log line, a cache entry, or a timing
        difference even when it never reaches the response.
        """
        source, _, _ = await seeded_source(acl_hash="acl-finance")
        engineer = Principal(tenant_id="tenant-a", user_id="u1", acl_hashes=("acl-eng",))

        result = await source.retrieve(
            a_leg("escalation"), engineer, Deadline.in_ms(2_000, label="t")
        )
        assert result.candidates == ()

    async def test_the_right_principal_does_see_it(self) -> None:
        source, _, _ = await seeded_source(acl_hash="acl-finance")
        finance = Principal(tenant_id="tenant-a", user_id="u3", acl_hashes=("acl-finance",))

        result = await source.retrieve(
            a_leg("escalation"), finance, Deadline.in_ms(2_000, label="t")
        )
        assert result.candidates

    async def test_a_leg_cannot_widen_its_own_scope(self) -> None:
        """The principal's filter is merged over the leg's, never under it."""
        source, _, _ = await seeded_source(tenant_id="tenant-a")
        intruder = Principal(tenant_id="tenant-b", user_id="u2")

        result = await source.retrieve(
            a_leg("escalation", filters={"tenant_id": "tenant-a"}),
            intruder,
            Deadline.in_ms(2_000, label="t"),
        )
        assert result.candidates == (), "a leg must not be able to override the ACL filter"


class TestSourceBehaviour:
    async def test_satisfies_the_protocol(self) -> None:
        source, _, _ = await seeded_source()
        assert isinstance(source, KnowledgeSource)

    async def test_declares_honest_capabilities(self) -> None:
        """Dense retrieval matches meaning, not terms.

        Claiming text support would let the planner route identifier-heavy queries here, where
        it returns plausible neighbours instead of the exact match BM25 would have found.
        """
        source, _, _ = await seeded_source()
        assert source.capabilities.supports_vectors
        assert not source.capabilities.supports_text
        assert source.capabilities.supports_filters

    async def test_missing_query_text_is_a_typed_error(self, principal: Principal) -> None:
        source, _, _ = await seeded_source()
        with pytest.raises(SourceUnavailable, match="query text"):
            await source.retrieve(a_leg("   "), principal, Deadline.in_ms(2_000, label="t"))

    async def test_reports_partial_when_out_of_time(self, principal: Principal) -> None:
        """A leg with no time left reports partial rather than raising.

        The plan may have other legs that did return, and a coverage warning beats a failure.
        """
        source, _, _ = await seeded_source()
        result = await source.retrieve(
            a_leg("escalation"), principal, Deadline.in_ms(0, label="spent")
        )
        assert result.status is LegStatus.PARTIAL
        assert result.error_reason_code == "deadline_exceeded"
        assert result.usable

    async def test_empty_collection_is_degraded_not_healthy(self) -> None:
        """An empty index answers everything with nothing, which hides a failed seed."""
        store = InMemoryVectorStore()
        store.create_collection("chunks", dimensions=32)
        source = VectorKnowledgeSource(
            source_id="vector.primary",
            store=store,
            embedder=DeterministicEmbeddingProvider(dimensions=32),
            collection="chunks",
        )
        health = await source.health()
        assert health.state.value == "degraded"
        assert health.usable

    async def test_seeded_collection_is_healthy(self) -> None:
        source, _, _ = await seeded_source()
        assert (await source.health()).state.value == "healthy"

    async def test_unknown_collection_is_unavailable(self) -> None:
        source = VectorKnowledgeSource(
            source_id="vector.primary",
            store=InMemoryVectorStore(),
            embedder=DeterministicEmbeddingProvider(dimensions=32),
            collection="never-created",
        )
        assert not (await source.health()).usable

    async def test_a_timeout_does_not_implicate_source_health(self, principal: Principal) -> None:
        """A slow request must not open the breaker on a healthy source.

        Wrapping a deadline breach as SourceUnavailable would take that source down for every
        other request too, turning one slow query into an outage.
        """
        source, _, _ = await seeded_source()
        result = await source.retrieve(
            a_leg("escalation"), principal, Deadline.in_ms(0, label="spent")
        )

        assert result.status is LegStatus.PARTIAL
        assert result.error_reason_code == "deadline_exceeded"
        assert (await source.health()).state.value == "healthy", "the source is fine"
