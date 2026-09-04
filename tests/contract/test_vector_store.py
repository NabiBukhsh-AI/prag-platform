"""Conformance suite for ``VectorStore``.

The assertions are the invariants that are cheap to hold in a list and easy to lose in a real
index: filters applied before scoring, ``top_k`` honoured after filtering, idempotent deletes,
and a dimension mismatch refused rather than silently producing nonsense.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from prag.core.errors import StorageError
from prag.core.models.common import Deadline
from prag.core.models.retrieval import VectorPoint
from prag.core.protocols import VectorStore
from prag.storage.vectorstore import InMemoryVectorStore, cosine_similarity

pytestmark = pytest.mark.contract

vector_store_implementations = [InMemoryVectorStore]


@pytest.fixture(params=vector_store_implementations, ids=lambda f: f.__name__)
def store(request: pytest.FixtureRequest) -> Any:
    return request.param()


def unit(*values: float) -> tuple[float, ...]:
    """A normalized 4-dimensional vector, so similarities are easy to reason about."""
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return tuple(v / norm for v in values)


def a_point(
    point_id: str,
    vector: tuple[float, ...],
    *,
    tenant_id: str = "tenant-a",
    acl_hash: str = "public",
    **payload: Any,
) -> VectorPoint:
    return VectorPoint(
        point_id=point_id,
        vector=vector,
        payload={"tenant_id": tenant_id, "acl_hash": acl_hash, **payload},
    )


class TestConformance:
    def test_satisfies_protocol(self, store: Any) -> None:
        assert isinstance(store, VectorStore)


class TestSearch:
    async def test_ranks_by_similarity(self, store: Any, deadline: Deadline) -> None:
        await store.upsert(
            "chunks",
            [
                a_point("near", unit(1.0, 0.0, 0.0, 0.0)),
                a_point("mid", unit(0.7, 0.7, 0.0, 0.0)),
                a_point("far", unit(0.0, 0.0, 0.0, 1.0)),
            ],
        )
        hits = await store.search("chunks", unit(1.0, 0.0, 0.0, 0.0), 3, None, deadline)

        assert [h.point_id for h in hits] == ["near", "mid", "far"]
        assert hits[0].score > hits[1].score > hits[2].score

    async def test_honours_top_k(self, store: Any, deadline: Deadline) -> None:
        await store.upsert(
            "chunks", [a_point(f"p{i}", unit(1.0, float(i), 0.0, 0.0)) for i in range(10)]
        )
        assert len(await store.search("chunks", unit(1.0, 0.0, 0.0, 0.0), 3, None, deadline)) == 3

    async def test_filters_apply_before_scoring(self, store: Any, deadline: Deadline) -> None:
        """A filtered top_k must return k *matching* results, not k minus the rejects.

        Scoring first and filtering after silently returns fewer results than asked for, and the
        shortfall looks like a sparse corpus rather than a bug.
        """
        await store.upsert(
            "chunks",
            [
                *(
                    a_point(f"other{i}", unit(1.0, 0.0, 0.0, 0.0), tenant_id="tenant-b")
                    for i in range(9)
                ),
                *(
                    a_point(f"mine{i}", unit(0.9, 0.1, 0.0, 0.0), tenant_id="tenant-a")
                    for i in range(3)
                ),
            ],
        )
        hits = await store.search(
            "chunks", unit(1.0, 0.0, 0.0, 0.0), 3, {"tenant_id": "tenant-a"}, deadline
        )
        assert len(hits) == 3
        assert all(h.point_id.startswith("mine") for h in hits)

    async def test_acl_membership_filter(self, store: Any, deadline: Deadline) -> None:
        """The filter that keeps unreadable content from ever being fetched."""
        await store.upsert(
            "chunks",
            [
                a_point("public-doc", unit(1.0, 0.0, 0.0, 0.0), acl_hash="public"),
                a_point("eng-doc", unit(1.0, 0.0, 0.0, 0.0), acl_hash="acl-eng"),
                a_point("hr-doc", unit(1.0, 0.0, 0.0, 0.0), acl_hash="acl-hr"),
            ],
        )
        hits = await store.search(
            "chunks",
            unit(1.0, 0.0, 0.0, 0.0),
            10,
            {"acl_hash": {"$in": ["acl-eng", "public"]}},
            deadline,
        )
        assert {h.point_id for h in hits} == {"public-doc", "eng-doc"}

    async def test_unknown_collection_is_empty_not_an_error(
        self, store: Any, deadline: Deadline
    ) -> None:
        """A collection not yet seeded is an ordinary state during ingestion."""
        assert (
            await store.search("never-created", unit(1.0, 0.0, 0.0, 0.0), 5, None, deadline) == ()
        )

    async def test_dimension_mismatch_is_refused(self, store: Any, deadline: Deadline) -> None:
        """Querying with the wrong dimensionality returns nonsense, not an error, if allowed."""
        await store.upsert("chunks", [a_point("p", unit(1.0, 0.0, 0.0, 0.0))])
        with pytest.raises(StorageError, match="dimension"):
            await store.search("chunks", (1.0, 0.0), 5, None, deadline)

    async def test_expired_deadline_is_refused(
        self, store: Any, expired_deadline: Deadline
    ) -> None:
        from prag.core.errors import DeadlineExceeded

        await store.upsert("chunks", [a_point("p", unit(1.0, 0.0, 0.0, 0.0))])
        with pytest.raises(DeadlineExceeded):
            await store.search("chunks", unit(1.0, 0.0, 0.0, 0.0), 5, None, expired_deadline)

    async def test_ordering_is_deterministic_on_ties(self, store: Any, deadline: Deadline) -> None:
        """Replay asserts identical decisions, which ties would otherwise break."""
        same = unit(1.0, 0.0, 0.0, 0.0)
        await store.upsert("chunks", [a_point(f"p{i}", same) for i in range(5)])

        first = await store.search("chunks", same, 5, None, deadline)
        second = await store.search("chunks", same, 5, None, deadline)
        assert [h.point_id for h in first] == [h.point_id for h in second]


class TestWrites:
    async def test_upsert_replaces_in_place(self, store: Any, deadline: Deadline) -> None:
        """Re-indexing unchanged content must not accumulate duplicates that all match."""
        await store.upsert("chunks", [a_point("p", unit(1.0, 0.0, 0.0, 0.0), text="old")])
        await store.upsert("chunks", [a_point("p", unit(1.0, 0.0, 0.0, 0.0), text="new")])

        hits = await store.search("chunks", unit(1.0, 0.0, 0.0, 0.0), 10, None, deadline)
        assert len(hits) == 1
        assert hits[0].payload["text"] == "new"

    async def test_dimension_mismatch_on_write_is_refused(self, store: Any) -> None:
        """Almost always a half-finished migration; accepting it degrades recall silently."""
        await store.upsert("chunks", [a_point("p", unit(1.0, 0.0, 0.0, 0.0))])
        with pytest.raises(StorageError, match="dimension"):
            await store.upsert("chunks", [a_point("q", (1.0, 0.0))])

    async def test_empty_upsert_is_harmless(self, store: Any) -> None:
        await store.upsert("chunks", [])

    async def test_delete_is_idempotent(self, store: Any, deadline: Deadline) -> None:
        """Erasure and reindex tombstoning both retry."""
        await store.upsert("chunks", [a_point("p", unit(1.0, 0.0, 0.0, 0.0))])
        await store.delete("chunks", ["p"])
        await store.delete("chunks", ["p"])
        await store.delete("chunks", ["never-existed"])

        assert await store.search("chunks", unit(1.0, 0.0, 0.0, 0.0), 5, None, deadline) == ()

    async def test_delete_on_unknown_collection_is_harmless(self, store: Any) -> None:
        await store.delete("nope", ["a"])


class TestCollectionInfo:
    async def test_reports_embedding_identity(self, store: Any) -> None:
        """The field that catches a query embedded by a different model than the index.

        A mismatch returns plausible nonsense rather than an error, so migration compares
        versions here instead of trusting a naming convention.
        """
        await store.upsert("chunks", [a_point("p", unit(1.0, 0.0, 0.0, 0.0))])
        info = await store.collection_info("chunks")

        assert info.name == "chunks"
        assert info.vector_count == 1
        assert info.dimensions == 4
        assert info.embedding_model
        assert info.embedding_version

    async def test_unknown_collection_raises(self, store: Any) -> None:
        with pytest.raises(StorageError):
            await store.collection_info("never-created")


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self) -> None:
        assert cosine_similarity((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self) -> None:
        assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)

    def test_zero_vector_scores_zero_rather_than_failing(self) -> None:
        """A degenerate embedding should rank last, not take every other candidate down."""
        assert cosine_similarity((0.0, 0.0), (1.0, 1.0)) == 0.0
