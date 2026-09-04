"""An in-memory ``VectorStore``.

Exact cosine search over a list. That is the right trade for what it is used for: the local
stack, unit tests, and the seeded corpus. An approximate index would introduce recall variance
into tests whose whole purpose is to assert exact behaviour, and at seed-corpus scale exact
search is faster than building an index anyway.

It is also the reference implementation the conformance suite runs first, so a pgvector or
Qdrant adapter that diverges from it fails the suite rather than surprising someone in
production. The invariants it holds are the ones easy to lose in a real backend: filters applied
before scoring, ``top_k`` honoured, deletes idempotent, and an embedding-dimension mismatch
refused rather than silently producing nonsense.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from prag.core.errors import StorageError
from prag.core.models.retrieval import ScoredPoint
from prag.storage.vectorstore.filters import matches

if TYPE_CHECKING:
    from collections.abc import Sequence

    from prag.core.models.common import Deadline
    from prag.core.models.retrieval import FilterExpr, VectorPoint

__all__ = ["InMemoryVectorStore", "cosine_similarity"]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, returning 0.0 for a zero-magnitude vector.

    Zero rather than an error: an empty or all-zero embedding is a degenerate input that should
    rank last, not fail the whole search and take every other candidate with it.
    """
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


class _Collection:
    __slots__ = ("dimensions", "embedding_model", "embedding_version", "points")

    def __init__(self, dimensions: int, model: str, version: str) -> None:
        self.dimensions = dimensions
        self.embedding_model = model
        self.embedding_version = version
        self.points: dict[str, tuple[tuple[float, ...], dict[str, Any]]] = {}


class InMemoryVectorStore:
    """Exact-search vector storage, keyed by collection."""

    def __init__(
        self,
        *,
        dimensions: int = 16,
        embedding_model: str = "fake.embed",
        embedding_version: str = "v1",
    ) -> None:
        self._default_dimensions = dimensions
        self._default_model = embedding_model
        self._default_version = embedding_version
        self._collections: dict[str, _Collection] = {}
        self.search_calls = 0

    def _collection(self, name: str, *, create_with: int | None = None) -> _Collection:
        existing = self._collections.get(name)
        if existing is not None:
            return existing
        if create_with is None:
            raise StorageError("unknown collection", collection=name)
        self._collections[name] = _Collection(
            create_with, self._default_model, self._default_version
        )
        return self._collections[name]

    async def upsert(self, collection: str, points: Sequence[VectorPoint]) -> None:
        if not points:
            return

        target = self._collection(collection, create_with=len(points[0].vector))
        for point in points:
            if len(point.vector) != target.dimensions:
                # A dimension mismatch is almost always a half-finished embedding migration.
                # Accepting it would fill the collection with vectors that cannot be compared
                # to anything, and the symptom would be poor recall rather than an error.
                raise StorageError(
                    "vector dimension does not match the collection",
                    collection=collection,
                    expected=target.dimensions,
                    received=len(point.vector),
                    point_id=point.point_id,
                )
            target.points[point.point_id] = (tuple(point.vector), dict(point.payload))

    async def search(
        self,
        collection: str,
        vector: Sequence[float],
        top_k: int,
        filters: FilterExpr | None,
        deadline: Deadline,
    ) -> Sequence[ScoredPoint]:
        self.search_calls += 1
        deadline.raise_if_expired()

        target = self._collections.get(collection)
        if target is None:
            # An empty result rather than an error. A collection that has not been seeded yet
            # is an ordinary state during ingestion, and failing the request would make a
            # half-populated system look broken rather than incomplete.
            return ()

        if len(vector) != target.dimensions:
            raise StorageError(
                "query vector dimension does not match the collection",
                collection=collection,
                expected=target.dimensions,
                received=len(vector),
            )

        # Filter before scoring, not after. Scoring first would compute similarity against
        # content the caller may not see, and a top_k applied after filtering would silently
        # return fewer results than asked for.
        scored = [
            ScoredPoint(point_id=point_id, score=cosine_similarity(vector, stored), payload=payload)
            for point_id, (stored, payload) in target.points.items()
            if matches(payload, filters)
        ]
        scored.sort(key=lambda p: (-p.score, p.point_id))
        return tuple(scored[:top_k])

    async def delete(self, collection: str, ids: Sequence[str]) -> None:
        """Remove points by id. Idempotent.

        Deletion is driven by right-to-erasure and by tombstoning during reindex, and both
        retry, so deleting an already-absent id is a success rather than an error.
        """
        target = self._collections.get(collection)
        if target is None:
            return
        for point_id in ids:
            target.points.pop(point_id, None)

    async def collection_info(self, collection: str):
        from prag.core.models.common import CollectionInfo

        target = self._collection(collection)
        return CollectionInfo(
            name=collection,
            vector_count=len(target.points),
            dimensions=target.dimensions,
            embedding_model=target.embedding_model,
            embedding_version=target.embedding_version,
        )

    # ------------------------------------------------------------------
    # Test and local-stack conveniences, deliberately off the protocol.
    # ------------------------------------------------------------------

    def create_collection(
        self, name: str, *, dimensions: int, model: str | None = None, version: str | None = None
    ) -> None:
        """Create an empty collection with an explicit embedding identity.

        Not on the protocol: real backends provision collections through their own tooling or
        through migrations, and putting provisioning on the read/write interface would oblige
        every adapter to implement an operation most deployments do out of band.
        """
        self._collections[name] = _Collection(
            dimensions, model or self._default_model, version or self._default_version
        )

    @property
    def collections(self) -> tuple[str, ...]:
        return tuple(sorted(self._collections))

    def count(self, collection: str) -> int:
        target = self._collections.get(collection)
        return len(target.points) if target else 0
