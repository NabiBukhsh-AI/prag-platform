"""Retrieval protocols: sources, the orchestrator, and the stores beneath them.

``KnowledgeSource`` is the extension point the platform is designed around. Adding a source is a
new implementation plus a config entry; the planner and the orchestrator do not change, because
they read ``capabilities`` rather than branching on ``source_id``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from prag.core.models.common import (
    CollectionInfo,
    Deadline,
    HealthStatus,
    SourceCapabilities,
)
from prag.core.models.identity import Budget, Principal
from prag.core.models.retrieval import (
    CandidatePool,
    FilterExpr,
    LegResult,
    RetrievalLeg,
    RetrievalPlan,
    ScoredPoint,
    VectorPoint,
)

__all__ = [
    "KnowledgeSource",
    "LexicalStore",
    "RetrievalOrchestrator",
    "VectorStore",
]


@runtime_checkable
class KnowledgeSource(Protocol):
    """A queryable source of non-parametric knowledge."""

    source_id: str
    capabilities: SourceCapabilities

    async def retrieve(
        self,
        leg: RetrievalLeg,
        principal: Principal,
        deadline: Deadline,
    ) -> LegResult:
        """Return candidates, or raise a ``RetrievalError``.

        Two obligations, and neither is optional:

        **Respect the deadline, returning partial results rather than exceeding it.** A source
        that overruns by 200 ms has not been slightly late; it has spent budget that belonged
        to a later stage, and the request will pay for it at the reranker or the model.

        **Apply the principal's ACL filters at the source.** An independent recheck happens
        downstream, but that is defence in depth and not a substitute. Filtering late means
        fetching data the caller may not see, and data fetched is data that can leak through a
        log line, a cache entry, or a timing difference.
        """
        ...

    async def health(self) -> HealthStatus:
        """Current health, for the breaker and the planner.

        Cheap and non-blocking. A health check that itself times out has told the caller
        nothing while costing what a real query would have.
        """
        ...


@runtime_checkable
class RetrievalOrchestrator(Protocol):
    """Executes a plan across sources, in parallel, within a budget."""

    async def execute(
        self,
        plan: RetrievalPlan,
        principal: Principal,
        budget: Budget,
    ) -> CandidatePool:
        """Run every leg and collect the results.

        Legs run concurrently and are individually deadlined. A failed optional leg produces a
        degraded pool rather than an exception; a failed required leg is decided by the plan's
        partial-results policy. The orchestrator never waits for a slow leg past the plan
        budget, because the plan budget is what the rest of the request was sized against.
        """
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Dense vector storage and search.

    Deliberately narrow. Anything richer would leak the backing store's model into callers, and
    the point of this boundary is that swapping Qdrant for pgvector or Milvus is an adapter
    change plus a config change.
    """

    async def upsert(self, collection: str, points: Sequence[VectorPoint]) -> None: ...

    async def search(
        self,
        collection: str,
        vector: Sequence[float],
        top_k: int,
        filters: FilterExpr | None,
        deadline: Deadline,
    ) -> Sequence[ScoredPoint]: ...

    async def delete(self, collection: str, ids: Sequence[str]) -> None:
        """Remove points by id.

        Must be idempotent. Deletion is driven by right-to-erasure and by tombstoning during
        reindex, and both retry; a second delete of an already-absent id is a success.
        """
        ...

    async def collection_info(self, collection: str) -> CollectionInfo:
        """Collection metadata, including which embedding model wrote its vectors.

        The embedding version is the field that matters. Querying a collection with vectors from
        a different model returns confident nonsense rather than an error, so migration compares
        versions here instead of trusting a naming convention.
        """
        ...


@runtime_checkable
class LexicalStore(Protocol):
    """Sparse, term-based search.

    Kept as its own protocol rather than folded into ``VectorStore`` because the two fail
    differently and degrade differently. Losing lexical search hurts most on identifier-heavy
    queries, where BM25 was doing the real work and dense retrieval will quietly return
    plausible neighbours instead of the exact match.
    """

    async def index(self, index: str, documents: Sequence[dict[str, object]]) -> None: ...

    async def search(
        self,
        index: str,
        query: str,
        top_k: int,
        filters: FilterExpr | None,
        deadline: Deadline,
    ) -> Sequence[ScoredPoint]: ...

    async def delete(self, index: str, ids: Sequence[str]) -> None: ...
