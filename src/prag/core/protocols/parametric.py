"""Parametric tier protocols: selection, storage, and the eligibility gate.

The tenant scope filter in ``AdapterSelector.select`` is the strictest constraint in the
platform. Every other isolation layer can be rechecked downstream; this one cannot, because
once a delta is merged into the serving weights there is no per-request filter that can take it
back out.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from prag.core.models.identity import Principal
from prag.core.models.parametric import (
    AdapterSet,
    EligibilityResult,
    KnowledgeRecord,
    LoadedAdapter,
    ParametricEconomics,
)
from prag.core.models.query import QueryAnalysis

__all__ = ["AdapterSelector", "AdapterStore", "ParametricEligibilityGate"]


@runtime_checkable
class AdapterSelector(Protocol):
    """Chooses which adapters, if any, serve a request.

    Selection is retrieval over adapter descriptors rather than over documents: each record
    carries a centroid embedding of its training cluster, and selection is vector similarity
    against those centroids.
    """

    async def select(
        self,
        analysis: QueryAnalysis,
        principal: Principal,
        max_adapters: int,
    ) -> AdapterSet:
        """Select adapters for this query.

        **Hard-filter by tenant scope before any scoring.** Not as a ranking penalty, not as a
        tie-breaker — as a filter applied before similarity is computed at all. A tenant-exclusive
        adapter may only load for requests carrying that tenant id. This is the one failure in
        the system that cannot be walked back after the fact.

        **Return an empty set rather than a low-coverage match.** An adapter that half-covers
        the query contributes confident noise, and confident noise is worse than the absence it
        replaced: the retrieval path would have produced something citable.

        Respect ``max_adapters``. Composition interference is real, and naively summing several
        low-rank deltas degrades all of them.
        """
        ...


@runtime_checkable
class AdapterStore(Protocol):
    """Residency management for adapter weights.

    Backed by a hot LRU cache over object storage. Cold load costs 50 to 200 ms, which is why
    residency is a question callers can ask rather than something they discover by waiting.
    """

    async def load(self, adapter_id: str, version: str) -> LoadedAdapter:
        """Make an adapter resident and ready to serve.

        Must verify the checksum. A corrupted delta does not error at inference; it degrades
        output in ways that look like a bad prompt, and tracking that back to a truncated
        object-store read costs days.
        """
        ...

    async def is_resident(self, adapter_id: str, version: str) -> bool:
        """Whether this adapter is already in GPU memory.

        Consulted by the selector when the budget is tight: a resident adapter is nearly free,
        a cold one may not fit in what remains of a time-to-first-token budget.
        """
        ...

    async def evict(self, adapter_id: str, version: str) -> None:
        """Remove an adapter from residency.

        Must be immediate and must be callable on revocation. When a source document is erased,
        every adapter containing it is revoked and evicted; waiting for LRU pressure would mean
        continuing to serve knowledge that was legally required to disappear.
        """
        ...


class ParametricEligibilityGate(Protocol):
    """Decides whether knowledge may become parametric.

    Not ``runtime_checkable``: it is a pure function with no I/O, so it is verified by its
    conformance suite rather than by an isinstance check at wiring time.
    """

    def evaluate(
        self, record: KnowledgeRecord, economics: ParametricEconomics
    ) -> EligibilityResult:
        """Evaluate the gate. Pure function.

        Pure because it runs in two places — at ingestion and at every retraining cycle — and
        the two must agree. A gate that consults live state could admit knowledge on Tuesday
        that it would have refused on Monday, with no record of why.

        **Must return every blocking reason, not just the first.** A curator who fixes one
        blocker only to hit the next has learned nothing about whether the knowledge is
        fundamentally ineligible. The question being asked is "can this ever be parametric",
        and it deserves a complete answer in one pass.

        Hard blockers are evaluated before economics, and no economic argument overrides one.
        Knowledge that cannot be filtered per request, cannot be revoked in time, or must be
        quoted verbatim is ineligible at any price.
        """
        ...
