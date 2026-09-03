"""Cross-cutting protocols: guardrails, memory, caching, evaluation, and graph nodes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from prag.core.errors import Severity
from prag.core.models.common import GuardrailPhase, MemoryNamespace
from prag.core.models.events import (
    CacheEntry,
    CacheKey,
    EvalInputs,
    EvalSample,
    MetricResult,
)
from prag.core.models.guardrails import GuardrailPayload, GuardrailVerdict
from prag.core.models.identity import Principal
from prag.core.models.memory import MemoryItem, MemorySelector, SessionSummary
from prag.core.models.state import NodeResult, RequestState

__all__ = ["CacheTier", "Evaluator", "GraphNode", "Guardrail", "MemoryStore"]


@runtime_checkable
class Guardrail(Protocol):
    """One check in the guardrail chain."""

    name: str
    phase: GuardrailPhase
    severity: Severity

    async def check(self, payload: GuardrailPayload) -> GuardrailVerdict:
        """Inspect the payload and return ALLOW, MODIFY, or BLOCK.

        **Side-effect free apart from emitting security events.** A guardrail that mutates
        shared state cannot be run twice, cannot be reordered, and cannot be tested in
        isolation — and the chain does all three. MODIFY expresses the change by returning a
        replacement payload, never by editing the one it was handed.

        MODIFY exists because the useful response to detected PII is usually redaction rather
        than refusal. Forcing that into a binary would make the chain either useless or
        unusable.
        """
        ...


@runtime_checkable
class MemoryStore(Protocol):
    """One memory tier: session, or long-term."""

    namespace: MemoryNamespace

    async def read(self, principal: Principal, query: str, limit: int) -> Sequence[MemoryItem]: ...

    async def write(self, principal: Principal, item: MemoryItem) -> None:
        """Persist an item.

        **Must reject items whose provenance is ``MODEL_GENERATED`` for the long-term
        namespace.** Enforced at the store rather than trusted from the caller. Letting a
        model's own output persist as a user fact is how a system starts confidently believing
        things nobody ever told it — and unlike a bad answer, that belief survives every
        subsequent session.
        """
        ...

    async def summarize(self, principal: Principal, session_id: str) -> SessionSummary:
        """Roll up a session, keeping entities and decisions verbatim.

        Verbatim because those are exactly what coreference resolution must be exact about, and
        exactly what a prose summary is most likely to blur.
        """
        ...

    async def forget(self, principal: Principal, selector: MemorySelector) -> int:
        """Delete matching items and return how many were removed.

        Returns a count because right-to-erasure requires an auditable record of what was
        actually erased. An empty selector must match nothing rather than everything.
        """
        ...


@runtime_checkable
class CacheTier(Protocol):
    """One cache tier: exact, semantic, embedding, retrieval, rerank, or analysis."""

    tier: str

    async def get(self, key: CacheKey) -> CacheEntry | None: ...

    async def set(self, key: CacheKey, entry: CacheEntry, ttl_s: int) -> None:
        """Store an entry.

        Nothing that failed validation, abstained, or carries a warning may be cached. That
        rule is enforced in key construction rather than left to callers, because a caching
        rule depending on every call site remembering it will be broken once and then stay
        broken invisibly — serving a stale warning-free copy of an answer that was never
        trustworthy.
        """
        ...

    async def invalidate_by_document(self, document_ids: Sequence[str]) -> int:
        """Drop every entry derived from these documents, returning the count.

        Backed by a document-to-key reverse index. Without one, a document update can only be
        handled by flushing the tier, and a system that flushes on every ingest has no cache.
        """
        ...


@runtime_checkable
class Evaluator(Protocol):
    """One metric, with one implementation.

    Deliberately one implementation shared by all three triggers — offline runs, the CI
    regression gate, and online sampling. Separate implementations drift, and once they have
    drifted the offline numbers stop predicting the online ones, which removes the only reason
    to run the offline suite at all.
    """

    metric_id: str
    #: What this metric needs in order to score. Declared so runners can batch fetches for a
    #: whole dataset rather than pulling per sample.
    requires: EvalInputs

    async def score(self, sample: EvalSample) -> MetricResult: ...


@runtime_checkable
class GraphNode(Protocol):
    """One step in the request graph.

    ``reads`` and ``writes`` are declarations the interpreter enforces, which is what makes the
    graph statically checkable: a node reading ``evidence`` cannot be scheduled before the node
    writing it, and that is caught at graph-load time rather than as a ``None`` dereference in
    production.
    """

    node_id: str
    reads: frozenset[str]
    writes: frozenset[str]
    timeout_ms: int
    #: Where to go when this node fails. ``None`` means the failure propagates, which is correct
    #: only for nodes whose absence makes the request meaningless.
    fallback_node: str | None

    async def run(self, state: RequestState) -> NodeResult:
        """Execute against the state and return the next one.

        Must not mutate the state it receives; return a new one via ``state.advanced(...)``.
        Mutation would make every recorded state a snapshot of the request's end rather than of
        this step, and recorded-state replay is the platform's regression mechanism.

        Must write only what ``writes`` declares. The interpreter checks, so a node that
        quietly sets an undeclared field fails its own test rather than someone else's.
        """
        ...
