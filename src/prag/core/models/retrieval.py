"""Retrieval plans, their results, and the candidates that come back.

The plan is a first-class object rather than a set of arguments. That is deliberate: a plan can
be logged, diffed against the plan a different router version would have produced, replayed, and
tested without executing anything. Retrieval that is assembled implicitly from scattered
conditionals cannot be reasoned about after the fact.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from prag.core.models.common import VolatilityClass

__all__ = [
    "Candidate",
    "CandidatePool",
    "ChunkMetadata",
    "EvidenceGroup",
    "FusionConfig",
    "FusionMethod",
    "LegResult",
    "LegStatus",
    "PartialResultsPolicy",
    "PlanBudget",
    "RerankConfig",
    "RetrievalLeg",
    "RetrievalPlan",
    "ScoredPoint",
    "VectorPoint",
]

#: A filter expression, kept as a plain nested mapping rather than a vendor query object.
#: Each store adapter translates it into its own dialect; nothing outside a store adapter knows
#: what that dialect looks like. This is the abstraction that lets Qdrant become Milvus.
FilterExpr = dict[str, Any]


class ChunkMetadata(BaseModel):
    """Everything about a chunk that a downstream decision depends on.

    Carried alongside the text rather than looked up later. A rerank, an ACL recheck, a freshness
    score, and an independence correction all need these fields, and re-fetching them per
    candidate would put a database round trip inside a loop that runs fifty times per request.
    """

    model_config = ConfigDict(frozen=True)

    #: Curator-assigned trust in the source, resolved per domain. A source authoritative about
    #: pricing is not automatically authoritative about security policy.
    authority: float = Field(ge=0.0, le=1.0)
    created_at_ms: int
    updated_at_ms: int
    #: Opaque ACL hash. Compared against the principal's set by the independent recheck.
    acl_hash: str
    volatility_class: VolatilityClass = VolatilityClass.SLOW
    #: Identity of the original document this chunk ultimately derives from. Two chunks sharing
    #: a lineage root are not independent evidence, however different their text looks, and the
    #: agreement correction in fusion depends on knowing that.
    lineage_root: str
    title: str | None = None
    language: str | None = None
    doc_type: str | None = None
    #: Which embedding model wrote the vector for this chunk. A mismatch against the query
    #: embedder returns plausible nonsense rather than an error, so migration compares this.
    embedding_version: str | None = None


class VectorPoint(BaseModel):
    """A vector plus its payload, as written to a vector store."""

    model_config = ConfigDict(frozen=True)

    point_id: str
    vector: tuple[float, ...]
    payload: dict[str, Any] = Field(default_factory=dict)


class ScoredPoint(BaseModel):
    """A vector store hit."""

    model_config = ConfigDict(frozen=True)

    point_id: str
    score: float
    payload: dict[str, Any] = Field(default_factory=dict)


class RetrievalLeg(BaseModel):
    """One source queried one way.

    Legs are the unit of parallelism and the unit of failure. ``required`` is the field that
    decides what a failure means: an optional leg dropping produces a coverage warning, a
    required leg dropping is a plan failure. Marking everything required removes the system's
    ability to degrade; marking nothing required removes its ability to notice that it has.
    """

    model_config = ConfigDict(frozen=True)

    leg_id: str
    source_id: str
    #: Which variant this leg queries with. Recorded for tracing and for measuring whether a
    #: transform earns its budget: "rewriting helped" is only checkable if the trace says which
    #: legs used the rewrite.
    query_variant: Literal["raw", "rewritten", "expanded", "entities_only", "sub_query"]
    #: The resolved text to query with. Carried on the leg rather than looked up by the source,
    #: because variant resolution is the planner's job and a source that resolved its own text
    #: would need to know about transforms it has no business knowing about.
    query_text: str = ""
    sub_query_id: str | None = None
    top_k: int = Field(gt=0)
    embedding_model: str | None = None
    filters: FilterExpr = Field(default_factory=dict)
    traversal_depth: int | None = Field(default=None, ge=0)
    timeout_ms: int = Field(gt=0)
    weight: float = Field(default=1.0, ge=0.0)
    required: bool = False


class FusionMethod(StrEnum):
    #: Reciprocal rank fusion. Rank-based, so it needs no score calibration between sources,
    #: which is why it is the default: BM25 scores and cosine similarities are not comparable.
    RRF = "rrf"
    WEIGHTED = "weighted"


class FusionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: FusionMethod = FusionMethod.RRF
    #: RRF's rank damping constant. Larger values flatten the contribution of top ranks.
    k: int = Field(default=60, gt=0)
    weights: dict[str, float] = Field(default_factory=dict)


class RerankConfig(BaseModel):
    """Which rerank tier to use, and what it is allowed to cost.

    Reranking is the single most expensive optional stage on the retrieval path, and the first
    thing the degradation ladder drops. It is configured per SLA tier rather than globally.
    """

    model_config = ConfigDict(frozen=True)

    tier: Literal["none", "light", "standard", "llm"] = "light"
    input_k: int = Field(default=30, gt=0)
    output_k: int = Field(default=8, gt=0)
    timeout_ms: int = Field(default=70, gt=0)


class PlanBudget(BaseModel):
    """What the whole retrieval stage may spend.

    Separate from per-leg timeouts because the two constrain different things: a leg timeout
    bounds one slow source, this bounds the sum. Parallel legs mean the sum is not the total,
    and a plan can still overrun by fanning out too wide.
    """

    model_config = ConfigDict(frozen=True)

    wall_ms: int = Field(gt=0)
    max_candidates: int = Field(default=200, gt=0)
    usd: float | None = Field(default=None, ge=0.0)


class PartialResultsPolicy(StrEnum):
    PROCEED_IF_REQUIRED_LEGS_SUCCEEDED = "proceed_if_required_legs_succeeded"
    ALL_OR_NOTHING = "all_or_nothing"


class RetrievalPlan(BaseModel):
    """A complete, executable description of what to retrieve.

    Built by the planner from the query analysis, then executed by the orchestrator. The
    separation means routing decisions can be evaluated offline against recorded analyses
    without touching an index.
    """

    model_config = ConfigDict(frozen=True)

    plan_id: str
    legs: tuple[RetrievalLeg, ...]
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    budget: PlanBudget
    partial_results_policy: PartialResultsPolicy = (
        PartialResultsPolicy.PROCEED_IF_REQUIRED_LEGS_SUCCEEDED
    )
    fallback_plan_id: str | None = None

    @property
    def required_leg_ids(self) -> frozenset[str]:
        return frozenset(leg.leg_id for leg in self.legs if leg.required)


class Candidate(BaseModel):
    """One retrieved chunk, with the scores each stage attached to it.

    Scores accumulate rather than overwrite: ``raw_score``, then ``fused_score``, then
    ``rerank_score``. Keeping all three is what makes it possible to ask whether the reranker
    actually improved the ordering, which is the only way to know if it is earning its latency.

    ``parent_text`` supports parent-child retrieval: match against a precise child chunk, then
    hand the model the surrounding parent so it has enough context to use the match.
    """

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    chunk_id: str
    document_id: str
    document_version: str
    source_id: str
    text: str
    parent_text: str | None = None

    raw_score: float
    #: Leg id to the rank this candidate held in that leg's results. Rank-based fusion needs the
    #: ranks, and retaining them per leg also shows which source actually found the answer.
    rank_by_leg: dict[str, int] = Field(default_factory=dict)
    fused_score: float | None = None
    rerank_score: float | None = None
    metadata: ChunkMetadata

    @property
    def effective_score(self) -> float:
        """The best score available, latest stage first.

        Callers that just want an ordering should use this rather than reaching for a specific
        stage's score and silently getting ``None`` when that stage was skipped by the
        degradation ladder.
        """
        if self.rerank_score is not None:
            return self.rerank_score
        if self.fused_score is not None:
            return self.fused_score
        return self.raw_score

    @property
    def context_text(self) -> str:
        """What should actually go in the prompt: the parent where one exists."""
        return self.parent_text if self.parent_text is not None else self.text


class LegStatus(StrEnum):
    OK = "ok"
    TIMED_OUT = "timed_out"
    FAILED = "failed"
    #: The breaker was open, so the leg was never attempted. Distinct from a failure: nothing
    #: was tried, and the source's own health is not further implicated.
    SKIPPED_BREAKER_OPEN = "skipped_breaker_open"
    #: Partial results returned within the deadline. Honouring a deadline by truncating is
    #: correct behaviour, not an error, and must be visibly different from both success and
    #: failure so that coverage warnings can be attributed accurately.
    PARTIAL = "partial"


class LegResult(BaseModel):
    """What one leg returned, and how it went."""

    model_config = ConfigDict(frozen=True)

    leg_id: str
    source_id: str
    status: LegStatus
    candidates: tuple[Candidate, ...] = ()
    latency_ms: int = Field(ge=0)
    error_reason_code: str | None = None

    @property
    def usable(self) -> bool:
        return self.status in (LegStatus.OK, LegStatus.PARTIAL)


class CandidatePool(BaseModel):
    """Everything retrieval produced for one request, before evidence processing.

    Holds the per-leg results as well as the merged candidate list, because attributing a bad
    answer to a bad source requires knowing which leg contributed what.
    """

    model_config = ConfigDict(frozen=True)

    plan_id: str
    leg_results: tuple[LegResult, ...] = ()
    candidates: tuple[Candidate, ...] = ()

    @property
    def failed_required_leg_ids(self) -> frozenset[str]:
        return frozenset(r.leg_id for r in self.leg_results if not r.usable)

    @property
    def degraded(self) -> bool:
        """Whether any leg failed to return complete results.

        Drives the coverage warning in the response. A pool that is silently short a source is
        the difference between "here is the answer" and "here is the answer, and I could not
        reach your runbooks".
        """
        return any(r.status is not LegStatus.OK for r in self.leg_results)


class EvidenceGroup(BaseModel):
    """Near-duplicate candidates collapsed into one unit of evidence.

    Grouping happens before ranking and before fusion, because the alternative is a context
    window filled with five copies of the same paragraph and a model that reads their agreement
    as corroboration.

    ``independent`` is the load-bearing field. Two groups sharing a ``lineage_root`` derive from
    the same original document, so their agreement is not evidence of anything: it is one source
    counted twice. Cross-source agreement is only meaningful between independent groups.
    """

    model_config = ConfigDict(frozen=True)

    group_id: str
    members: tuple[Candidate, ...]
    representative: Candidate
    lineage_root: str
    authority: float = Field(ge=0.0, le=1.0)
    #: Freshness relative to the query's estimated half-life, not absolute age.
    freshness: float = Field(ge=0.0, le=1.0)
    independent: bool = True
    #: Stable marker rendered into the prompt and referenced by citations, such as ``E3``.
    citation_marker: str
