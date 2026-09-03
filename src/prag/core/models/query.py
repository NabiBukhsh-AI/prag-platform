"""The structured query representation and the strategy decision built from it.

Produced by the query intelligence layer in roughly 15 to 20 ms by a multi-head encoder
classifier. Not by an LLM: a 300 ms model call cannot sit on the critical path of a 650 ms
time-to-first-token budget, and an LLM classifier is the low-confidence fallback only.

Two properties of this schema carry real weight downstream:

**Confidence is per field, not global.** One scalar for the whole analysis cannot express the
common case where intent is obvious and domain is a coin flip. Routing needs to know which
specific head was unsure, because that determines whether hedged execution is worth its cost.

**Temporality is a half-life estimate, not a boolean.** Absolute document age means nothing on
its own. A two-year-old constitutional provision is fresh; a two-day-old stock price is stale.
Freshness scoring divides by this number, so a boolean here would flatten the distinction the
whole fusion layer depends on.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from prag.core.models.common import SlaTier, VolatilityClass

__all__ = [
    "Ambiguity",
    "BudgetClass",
    "Entity",
    "FieldPrediction",
    "KnowledgeRequirement",
    "PolicyRisk",
    "QueryAnalysis",
    "QueryQuality",
    "QueryStructure",
    "QueryVariants",
    "SafetyPreflags",
    "SessionContext",
    "SourceHint",
    "Strategy",
    "StrategyDecision",
    "SubQuery",
    "Temporality",
]


class FieldPrediction(BaseModel):
    """One classifier head's output.

    ``runner_up`` is retained because it is what makes a low-confidence prediction actionable.
    Knowing the model said ``internal_operations`` at 0.51 is much less useful than also knowing
    its second choice was ``security`` at 0.47, which is the signal that a hedged retrieval plan
    should cover both domains.
    """

    model_config = ConfigDict(frozen=True)

    value: Any
    confidence: float = Field(ge=0.0, le=1.0)
    runner_up: Any | None = None

    def is_confident(self, threshold: float) -> bool:
        return self.confidence >= threshold


class KnowledgeRequirement(StrEnum):
    """The five knowledge questions whose answers can eliminate a strategy outright.

    Each is a hard constraint rather than a scored preference. ``REQUIRES_PRIVATE_DATA`` and
    ``REQUIRES_CITATION`` together forbid the parametric route regardless of what the utility
    scores say, because parameters cannot be filtered per request and cannot be cited.
    """

    REQUIRES_EXTERNAL_KNOWLEDGE = "requires_external_knowledge"
    REQUIRES_LIVE_DATA = "requires_live_data"
    REQUIRES_PRIVATE_DATA = "requires_private_data"
    REQUIRES_EXACT_QUOTATION = "requires_exact_quotation"
    REQUIRES_CITATION = "requires_citation"


class Temporality(BaseModel):
    """How fast the answer to this query changes."""

    model_config = ConfigDict(frozen=True)

    volatility_class: VolatilityClass
    estimated_half_life_days: float = Field(gt=0.0)
    #: A time reference the user stated explicitly, such as "in Q3 2025". Overrides the estimate.
    explicit_time_reference: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class Entity(BaseModel):
    """A recognised entity, resolved to a canonical identifier where possible.

    ``canonical_id`` is what makes graph traversal and alias expansion work. Without it,
    "sev-1", "SEV1", and "severity one" are three unrelated strings.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    type: str
    canonical_id: str | None = None


class SubQuery(BaseModel):
    """One node of a decomposed multi-hop query.

    ``depends_on`` makes the decomposition a DAG rather than a list, which is the point:
    independent hops execute in parallel, and that is a meaningful latency win on exactly the
    queries that are otherwise slowest.
    """

    model_config = ConfigDict(frozen=True)

    sub_query_id: str
    text: str
    depends_on: tuple[str, ...] = ()


class QueryStructure(BaseModel):
    """Shape of the query: whether it needs multiple hops, and what it refers to."""

    model_config = ConfigDict(frozen=True)

    multi_hop: FieldPrediction
    sub_queries: tuple[SubQuery, ...] = ()
    entities: tuple[Entity, ...] = ()
    constraints: tuple[str, ...] = ()


class Ambiguity(BaseModel):
    """Whether the query needs clarifying before it can be answered well.

    Drives the clarification loop, which is one of the three cycles the graph genuinely needs
    and a reason a pure DAG was rejected as the orchestration model.
    """

    model_config = ConfigDict(frozen=True)

    is_ambiguous: bool = False
    ambiguity_type: str | None = None
    clarification_candidates: tuple[str, ...] = ()


class SourceHint(BaseModel):
    """A prior belief that one source is worth querying for this query.

    Priors, not decisions. The planner combines them with source health, budget, and
    capabilities; a high prior on an unavailable source produces no leg.
    """

    model_config = ConfigDict(frozen=True)

    source_id: str
    prior: float = Field(ge=0.0, le=1.0)


class QueryQuality(BaseModel):
    """Whether the query as written is good enough to retrieve against."""

    model_config = ConfigDict(frozen=True)

    needs_rewrite: bool
    score: float = Field(ge=0.0, le=1.0)


class BudgetClass(BaseModel):
    """The latency and cost tiers this query should be served under.

    Distinct from the principal's SLA tier: a batch-tier caller can still send a query whose
    answer is worthless if it arrives late, and an interactive caller can send one they are
    willing to wait for.
    """

    model_config = ConfigDict(frozen=True)

    latency_tier: SlaTier
    cost_tier: SlaTier


class PolicyRisk(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SafetyPreflags(BaseModel):
    """Cheap safety signals from the classifier, ahead of the full guardrail chain.

    Pre-flags, not verdicts. They let the planner avoid expensive work on a query that is about
    to be blocked; they never substitute for the guardrail chain, which always runs.
    """

    model_config = ConfigDict(frozen=True)

    injection_suspected: bool = False
    pii_detected: bool = False
    policy_risk: PolicyRisk = PolicyRisk.NONE


class SessionContext(BaseModel):
    """What the conversation has established so far.

    Carries a rolling summary rather than full history, and keeps entities and decisions
    verbatim because those are what coreference resolution and follow-up handling need to be
    exact about.
    """

    model_config = ConfigDict(frozen=True)

    session_id: str
    turn_index: int = Field(ge=0)
    summary: str | None = None
    entities: tuple[Entity, ...] = ()
    decisions: tuple[str, ...] = ()
    summary_hash: str | None = None


class QueryAnalysis(BaseModel):
    """The full structured representation of one query.

    Forward-compatible across schema versions: unknown fields are preserved rather than
    rejected, so a rolling deploy where two versions run side by side does not drop data written
    by the newer one.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    schema_version: Literal["qa.v2"] = "qa.v2"
    request_id: str
    raw_query: str
    normalized_query: str
    language: str

    intent: FieldPrediction
    domain: FieldPrediction
    complexity: FieldPrediction
    knowledge_requirements: dict[KnowledgeRequirement, FieldPrediction]

    temporality: Temporality
    structure: QueryStructure
    ambiguity: Ambiguity
    source_hints: tuple[SourceHint, ...] = ()
    query_quality: QueryQuality
    budget_class: BudgetClass
    safety_preflags: SafetyPreflags

    #: Aggregate calibrated uncertainty across the heads that feed strategy selection. Above the
    #: configured threshold the orchestrator hedges: retrieval runs even on a parametric lean,
    #: so the more expensive path is available if fusion turns out to need it.
    router_uncertainty: float = Field(ge=0.0, le=1.0)
    classifier_tier_used: Literal["T0", "T1", "T2"]
    analysis_latency_ms: int = Field(ge=0)

    def requires(self, requirement: KnowledgeRequirement) -> bool:
        """Whether a knowledge requirement holds.

        Absent means false: a head that did not run cannot assert a requirement. The inverse
        default would be safer in isolation but would eliminate every strategy on a partial
        analysis, turning a degraded classifier into a total outage.
        """
        prediction = self.knowledge_requirements.get(requirement)
        return bool(prediction and prediction.value)

    def confidence_in(self, requirement: KnowledgeRequirement) -> float:
        prediction = self.knowledge_requirements.get(requirement)
        return prediction.confidence if prediction else 0.0


class Strategy(StrEnum):
    """Which class of knowledge answers this query."""

    PARAMETRIC = "PARAMETRIC"
    NON_PARAMETRIC = "NON_PARAMETRIC"
    HYBRID = "HYBRID"


class StrategyDecision(BaseModel):
    """The routing outcome, with enough detail to explain itself without being re-run.

    ``eliminated`` is the field that makes routing debuggable. A trace showing only the winner
    tells you nothing about whether the router considered the option you expected; one showing
    that ``PARAMETRIC`` was eliminated because ``requires_exact_quotation`` held answers the
    question immediately.
    """

    model_config = ConfigDict(frozen=True)

    strategy: Strategy
    utility_scores: dict[Strategy, float] = Field(default_factory=dict)
    #: Strategy to the hard-constraint reason it was eliminated, before any scoring happened.
    eliminated: dict[Strategy, str] = Field(default_factory=dict)
    #: Retrieval runs despite a parametric lean, because router uncertainty was high.
    hedged: bool = False
    speculative_retrieval_started: bool = False
    #: Set when this request is part of the exploration fraction routed against the argmax to
    #: keep the strategy quality table honest. These are the counterfactual regret sample.
    exploration: bool = False


class QueryVariants(BaseModel):
    """The outputs of the four query transforms.

    Each is independently gated and independently budgeted, so a variant being absent means its
    transform did not apply or did not fit its budget, not that it failed.
    """

    model_config = ConfigDict(frozen=True)

    raw: str
    rewritten: str | None = None
    coreference_resolved: str | None = None
    expanded: str | None = None
    entities_only: str | None = None
    sub_queries: tuple[SubQuery, ...] = ()

    def for_variant(self, variant: str) -> str | None:
        """Look up a variant by the name a retrieval leg uses to request it."""
        return {
            "raw": self.raw,
            "rewritten": self.rewritten,
            "coreference_resolved": self.coreference_resolved,
            "expanded": self.expanded,
            "entities_only": self.entities_only,
        }.get(variant)
