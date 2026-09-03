"""Reconciling what the model believes with what the sources say.

Fusion decides four things: which knowledge grounds the answer, what confidence to attach,
whether conflicts exist and how to present them, and whether to answer at all.

It is **not** a scalar sum. A sum lets a high value on one axis mask a disqualifying value on
another — a perfectly fresh, high-agreement, low-authority source set should not override a
well-established parametric fact, and no weighted sum expresses that cleanly. So there is a
score, and then there is a policy table with hard gates the score cannot override.

The single most damaging failure this system can produce is a confident-sounding general answer
to a question about the tenant's own data. That case is a hard gate, not a score contribution.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from prag.core.models.parametric import AdapterRef

__all__ = [
    "Abstention",
    "AbstentionCode",
    "ConfidenceBand",
    "ConfidenceBlock",
    "ConflictEvent",
    "ConflictKind",
    "ConflictPosition",
    "ConflictResolution",
    "KnowledgeBasis",
    "KnowledgeDecision",
    "ParametricSignal",
    "RawConfidenceSignals",
    "StalenessWarning",
    "Stance",
]


class RawConfidenceSignals(BaseModel):
    """Uncalibrated confidence inputs, before the calibration model sees them.

    Raw token logprobs from an instruction-tuned model are badly calibrated, so nothing here is
    used directly as a probability. These are features; the calibrator turns them into one.

    Keeping the raw signals alongside the calibrated output is what makes recalibration
    possible: when the base model or adapter set changes, the calibrator is refit against
    recorded raw signals rather than by re-running production traffic.
    """

    model_config = ConfigDict(frozen=True)

    #: Mean token logprob over the content tokens of a short probe generation.
    mean_logprob: float | None = None
    #: Semantic agreement across n samples at temperature 0.7. Expensive, so it runs only on
    #: the parametric route and only when the budget allows.
    self_consistency: float | None = Field(default=None, ge=0.0, le=1.0)
    #: The model's own stated confidence. Weakly informative, but nearly free.
    verbalized_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    #: Similarity between the query and the active adapter's cluster centroid.
    adapter_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    #: How many samples the self-consistency measure used, so a single-sample estimate is not
    #: mistaken for a three-sample one.
    sample_count: int = Field(default=1, ge=1)


class ParametricSignal(BaseModel):
    """What the weights believe, and how much to trust it."""

    model_config = ConfigDict(frozen=True)

    adapters: tuple[AdapterRef, ...] = ()
    raw_confidence: RawConfidenceSignals
    calibrated_confidence: float = Field(ge=0.0, le=1.0)
    #: Which calibrator produced the number. Recorded because a confidence value is meaningless
    #: without knowing which fit produced it, and calibrators are retrained on their own cadence.
    calibrator_version: str
    #: A short probe answer, used for conflict detection against retrieved evidence. Generating
    #: it is what makes parametric/retrieval contradiction detectable at all — without a stated
    #: parametric position there is nothing to contradict.
    probe_answer: str | None = None


class Stance(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


class ConflictKind(StrEnum):
    """What kind of disagreement was detected.

    ``PARAMETRIC_VS_RETRIEVED`` is the one that feeds back into training: every instance is a
    case where the weights were wrong, which is exactly the labelled data needed both to trigger
    retraining and to improve the router.
    """

    PARAMETRIC_VS_RETRIEVED = "parametric_vs_retrieved"
    SOURCE_VS_SOURCE = "source_vs_source"
    VERSION_VS_VERSION = "version_vs_version"


class ConflictResolution(StrEnum):
    """How a conflict was settled.

    ``SURFACED`` is not a failure to decide. When two positions have comparable authority,
    presenting both with their dates and sources is the correct answer, and silently picking one
    is how a system becomes confidently wrong in a way nobody can audit.
    """

    EVIDENCE_WINS = "evidence_wins"
    AUTHORITY_WINS = "authority_wins"
    SURFACED = "surfaced"
    ABSTAINED = "abstained"


class ConflictPosition(BaseModel):
    """One side of a conflict, with the metadata needed to weigh it."""

    model_config = ConfigDict(frozen=True)

    #: A source id or an adapter id, depending on the conflict kind.
    origin_id: str
    stance: Stance
    authority: float = Field(ge=0.0, le=1.0)
    as_of_ms: int | None = None
    excerpt: str | None = None


class ConflictEvent(BaseModel):
    """A detected contradiction.

    Published to the event bus as well as carried in the response. The aggregate per adapter is
    what drives automated retraining: a conflict rate above threshold marks an adapter stale and
    enqueues its cluster, and above a critical threshold demotes it and falls back to the
    non-parametric path for that cluster.

    This is the loop that closes serving back to training, and the primary automated defence
    against parametric drift.
    """

    model_config = ConfigDict(frozen=True)

    conflict_id: str
    kind: ConflictKind
    claim: str
    positions: tuple[ConflictPosition, ...]
    resolution: ConflictResolution
    #: Populated for parametric conflicts. Drives the per-adapter staleness signal.
    adapter_ids: tuple[str, ...] = ()

    @property
    def authority_delta(self) -> float:
        """Spread between the strongest and weakest position's authority.

        Small means comparable authority, which is the condition for surfacing rather than
        picking. The threshold itself is configuration, not a constant here.
        """
        if not self.positions:
            return 0.0
        authorities = [p.authority for p in self.positions]
        return max(authorities) - min(authorities)


class AbstentionCode(StrEnum):
    """Machine-readable reason for declining to answer.

    Every abstention carries one. An abstention that does not tell the user why is a failure of
    the abstention path, not a use of it — and a distribution of these codes is what makes an
    abstention rate diagnosable rather than merely alarming.
    """

    KNOWLEDGE_BELOW_FLOOR = "knowledge_below_floor"
    #: The query targets private or tenant-specific knowledge and retrieval found nothing usable.
    #: The hard gate that exists to prevent the system's worst failure.
    PRIVATE_QUERY_NO_EVIDENCE = "private_query_no_evidence"
    RETRIEVAL_TOTAL_FAILURE = "retrieval_total_failure"
    CONTEXT_VALIDATION_FAILED = "context_validation_failed"
    IRRECONCILABLE_CONFLICT = "irreconcilable_conflict"
    EVIDENCE_TOO_STALE = "evidence_too_stale"
    GUARDRAIL_BLOCKED = "guardrail_blocked"
    BUDGET_EXCEEDED = "budget_exceeded"
    AMBIGUOUS_NEEDS_CLARIFICATION = "ambiguous_needs_clarification"


class Abstention(BaseModel):
    """A declined answer.

    A success state, reported as such in metrics. ``suggested_action`` is what separates a useful
    abstention from a dead end: rephrase, broaden the scope, request access, or contact the
    source owner.
    """

    model_config = ConfigDict(frozen=True)

    reason_code: AbstentionCode
    explanation: str
    suggested_action: str | None = None


class KnowledgeBasis(StrEnum):
    """What the answer is grounded in."""

    RETRIEVED_EVIDENCE = "retrieved_evidence"
    PARAMETRIC = "parametric"
    HYBRID = "hybrid"
    ABSTAIN = "abstain"


class ConfidenceBand(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ConfidenceBlock(BaseModel):
    """Structured confidence, always present in the envelope.

    Prose hedging in the answer text is generated *from* this structure rather than
    independently, so the words and the numbers cannot disagree. A model left to hedge on its
    own will say "I am fairly confident" next to a score of 0.31.
    """

    model_config = ConfigDict(frozen=True)

    score: float = Field(ge=0.0, le=1.0)
    band: ConfidenceBand
    basis: KnowledgeBasis


class StalenessWarning(BaseModel):
    """The evidence is old relative to how fast this knowledge changes.

    Both numbers are reported, because neither means anything alone: the age of the oldest
    evidence is only interpretable against the query's estimated half-life.
    """

    model_config = ConfigDict(frozen=True)

    oldest_evidence_ms: int
    half_life_days: float = Field(gt=0.0)
    #: Age divided by half-life. Above 1.0 the evidence has had time to become wrong.
    staleness_ratio: float = Field(ge=0.0)


class KnowledgeDecision(BaseModel):
    """Fusion's verdict for one request."""

    model_config = ConfigDict(frozen=True)

    basis: KnowledgeBasis
    knowledge_score: float
    #: Calibrated parametric confidence, as it entered the decision.
    p_parametric: float = Field(ge=0.0, le=1.0)
    #: Calibrated retrieval confidence.
    p_retrieval: float = Field(ge=0.0, le=1.0)
    #: Cross-source agreement *after* the independence correction. Three documents quoting the
    #: same press release count once. Skipping this correction systematically inflates confidence
    #: exactly when the system is most wrong, which is the quiet failure mode this field exists
    #: to prevent.
    agreement_independent: float = Field(ge=0.0, le=1.0)
    authority_max: float = Field(ge=0.0, le=1.0)
    conflicts: tuple[ConflictEvent, ...] = ()
    abstention: Abstention | None = None
    #: Rendered into the prompt so the model states its epistemic position rather than inventing
    #: one, for example "from general knowledge, not from your documents". Composed by the
    #: platform, never by the model.
    epistemic_marking: str | None = None

    @property
    def abstained(self) -> bool:
        return self.basis is KnowledgeBasis.ABSTAIN

    @property
    def requires_citations(self) -> bool:
        """Whether this answer must carry citations.

        A parametric answer cannot cite directly, but provenance shadowing may attach citations
        to the claims a lexical lookup can actually support. So the parametric basis does not
        exempt an answer from citation, it changes how citations are obtained.
        """
        return self.basis in (KnowledgeBasis.RETRIEVED_EVIDENCE, KnowledgeBasis.HYBRID)

    @property
    def unresolved_conflicts(self) -> tuple[ConflictEvent, ...]:
        return tuple(c for c in self.conflicts if c.resolution is ConflictResolution.SURFACED)


#: The five decision-policy weights. Configurable per tenant and per domain; these are the
#: shipped defaults, and they live here so the policy table has one place to read them from.
DecisionWeightKey = Literal["w1", "w2", "w3", "w4", "w5"]

DEFAULT_DECISION_WEIGHTS: dict[DecisionWeightKey, float] = {
    "w1": 0.40,  # retrieval confidence, scaled by authority and freshness
    "w2": 0.20,  # parametric confidence, scaled by its fixed authority and freshness
    "w3": 0.20,  # independence-corrected cross-source agreement
    "w4": 0.15,  # conflict penalty
    "w5": 0.05,  # coverage gap penalty
}

#: Parametric knowledge is assigned a fixed, deliberately modest authority. It cannot be
#: audited per claim, so it does not get to outrank a curated source on authority alone.
DEFAULT_PARAMETRIC_AUTHORITY = 0.55
