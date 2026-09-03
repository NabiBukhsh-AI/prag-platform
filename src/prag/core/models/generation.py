"""Generation requests, results, and the response envelope.

The envelope is the platform's public contract. It is versioned, and it always carries
structured confidence, grounding counts, and any conflicts — never prose hedging alone. A client
that wants to render "I am not sure about this" can derive it; a client that wants to gate on a
threshold can do that too. Neither is possible if the uncertainty exists only as adjectives in
the answer text.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from prag.core.models.context import CoverageWarning, RenderedRegion
from prag.core.models.fusion import (
    Abstention,
    ConfidenceBlock,
    ConflictEvent,
    StalenessWarning,
)
from prag.core.models.parametric import AdapterRef

__all__ = [
    "AnswerEnvelope",
    "Citation",
    "ClaimVerdict",
    "Diagnostics",
    "FinishReason",
    "GenerationChunk",
    "GenerationRequest",
    "GenerationResult",
    "GroundingReport",
    "ModelSpec",
    "TokenUsage",
    "ToolSchema",
]


class ModelSpec(BaseModel):
    """Which model, which version, which adapters, which profile.

    Fully resolved before generation starts. Costs are carried here rather than looked up at
    billing time so that per-request cost attribution uses the same numbers the router used to
    choose, which is the only way cost-aware routing can be checked against reality.
    """

    model_config = ConfigDict(frozen=True)

    model_id: str
    model_version: str
    provider_id: str
    adapters: tuple[AdapterRef, ...] = ()
    #: Names a generation profile in configuration rather than inlining sampling parameters, so
    #: a profile change is a config change and not a deploy.
    profile: str
    context_window: int = Field(gt=0)
    supports_tools: bool = False
    supports_structured_output: bool = False
    cost_per_1k_in: float = Field(ge=0.0)
    cost_per_1k_out: float = Field(ge=0.0)

    def estimated_cost_usd(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.cost_per_1k_in + tokens_out * self.cost_per_1k_out) / 1000.0


class ToolSchema(BaseModel):
    """A tool the model may call.

    ``requires_provenance`` is the provenance gate. A tool marked this way can only be invoked
    when the authorising instruction came from a region that carries instruction authority —
    never from retrieved evidence. This is the defence that holds after the model has already
    been fooled by a document-borne injection, which is why it is a property of the tool rather
    than a check somewhere in the executor.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    requires_provenance: bool = True
    #: Side-effecting tools are held to a stricter standard than read-only ones, because a
    #: wrongly authorised read is a leak and a wrongly authorised write is a change.
    has_side_effects: bool = False


class GenerationRequest(BaseModel):
    """Everything a provider needs for one generation.

    Carries rendered regions rather than a prompt string. The provider assembles the final
    payload in whatever shape its API wants, and the region boundaries — including which region
    may authorise a tool call — survive all the way to the call site.
    """

    model_config = ConfigDict(frozen=True)

    request_id: str
    spec: ModelSpec
    regions: tuple[RenderedRegion, ...]
    tools: tuple[ToolSchema, ...] = ()
    profile_overrides: dict[str, Any] = Field(default_factory=dict)
    stream: bool = True


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALL = "tool_call"
    CONTENT_FILTER = "content_filter"
    #: The stream was cut because a guardrail fired mid-generation. Distinct from a filter
    #: refusal by the provider: this one is ours, and it is attributable to a named guardrail.
    GUARDRAIL = "guardrail"
    ERROR = "error"


class TokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    #: Prompt tokens served from a provider-side cache. Worth tracking separately because it is
    #: the difference between the cost the router predicted and the cost actually incurred.
    cached_tokens_in: int = Field(default=0, ge=0)


class GenerationChunk(BaseModel):
    """One streamed increment.

    The first chunk carries ``ttft_ms``, because time to first token is the latency number that
    actually describes the user's experience and it cannot be reconstructed after the fact.
    """

    model_config = ConfigDict(frozen=True)

    text: str = ""
    index: int = Field(default=0, ge=0)
    ttft_ms: int | None = Field(default=None, ge=0)
    finish_reason: FinishReason | None = None
    usage: TokenUsage | None = None


class GenerationResult(BaseModel):
    """A completed non-streaming generation."""

    model_config = ConfigDict(frozen=True)

    text: str
    finish_reason: FinishReason
    usage: TokenUsage
    ttft_ms: int | None = Field(default=None, ge=0)
    total_ms: int = Field(ge=0)
    model_id: str
    model_version: str
    provider_id: str


class Citation(BaseModel):
    """A claim's link to the evidence that supports it.

    ``entailment_score`` is what makes this a citation rather than a decoration. Citations below
    the configured threshold are stripped and their claims marked unsourced, because attaching a
    plausible-looking citation to a claim the source does not actually support is worse than
    attaching none: it converts an unsupported statement into an apparently verified one.
    """

    model_config = ConfigDict(frozen=True)

    marker: str
    group_id: str
    document_id: str
    document_version: str
    source_id: str
    #: Character span within the cited text, where the verifier could localise it.
    span: tuple[int, int] | None = None
    entailment_score: float = Field(ge=0.0, le=1.0)


class ClaimVerdict(BaseModel):
    """One extracted claim and whether the evidence backs it."""

    model_config = ConfigDict(frozen=True)

    claim: str
    entailed: bool
    entailment_score: float = Field(ge=0.0, le=1.0)
    cited_group_ids: tuple[str, ...] = ()


class GroundingReport(BaseModel):
    """Whether the answer says only what its evidence supports.

    Counts rather than a pass/fail flag. "Six of seven claims cited" is actionable in a way that
    "grounding: failed" is not, and the ratio is what the faithfulness metric is computed from.
    """

    model_config = ConfigDict(frozen=True)

    claims_total: int = Field(ge=0)
    claims_cited: int = Field(ge=0)
    claims_unsourced: int = Field(ge=0)
    verdicts: tuple[ClaimVerdict, ...] = ()
    #: Set when the answer was regenerated once with a tightened prompt after a grounding
    #: failure. One retry only; a second would be an unbounded loop on a bad evidence set.
    regenerated: bool = False

    @property
    def groundedness(self) -> float:
        """Fraction of claims backed by evidence.

        An answer with no factual claims is fully grounded by definition — a formatting request
        or a refusal has nothing to be unfaithful about, and scoring it zero would drag the
        aggregate metric down for the wrong reason.
        """
        if self.claims_total == 0:
            return 1.0
        return self.claims_cited / self.claims_total


class Diagnostics(BaseModel):
    """What actually happened, for the client and for the trace.

    Returned to callers with sufficient scope rather than kept internal. A client that can see
    which route served its request, how degraded it was, and what it cost can diagnose its own
    latency complaints without opening a support ticket.
    """

    model_config = ConfigDict(frozen=True)

    route_class: str
    strategy: str
    model_id: str
    model_version: str
    adapters: tuple[AdapterRef, ...] = ()
    classifier_tier_used: str | None = None

    ttft_ms: int | None = Field(default=None, ge=0)
    total_ms: int = Field(ge=0)
    usage: TokenUsage | None = None
    usd_cost: float = Field(default=0.0, ge=0.0)

    cache_hit: bool = False
    cache_tier: str | None = None
    #: Which rung of the degradation ladder this request finished on. Zero means undegraded.
    #: Non-zero here explains a quality dip that would otherwise look like an unexplained
    #: regression in evaluation.
    degradation_level: int = Field(default=0, ge=0, le=6)
    rerank_skipped_reason: str | None = None
    node_timings_ms: dict[str, int] = Field(default_factory=dict)


class AnswerEnvelope(BaseModel):
    """The response.

    Forward-compatible: unknown fields are preserved rather than rejected, so a client written
    against v2 keeps working when a v3 field appears, and a rolling deploy does not drop data
    written by whichever version happens to be newer.

    An abstention is a valid, successful envelope. It carries an ``abstention`` with a reason
    code and an empty ``answer`` — not an HTTP error, because declining to answer is an outcome
    of the request rather than a failure to process it.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    schema_version: Literal["answer_envelope.v2"] = "answer_envelope.v2"
    request_id: str
    answer: str
    citations: tuple[Citation, ...] = ()
    confidence: ConfidenceBlock
    grounding: GroundingReport
    conflicts: tuple[ConflictEvent, ...] = ()
    coverage_warning: CoverageWarning | None = None
    staleness_warning: StalenessWarning | None = None
    abstention: Abstention | None = None
    diagnostics: Diagnostics

    @property
    def abstained(self) -> bool:
        return self.abstention is not None

    @property
    def has_warnings(self) -> bool:
        """Whether anything qualifies this answer.

        The set of conditions that must never be cached, which is why it is one property rather
        than three checks at each cache call site.
        """
        return bool(
            self.coverage_warning or self.staleness_warning or self.abstention or self.conflicts
        )

    @property
    def cacheable(self) -> bool:
        """Whether this response may enter a cache.

        Nothing that failed validation, abstained, or carries a warning is cacheable. Enforced
        here and in key construction rather than left to callers, because a caching rule that
        depends on every call site remembering it is a rule that will be broken once and then
        stay broken invisibly.
        """
        return not self.has_warnings and self.grounding.claims_unsourced == 0
