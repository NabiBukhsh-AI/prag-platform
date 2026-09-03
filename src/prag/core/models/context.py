"""The assembled prompt, as a structured object rather than a string.

Context is built as regions with explicit token allocations, not concatenated. The reason is
security as much as budgeting: retrieved text must occupy a structurally isolated region that
carries no instruction authority. A string built by concatenation cannot express "this part is
data", and a model given no structural signal will happily follow instructions it finds in a
document.

The bundle deliberately does *not* hold the rendered prompt. It holds a hash of it. Prompts
contain evidence text and query text, both of which have short retention windows, and a model
whose lifetime is the request would otherwise pin them into every trace and cache entry.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from prag.core.models.memory import MemoryItem
from prag.core.models.retrieval import EvidenceGroup

__all__ = [
    "CompressionStage",
    "ContextBundle",
    "ContextRegion",
    "ContextValidation",
    "Contradiction",
    "CoverageWarning",
    "OrderingMode",
    "RegionName",
    "RenderedRegion",
    "ValidationAction",
]


class RegionName(StrEnum):
    """The six prompt regions, in assembly order.

    ``EVIDENCE`` is the untrusted one. Everything about how it is delimited, scanned, and
    ordered follows from that, and it is the only region whose content the platform did not
    author.
    """

    SYSTEM = "SYSTEM"
    TOOLS = "TOOLS"
    MEMORY = "MEMORY"
    EVIDENCE = "EVIDENCE"
    QUERY = "QUERY"
    OUTPUT = "OUTPUT"


class ContextRegion(BaseModel):
    """One region's token allocation and actual usage.

    ``trimmable`` is what the compression ladder consults. The system prompt and the query are
    not trimmable at any degradation level: a request that cannot afford the user's own question
    has no useful degraded form, and should abstain instead.
    """

    model_config = ConfigDict(frozen=True)

    name: RegionName
    allocated_tokens: int = Field(ge=0)
    used_tokens: int = Field(ge=0)
    trimmable: bool

    @property
    def overflowed(self) -> bool:
        return self.used_tokens > self.allocated_tokens

    @property
    def headroom_tokens(self) -> int:
        return max(0, self.allocated_tokens - self.used_tokens)


class OrderingMode(StrEnum):
    """How evidence is ordered within its region.

    Ordering is not cosmetic. Attention is not uniform across a long context, so the same
    evidence in a different order produces measurably different answers. ``EDGE_WEIGHTED`` puts
    the strongest evidence at the start and end, where recall is best, and the weakest in the
    middle, where it is worst.
    """

    EDGE_WEIGHTED = "edge_weighted"
    DESCENDING = "descending"
    CHRONOLOGICAL = "chronological"
    SOURCE_GROUPED = "source_grouped"


class CompressionStage(StrEnum):
    """The compression ladder, cheapest and safest first.

    ``ABSTRACTIVE`` is last and off by default, because a small model summarising evidence can
    fabricate, and a fabrication introduced during compression is indistinguishable downstream
    from one the answer model invented. It is never applied to numerals, dates, dosages, money,
    identifiers, code, or quoted text.
    """

    DROP = "drop"
    PARENT_NARROW = "parent_narrow"
    EXTRACTIVE = "extractive"
    HIERARCHICAL = "hierarchical"
    ABSTRACTIVE = "abstractive"


class RenderedRegion(BaseModel):
    """A region rendered to text, ready to hand to a provider.

    Kept separate from ``ContextRegion`` so the accounting object can be stored and traced while
    the text stays in memory for the length of one generation call.
    """

    model_config = ConfigDict(frozen=True)

    name: RegionName
    content: str
    #: Whether content in this region may authorise a tool call. False for ``EVIDENCE``, always.
    #: The provenance gate reads this, and it is the defence that holds when the model has
    #: already been fooled.
    grants_instruction_authority: bool = False


class Contradiction(BaseModel):
    """Two evidence groups that cannot both be true.

    Detected by pairwise entailment during validation. Surfaced rather than silently resolved
    when the two sides have comparable authority, because picking one and saying nothing is how
    a system becomes confidently wrong.
    """

    model_config = ConfigDict(frozen=True)

    left_group_id: str
    right_group_id: str
    claim: str
    score: float = Field(ge=0.0, le=1.0)


class CoverageWarning(BaseModel):
    """The context does not cover everything the query asked about.

    Reported to the client. An answer built on partial evidence is often still useful, but only
    if the gap is stated; unstated, it is indistinguishable from a complete answer.
    """

    model_config = ConfigDict(frozen=True)

    coverage: float = Field(ge=0.0, le=1.0)
    uncovered_aspects: tuple[str, ...] = ()
    dropped_group_count: int = Field(default=0, ge=0)
    reason: str | None = None


class ContextBundle(BaseModel):
    """The assembled context for one request."""

    model_config = ConfigDict(frozen=True)

    bundle_id: str
    regions: tuple[ContextRegion, ...]
    evidence: tuple[EvidenceGroup, ...] = ()
    memory_items: tuple[MemoryItem, ...] = ()
    ordering_mode: OrderingMode = OrderingMode.EDGE_WEIGHTED
    #: 0 means nothing was compressed; 5 is the full ladder applied.
    compression_level: int = Field(default=0, ge=0, le=5)
    dropped_group_ids: tuple[str, ...] = ()
    coverage_warning: bool = False
    #: Hash of the rendered prompt, for replay assertions and cache keys. Not the prompt itself.
    rendered_prompt_hash: str

    @property
    def total_used_tokens(self) -> int:
        return sum(region.used_tokens for region in self.regions)

    def region(self, name: RegionName) -> ContextRegion | None:
        return next((r for r in self.regions if r.name is name), None)

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidence)

    @property
    def independent_evidence_count(self) -> int:
        """How many genuinely independent evidence groups survived.

        The number that matters for cross-source agreement. Ten groups from one document are one
        source, and treating them as ten is how a single stale page becomes a consensus.
        """
        return sum(1 for group in self.evidence if group.independent)


class ValidationAction(StrEnum):
    """What to do when context validation fails."""

    RE_RETRIEVE_THEN_WARN = "re_retrieve_then_warn"
    WARN = "warn"
    ABSTAIN = "abstain"


class ContextValidation(BaseModel):
    """Whether the assembled context is good enough to generate from.

    Runs before generation, which is the whole point: the cheapest way to avoid a bad answer is
    to notice that the evidence could not support a good one, before paying for the tokens.
    """

    model_config = ConfigDict(frozen=True)

    passed: bool
    relevance_mean: float = Field(ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)
    contradictions: tuple[Contradiction, ...] = ()
    duplication_ratio: float = Field(ge=0.0, le=1.0)
    noise_ratio: float = Field(ge=0.0, le=1.0)
    #: Injection patterns found in retrieved text at validation time. Non-zero is not
    #: necessarily fatal, since the evidence region carries no authority, but it is a security
    #: event and a signal about the source.
    injection_hits: int = Field(default=0, ge=0)
    max_authority: float = Field(default=0.0, ge=0.0, le=1.0)
    failure_reasons: tuple[str, ...] = ()

    @property
    def has_contradictions(self) -> bool:
        return bool(self.contradictions)


#: Region layout for a request that carries no tools, used as the default allocation shape.
DEFAULT_REGION_ORDER: tuple[RegionName, ...] = (
    RegionName.SYSTEM,
    RegionName.MEMORY,
    RegionName.EVIDENCE,
    RegionName.QUERY,
)

#: Regions the compression ladder may touch, in the order it may touch them. Evidence first,
#: because it is both the largest and the only region whose loss degrades gracefully.
TRIMMABLE_REGIONS: tuple[RegionName, ...] = (RegionName.EVIDENCE, RegionName.MEMORY)

RegionAllocation = dict[RegionName, int]

ClaimIndex = Literal["sentence", "clause"]
