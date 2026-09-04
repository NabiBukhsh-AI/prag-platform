"""The knowledge registry: what the platform knows about each source it indexes.

This is governance metadata as much as retrieval metadata, and the fields that look like
bureaucracy are the ones that stop the system from being confidently wrong.

**Authority is domain-scoped.** A single trust score per source is the wrong shape: a source
that is the system of record for legal matters may be worthless on engineering ones, and a flat
score forces one of those two judgements to be wrong everywhere.

**Staleness is a state machine, not a timestamp.** "Last changed on the 19th" answers nothing
without knowing how fast this knowledge decays. The states encode the answer.

**Index health is measured, not assumed.** Probe recall per source is the signal that catches a
chunking strategy which is wrong for one source type while every other source looks fine.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from prag.core.models.common import VolatilityClass
from prag.core.models.parametric import EligibilityResult

__all__ = [
    "AccessPolicy",
    "AclMode",
    "Authority",
    "IndexHealth",
    "IndexingProfile",
    "LineageEdge",
    "SourceRecord",
    "SourceTemporality",
    "SourceType",
    "StalenessState",
]


class SourceType(StrEnum):
    DOCUMENT_COLLECTION = "document_collection"
    STRUCTURED_TABLE = "structured_table"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    EXTERNAL_API = "external_api"


class StalenessState(StrEnum):
    """Where a source sits in its freshness lifecycle.

    A state machine rather than a computed boolean, because the transitions are what trigger
    action: entering ``STALE`` enqueues a reindex, entering ``EXPIRED`` makes the source
    unusable in strict mode. A boolean recomputed on read would fire the same trigger on every
    request or on none.
    """

    FRESH = "fresh"
    #: Past its expected half-life but still usable, with a staleness warning attached.
    AGING = "aging"
    #: Past the point where answers should be trusted. Reindex is queued.
    STALE = "stale"
    #: Beyond its TTL. Unusable in strict mode; served only with an explicit age warning.
    EXPIRED = "expired"
    #: Source change detected, reindex in flight. Existing content is still served.
    REINDEXING = "reindexing"

    @property
    def warrants_warning(self) -> bool:
        return self is not StalenessState.FRESH

    @property
    def blocks_in_strict_mode(self) -> bool:
        return self in (StalenessState.STALE, StalenessState.EXPIRED)


class Authority(BaseModel):
    """How much this source is trusted, and by whom it was decided.

    ``verified_by`` and ``verified_at`` are not decoration. Authority is the strongest signal in
    conflict resolution, so an authority score with no accountable owner is a way for one
    curator's opinion to silently outrank a system of record.
    """

    model_config = ConfigDict(frozen=True)

    base_score: float = Field(ge=0.0, le=1.0)
    #: Per-domain overrides. A source can be canonical in one domain and unreliable in another,
    #: and collapsing that into one number makes one of those judgements wrong everywhere.
    domain_overrides: dict[str, float] = Field(default_factory=dict)
    rationale: str | None = None
    verified_by: str | None = None
    verified_at_ms: int | None = None

    def for_domain(self, domain: str) -> float:
        """Authority in a specific domain, falling back to the base score."""
        return self.domain_overrides.get(domain, self.base_score)


class SourceTemporality(BaseModel):
    """How fast this source's content goes stale.

    Distinct from a query's temporality: this describes the corpus, that describes what the
    asker needs. Freshness scoring compares the two.
    """

    model_config = ConfigDict(frozen=True)

    volatility_class: VolatilityClass
    expected_half_life_days: float = Field(gt=0.0)
    ttl_seconds: int | None = Field(default=None, gt=0)
    last_source_change_ms: int | None = None
    staleness_state: StalenessState = StalenessState.FRESH


class IndexHealth(BaseModel):
    """Whether this source is actually retrievable, as opposed to merely indexed.

    ``probe_recall_at_10`` is the number that matters. Chunk counts say the pipeline ran;
    probe recall says the result is findable. A chunking strategy wrong for one source type caps
    retrieval quality regardless of everything downstream, and only a per-source probe catches
    it while every other source looks healthy.
    """

    model_config = ConfigDict(frozen=True)

    chunk_count: int = Field(default=0, ge=0)
    #: Chunks the pipeline refused. A rising rate points at an extractor, not at retrieval.
    rejected_chunks: int = Field(default=0, ge=0)
    probe_recall_at_10: float | None = Field(default=None, ge=0.0, le=1.0)
    last_probed_at_ms: int | None = None

    @property
    def rejection_rate(self) -> float:
        total = self.chunk_count + self.rejected_chunks
        return self.rejected_chunks / total if total else 0.0


class IndexingProfile(BaseModel):
    """How this source was turned into an index, precisely enough to reproduce or invalidate it.

    Every field here is part of a cache key or a migration decision. A change to the chunking
    config means existing chunks were produced by a process that no longer exists, and a change
    to the embedding model means the vectors are not comparable to new queries.
    """

    model_config = ConfigDict(frozen=True)

    embedding_model: str
    embedding_model_version: str
    embedding_dim: int = Field(gt=0)
    chunking_strategy: str
    chunking_config_hash: str | None = None
    index_targets: tuple[str, ...] = ()
    last_indexed_ms: int | None = None
    health: IndexHealth = Field(default_factory=IndexHealth)


class AclMode(StrEnum):
    PUBLIC = "public"
    TENANT = "tenant"
    GROUP = "group"
    USER = "user"

    @property
    def narrower_than_tenant(self) -> bool:
        """Whether this mode restricts visibility below tenant-wide.

        The parametric eligibility gate reads this: anything narrower than tenant-global cannot
        be parameterized, because a weight delta cannot be filtered per request.
        """
        return self in (AclMode.GROUP, AclMode.USER)


class AccessPolicy(BaseModel):
    """Who may see this source, and whether it contains anything sensitive."""

    model_config = ConfigDict(frozen=True)

    acl_mode: AclMode
    acl_groups: tuple[str, ...] = ()
    #: Precomputed hash written into the vector payload, so filtering is a cheap equality check
    #: rather than a join against a permissions table on the retrieval path.
    acl_hash: str
    pii_present: bool = False
    pii_categories: tuple[str, ...] = ()


class LineageEdge(BaseModel):
    """One derivation: this artifact came from that one.

    The lineage graph is what makes the independence correction possible. Three documents
    quoting the same press release share a root, and without these edges their agreement reads
    as corroboration rather than as one source counted three times.
    """

    model_config = ConfigDict(frozen=True)

    child_id: str
    parent_id: str
    relation: Literal["derived_from", "quotes", "translates", "supersedes", "chunk_of"]
    connector: str | None = None
    recorded_at_ms: int


class SourceRecord(BaseModel):
    """One registered knowledge source.

    Forward-compatible: unknown fields are preserved so a rolling deploy does not drop data
    written by a newer version.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    schema_version: Literal["kr.v2"] = "kr.v2"
    source_id: str
    tenant_id: str
    type: SourceType

    authority: Authority
    temporality: SourceTemporality
    indexing: IndexingProfile
    access: AccessPolicy

    source_version: str | None = None
    document_count: int = Field(default=0, ge=0)
    content_hash: str | None = None
    supersedes: str | None = None

    #: The parametric gate's verdict, stored with its blocking reasons for audit. Answers "why
    #: is this not parametric" without re-deriving a decision from inputs that have since moved.
    parametric_eligibility: EligibilityResult | None = None

    owner: str | None = None
    created_at_ms: int
    updated_at_ms: int | None = None

    def authority_for(self, domain: str) -> float:
        return self.authority.for_domain(domain)

    @property
    def is_stale(self) -> bool:
        return self.temporality.staleness_state.warrants_warning

    @property
    def parametric_eligible(self) -> bool:
        """Whether this source may be parameterized right now.

        Absent evaluation means no. A source that has never been through the gate has not been
        cleared by it, and defaulting the other way would let an unevaluated source become
        parametric through nothing more than an oversight.
        """
        return bool(self.parametric_eligibility and self.parametric_eligibility.eligible)
