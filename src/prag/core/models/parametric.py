"""The parametric tier: adapter records, selection results, and the eligibility gate's verdict.

One fact governs every model in this module: **a parameter delta is uncitable, unrevocable
within a request, and unfilterable per request.** A text chunk can be excluded from a result set
based on who is asking. A weight delta cannot. Once a document is baked into an adapter, any
request served by that adapter can surface its contents.

Everything here follows from that. Adapters carry a tenant scope enforced as a hard filter
before any scoring. Eligibility is a gate rather than a heuristic. Registry rows are immutable,
because "which version of these weights answered that question" must be answerable months later.
And revocation degrades to the non-parametric path; it never degrades to serving stale weights.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum

from pydantic import BaseModel, ConfigDict, Field

from prag.core.models.common import VolatilityClass

__all__ = [
    "GLOBAL_TENANT_SCOPE",
    "AdapterRecord",
    "AdapterRef",
    "AdapterSet",
    "AdapterStatus",
    "AdapterTier",
    "CompositionMode",
    "DomainCoverage",
    "EligibilityResult",
    "KnowledgeClass",
    "KnowledgeRecord",
    "LoadedAdapter",
    "ParametricEconomics",
    "ProbeResults",
]

#: Tenant scope value meaning "may serve any tenant". Only knowledge whose ACL is tenant-global
#: or public may reach an adapter carrying this scope, and the eligibility gate enforces that.
GLOBAL_TENANT_SCOPE = "global"


class AdapterTier(IntEnum):
    """Which tier of the parametric store an adapter belongs to.

    The tiers differ in what they encode and how often they go stale, which is why they are
    refreshed on different cadences rather than treated as one pool.
    """

    #: Form, not fact: output schema, domain vocabulary, tone, refusal calibration. Cheap to
    #: maintain precisely because it does not go stale when the corpus changes.
    DOMAIN_FORM = 1
    #: Topic-cluster knowledge. This is the actual parametric RAG mechanism — the tier that
    #: replaces evidence tokens with weights.
    CLUSTER_KNOWLEDGE = 2
    #: Per-document, hard-capped. For the small set of documents that dominate query volume,
    #: where cluster granularity dilutes the signal. Per-document parameterization does not
    #: scale, so this tier exists under a count cap and not as a general mechanism.
    HOT_DOCUMENT = 3


class AdapterStatus(StrEnum):
    """Lifecycle position of one adapter version.

    ``REVOKED`` is terminal and immediate. It is set when a source document must be erased, and
    it evicts from cache rather than waiting for a retrain: serving knowledge that was legally
    required to disappear is not an acceptable degraded state.
    """

    CANDIDATE = "candidate"
    SHADOW = "shadow"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    REVOKED = "revoked"

    @property
    def servable(self) -> bool:
        """Whether an adapter in this state may serve production traffic.

        Shadow adapters run against mirrored traffic for comparison, so they are executed but
        their output is never returned to a caller. That distinction lives at the call site,
        not here.
        """
        return self is AdapterStatus.ACTIVE


class CompositionMode(StrEnum):
    """How multiple adapters combine for one request.

    Composition interference is real: naively summing several low-rank deltas degrades all of
    them. Hence ``SINGLE_BEST`` as the default rather than the most sophisticated option, and
    hence a hard cap on concurrent adapters.
    """

    #: One adapter, the highest-coverage match. Safest, and the default.
    SINGLE_BEST = "single_best"
    #: Weighted sum, with weights from selection scores and a validated scale factor.
    WEIGHTED_MERGE = "weighted_merge"
    #: The top adapter answers; a second is tried only if confidence is low.
    SEQUENTIAL_PROBE = "sequential_probe"


class KnowledgeClass(StrEnum):
    """What kind of knowledge a registry record holds.

    Every record carries exactly one primary class, and the class determines whether the record
    is even a candidate for parameterization. Note that *hybrid* is absent: it is a property of
    an answer, measured after the fact, not a storage class something can be declared to be.
    """

    PARAMETRIC_BASE = "parametric_base"
    PARAMETRIC_ADAPTED = "parametric_adapted"
    NON_PARAMETRIC_STATIC = "non_parametric_static"
    NON_PARAMETRIC_DYNAMIC = "non_parametric_dynamic"
    REAL_TIME = "real_time"
    SESSION = "session"
    LONG_TERM_USER = "long_term_user"


class DomainCoverage(BaseModel):
    """How much of a domain an adapter's training data covered."""

    model_config = ConfigDict(frozen=True)

    domain: str
    weight: float = Field(ge=0.0, le=1.0)


class ProbeResults(BaseModel):
    """What the promotion gate measured before an adapter was allowed to serve.

    Three numbers, and all three must pass. ``knowledge_recall`` alone is not enough: an adapter
    that learned its cluster perfectly while degrading the model's general ability is a net loss,
    and one that degrades its siblings is a net loss the moment it is selected alongside them.
    """

    model_config = ConfigDict(frozen=True)

    #: Held-out QA accuracy from the adapter's own cluster. Did it learn the knowledge?
    knowledge_recall: float = Field(ge=0.0, le=1.0)
    #: Loss on a general-capability probe. Did learning this cost general ability?
    general_regression: float = Field(ge=0.0)
    #: Degradation when merged with the most co-selected sibling adapters. Does it play well
    #: with the adapters it will actually be composed with?
    interference_delta: float = Field(ge=0.0)


class AdapterRecord(BaseModel):
    """One immutable adapter version in the registry.

    Immutable by design: a change produces a new version rather than mutating a row. An audit
    that cannot establish which weights answered a given request is not an audit, and mutable
    registry rows make that question unanswerable.

    ``source_document_ids`` is what makes revocation tractable. When a document must be erased,
    the registry identifies every adapter containing it, those adapters are marked revoked and
    evicted, and the affected clusters are queued for emergency retraining. The non-parametric
    path keeps serving during the gap.
    """

    model_config = ConfigDict(frozen=True)

    adapter_id: str
    version: str
    tier: AdapterTier

    #: Adapters are only valid against the base model version they were trained on. Serving a
    #: delta against a different base produces degraded output with no error to notice.
    base_model_id: str
    base_model_version: str

    #: Either ``GLOBAL_TENANT_SCOPE`` or a specific tenant id. This is the isolation boundary,
    #: and it is applied as a hard filter before scoring rather than as a ranking signal.
    tenant_scope: str

    cluster_id: str | None = None
    centroid_embedding: tuple[float, ...] = ()
    embedding_model_version: str | None = None
    domain_coverage: tuple[DomainCoverage, ...] = ()
    #: MinHash sketch of entities present in the training data, for cheap coverage checks.
    entity_coverage_sketch: str | None = None

    rank: int = Field(gt=0)
    alpha: float = Field(gt=0.0)
    target_modules: tuple[str, ...] = ()

    training_run_id: str | None = None
    training_dataset_id: str | None = None
    augmentation_config_hash: str | None = None
    probe_results: ProbeResults | None = None
    #: The eligibility gate's verdict at training time, retained for audit. Answers "why was
    #: this ever allowed to become parametric" without re-deriving a decision from changed inputs.
    eligibility_snapshot: EligibilityResult | None = None

    status: AdapterStatus = AdapterStatus.CANDIDATE
    #: Lineage. Required for both revocation and provenance shadowing.
    source_document_ids: tuple[str, ...] = ()

    created_at_ms: int
    promoted_at_ms: int | None = None
    deprecated_at_ms: int | None = None

    @property
    def is_global(self) -> bool:
        return self.tenant_scope == GLOBAL_TENANT_SCOPE

    def servable_for(self, tenant_id: str) -> bool:
        """Whether this adapter may be loaded for a request from ``tenant_id``.

        The single most important check in the parametric tier. A tenant-exclusive adapter may
        only be loaded on requests carrying that tenant id; a global adapter may serve anyone
        because the eligibility gate guaranteed it contains nothing narrower than tenant-global.
        """
        if not self.status.servable:
            return False
        return self.is_global or self.tenant_scope == tenant_id

    def covers_domain(self, domain: str, *, floor: float) -> bool:
        return any(c.domain == domain and c.weight >= floor for c in self.domain_coverage)


class AdapterRef(BaseModel):
    """A pointer to a specific adapter version, plus why it was chosen.

    Travels in the model spec, the answer diagnostics, and every conflict event. ``coverage``
    is retained because an answer produced by a barely-qualifying adapter deserves different
    treatment in fusion than one produced by a strong match.
    """

    model_config = ConfigDict(frozen=True)

    adapter_id: str
    version: str
    tier: AdapterTier
    #: Similarity between the query embedding and this adapter's cluster centroid.
    coverage: float = Field(ge=0.0, le=1.0)


class AdapterSet(BaseModel):
    """The adapters selected for one request.

    Empty is a normal, expected outcome and not a failure. Selection returns nothing rather
    than a low-coverage match, because an adapter that half-covers the query contributes
    confident noise, and confident noise is worse than the absence it replaced.
    """

    model_config = ConfigDict(frozen=True)

    adapters: tuple[AdapterRef, ...] = ()
    composition_mode: CompositionMode = CompositionMode.SINGLE_BEST
    #: Candidates that cleared the tenant filter but fell below the coverage threshold. Kept for
    #: the trace: "no adapter was selected" and "three adapters nearly qualified" are different
    #: situations, and only one of them suggests tuning the threshold.
    rejected_for_coverage: tuple[AdapterRef, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.adapters

    @property
    def best(self) -> AdapterRef | None:
        return max(self.adapters, key=lambda a: a.coverage, default=None)


class LoadedAdapter(BaseModel):
    """An adapter resident and ready to serve.

    ``cold_loaded`` distinguishes a cache hit from a 50 to 200 ms fetch out of object storage.
    A rising cold-load rate is an alert with a known remedy — grow the cache or pre-warm from
    the previous hour's selection distribution — so the signal is worth carrying per load rather
    than inferring from latency.
    """

    model_config = ConfigDict(frozen=True)

    adapter_id: str
    version: str
    resident: bool = True
    cold_loaded: bool = False
    load_latency_ms: int = Field(default=0, ge=0)
    checksum: str | None = None


class KnowledgeRecord(BaseModel):
    """A unit of knowledge in the registry, as the eligibility gate sees it.

    The fields here are exactly the ones the gate needs, which is why several of them look like
    governance metadata rather than retrieval metadata. Whether something may become parametric
    is a governance question first and an economic question second.
    """

    model_config = ConfigDict(frozen=True)

    record_id: str
    knowledge_class: KnowledgeClass
    tenant_id: str
    #: True when the ACL is narrower than tenant-global, meaning some users within the tenant
    #: cannot see this. An unconditional blocker for shared adapters.
    acl_narrower_than_tenant: bool
    volatility_class: VolatilityClass
    estimated_half_life_days: float = Field(gt=0.0)
    #: Contractual or regulatory deadline for honouring a takedown. Compared against the retrain
    #: cadence: knowledge that must vanish faster than weights can be rebuilt cannot live in
    #: weights at all.
    revocation_sla_hours: float | None = Field(default=None, gt=0.0)
    requires_exact_quotation: bool = False
    requires_per_claim_provenance: bool = False
    contains_pii: bool = False
    #: Marked as disputed or under review. Encoding a contested fact into weights hides the
    #: dispute, and there is no per-request way to reveal it again.
    contested: bool = False
    source_document_ids: tuple[str, ...] = ()


class ParametricEconomics(BaseModel):
    """The cost inputs that decide whether parameterization pays for itself.

    Non-parametric RAG has near-zero write cost and a per-query prefill cost proportional to
    evidence length. The parametric path inverts that: high write cost, near-zero per-query
    knowledge cost. So it only pays off above a query-volume threshold against stable knowledge,
    and the threshold is computed rather than assumed.
    """

    model_config = ConfigDict(frozen=True)

    training_gpu_cost_usd: float = Field(ge=0.0)
    storage_cost_per_period_usd: float = Field(ge=0.0)
    expected_queries_per_period: float = Field(ge=0.0)
    #: Evidence tokens the parametric path would displace, per query.
    evidence_tokens_displaced: int = Field(ge=0)
    prefill_cost_per_token_usd: float = Field(ge=0.0)
    retrieval_infra_cost_per_query_usd: float = Field(ge=0.0)
    #: How topically tight the training cluster is. A low score produces an adapter that knows
    #: a little about everything and nothing reliably.
    cluster_coherence_score: float = Field(ge=0.0, le=1.0)

    @property
    def amortized_parametric_cost_usd(self) -> float:
        """Parametric cost per query, or infinity when no queries are expected.

        Infinity rather than a division error: zero expected volume is a legitimate input that
        should fail the economic gate, not crash the pipeline evaluating it.
        """
        if self.expected_queries_per_period <= 0:
            return float("inf")
        total = self.training_gpu_cost_usd + self.storage_cost_per_period_usd
        return total / self.expected_queries_per_period

    @property
    def per_query_nonparametric_cost_usd(self) -> float:
        """What one query costs today, on the path parameterization would replace."""
        return (
            self.evidence_tokens_displaced * self.prefill_cost_per_token_usd
            + self.retrieval_infra_cost_per_query_usd
        )


class EligibilityResult(BaseModel):
    """The gate's verdict.

    ``blocking_reasons`` holds *every* blocker, never just the first. A curator who fixes one
    blocker only to discover the next has learned nothing about whether the knowledge is
    fundamentally ineligible, and will keep coming back. Returning the full set answers the
    real question — "can this ever be parametric" — in one pass.
    """

    model_config = ConfigDict(frozen=True)

    eligible: bool
    blocking_reasons: tuple[str, ...] = ()
    #: Set when every hard blocker cleared but the economics did not justify it. Distinct from a
    #: blocker: the knowledge is permissible to parameterize and merely not yet worth it, so it
    #: is worth re-evaluating as volume grows.
    economic_reasons: tuple[str, ...] = ()
    amortized_cost_usd: float | None = None
    nonparametric_cost_usd: float | None = None
    evaluated_at_ms: int

    @property
    def blocked_on_policy(self) -> bool:
        """Whether this was refused on governance grounds rather than economics.

        A policy block will not change when traffic grows; an economic one might. Worth
        distinguishing so nobody keeps re-running a gate that will always say no.
        """
        return bool(self.blocking_reasons)


# ``AdapterRecord`` references ``EligibilityResult``, which is defined below it because the
# gate's verdict reads more naturally alongside the economics it weighs. Resolving the forward
# reference explicitly here beats reordering the module into a shape that hides the connection.
AdapterRecord.model_rebuild()
