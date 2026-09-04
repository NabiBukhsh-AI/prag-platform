"""The configuration contract.

Pydantic Settings, so an invalid config fails at startup rather than at 3 a.m. on a code path
nobody exercised. Every behaviour in the platform gets a field here with a documented default
*before* the code that reads it, which is why sections exist for phases that are not built yet:
the contract is where a feature is designed, and a magic number in a module is a decision nobody
can find.

``config_version`` is part of every cache key. A config change therefore invalidates caches
automatically, which is the only way to keep a tuning change from being served from entries
produced under the old behaviour.

**Tenants may tune, not disable.** ``allowed_override_keys`` is deliberately explicit. A tenant
can make the system more cautious or shift its latency/cost balance; it cannot switch off a
guardrail, weaken isolation, or make a failed validation cacheable.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from prag.core.models.context import OrderingMode, ValidationAction
from prag.core.models.identity import UtilityWeights
from prag.core.models.parametric import CompositionMode

__all__ = [
    "ALLOWED_OVERRIDE_KEYS",
    "CachingConfig",
    "ContextConfig",
    "EvaluationConfig",
    "FusionConfig",
    "GenerationConfig",
    "GuardrailsConfig",
    "IntelligenceConfig",
    "MemoryConfig",
    "ObservabilityConfig",
    "OrchestrationConfig",
    "ParametricConfig",
    "PragSettings",
    "RerankingConfig",
    "RetrievalConfig",
    "RoutingConfig",
    "ServerConfig",
    "SourceConfig",
]


class ServerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    host: str = "0.0.0.0"
    port: int = Field(default=8080, gt=0, le=65535)
    request_timeout_ms: int = Field(default=30_000, gt=0)
    max_concurrent_requests: int = Field(default=512, gt=0)


class RetryPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    attempts: int = Field(default=2, ge=1)
    backoff_ms: int = Field(default=50, ge=0)
    jitter: bool = True


class CircuitBreakerPolicy(BaseModel):
    """Not optional at this dependency count.

    Without a breaker, one unhealthy source turns every request into a timeout, and the
    degradation ladder never fires because nothing reports failure fast enough to trigger it.
    """

    model_config = ConfigDict(frozen=True)

    failure_threshold: int = Field(default=5, gt=0)
    cooldown_s: int = Field(default=30, gt=0)
    half_open_probes: int = Field(default=2, gt=0)


class OrchestrationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_graph: str = "standard_answer"
    graphs: dict[str, str] = Field(
        default_factory=lambda: {
            "standard_answer": "graphs/standard_answer.yaml",
            "multi_hop": "graphs/multi_hop.yaml",
            "ingest_qa": "graphs/ingest_qa.yaml",
        }
    )
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    circuit_breaker: CircuitBreakerPolicy = Field(default_factory=CircuitBreakerPolicy)


class ClassifierTierConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    model: str | None = None
    device: str = "cpu"
    batch_window_ms: int = Field(default=8, ge=0)
    #: Cap on how much traffic may reach the expensive LLM fallback. Without a cap, a bad day
    #: for the fast classifier silently becomes an LLM call on every request.
    max_share_of_traffic: float = Field(default=0.05, ge=0.0, le=1.0)


class TransformConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    budget_ms: int = Field(gt=0)
    max_sub_queries: int = Field(default=6, gt=0)


class IntelligenceConfig(BaseModel):
    """Query understanding and routing, which ship as one module."""

    model_config = ConfigDict(frozen=True)

    t0_rules: bool = True
    t1: ClassifierTierConfig = Field(
        default_factory=lambda: ClassifierTierConfig(model="prag-router-v4")
    )
    t2_fallback: ClassifierTierConfig = Field(
        default_factory=lambda: ClassifierTierConfig(model="small.general")
    )
    confidence_thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "intent": 0.75,
            "domain": 0.70,
            "complexity": 0.70,
            "escalate_to_t2_below": 0.65,
        }
    )
    #: Above this uncertainty the orchestrator hedges: retrieval runs even on a parametric lean,
    #: so the more expensive path is available if fusion turns out to need it.
    hedge_above_uncertainty: float = Field(default=0.35, ge=0.0, le=1.0)
    speculative_retrieval: bool = True
    transforms: dict[str, TransformConfig] = Field(
        default_factory=lambda: {
            "rewrite": TransformConfig(budget_ms=60),
            "coreference": TransformConfig(budget_ms=25),
            "expansion": TransformConfig(budget_ms=5),
            "decomposition": TransformConfig(budget_ms=150),
        }
    )


class RoutingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    utility_weights: dict[str, UtilityWeights] = Field(
        default_factory=lambda: {
            "interactive": UtilityWeights(quality=0.50, latency=0.35, cost=0.15),
            "standard": UtilityWeights(quality=0.60, latency=0.20, cost=0.20),
            "high_stakes": UtilityWeights(quality=0.85, latency=0.05, cost=0.10),
            "batch": UtilityWeights(quality=0.70, latency=0.00, cost=0.30),
        }
    )
    #: Fraction of traffic routed against the argmax, to keep the strategy quality table honest.
    #: These requests are the counterfactual sample used to measure routing regret; without
    #: them the table only ever confirms what it already believes.
    exploration_fraction: float = Field(default=0.01, ge=0.0, le=0.2)
    quality_table_refresh: str = "0 3 * * *"


class EligibilityConfig(BaseModel):
    """The Parametric Eligibility Gate's thresholds.

    Governance first, economics second. No economic argument overrides a hard blocker: knowledge
    that cannot be filtered per request, revoked in time, or quoted verbatim is ineligible at
    any price.
    """

    model_config = ConfigDict(frozen=True)

    #: Below this, the adapter is stale before it is useful.
    min_half_life_days: float = Field(default=90.0, gt=0.0)
    min_queries_per_month: int = Field(default=500, ge=0)
    #: The parametric path must be at least this much cheaper to justify the operational burden
    #: it adds. 0.6 means a 40 percent saving is the floor.
    savings_threshold: float = Field(default=0.6, gt=0.0, le=1.0)
    #: Below this, the cluster is incoherent and produces an adapter that knows a little about
    #: everything and nothing reliably.
    coherence_floor: float = Field(default=0.55, ge=0.0, le=1.0)
    block_if_pii: bool = True
    block_if_acl_narrower_than_tenant: bool = True
    block_if_requires_exact_quotation: bool = True


class AdapterSelectionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    #: Composition interference is real: naively summing low-rank deltas degrades all of them.
    max_concurrent_adapters: int = Field(default=2, ge=1, le=4)
    #: Below this coverage, selection returns nothing. An adapter that half-covers the query
    #: contributes confident noise, which is worse than the absence it replaced.
    min_coverage_similarity: float = Field(default=0.62, ge=0.0, le=1.0)
    composition_mode: CompositionMode = CompositionMode.SINGLE_BEST


class AdapterTrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    rank: int = Field(default=8, gt=0)
    alpha: float = Field(default=16.0, gt=0.0)
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    epochs: int = Field(default=2, gt=0)
    lr: float = Field(default=1.0e-4, gt=0.0)
    paraphrases_per_chunk: int = Field(default=3, ge=0)
    qa_pairs_per_chunk: int = Field(default=5, ge=0)
    #: Synthetic QA pairs not entailed by the source teach the model falsehoods, permanently,
    #: with no way to cite or revoke them at request time. This filter is not optional.
    entailment_filter: float = Field(default=0.85, ge=0.0, le=1.0)
    min_knowledge_recall: float = Field(default=0.80, ge=0.0, le=1.0)
    max_general_regression: float = Field(default=0.02, ge=0.0)
    max_interference_delta: float = Field(default=0.03, ge=0.0)
    shadow_traffic_requests: int = Field(default=500, ge=0)


class ParametricConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    #: Off by default, and Phase 1 ships without it. You cannot know what is worth
    #: parameterizing until a non-parametric baseline has been measured, and the system is
    #: designed to be fully useful with this false.
    enabled: bool = False
    eligibility: EligibilityConfig = Field(default_factory=EligibilityConfig)
    selection: AdapterSelectionConfig = Field(default_factory=AdapterSelectionConfig)
    hot_adapters: int = Field(default=24, ge=0)
    prewarm_from_last_hours: int = Field(default=1, ge=0)
    training: AdapterTrainingConfig = Field(default_factory=AdapterTrainingConfig)


class SourceConfig(BaseModel):
    """One registered knowledge source.

    ``impl`` names an implementation the factory resolves, so adding a source is a config entry
    rather than a code change and vendor choice stays out of the codebase.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    impl: str
    collection: str | None = None
    index: str | None = None
    #: A failed required leg is a plan failure; a failed optional leg is a coverage warning.
    #: Marking everything required removes the ability to degrade, and nothing required removes
    #: the ability to notice that it has.
    required: bool = False
    weight: float = Field(default=1.0, ge=0.0)


class EarlyExitConfig(BaseModel):
    """Skip the rest of retrieval when the top hit is decisive.

    Both conditions must hold: a high absolute score and a clear margin over the runner-up. A
    high score alone is common when everything matches equally well, and exiting there discards
    the evidence that would have shown the ambiguity.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    min_top1_score: float = Field(default=0.86, ge=0.0, le=1.0)
    min_margin: float = Field(default=0.18, ge=0.0, le=1.0)


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    sources: tuple[SourceConfig, ...] = ()
    top_k: int = Field(default=24, gt=0)
    wall_ms: int = Field(default=260, gt=0)
    fusion_method: Literal["rrf", "weighted"] = "rrf"
    #: RRF's rank damping constant. Rank-based fusion needs no score calibration between
    #: sources, which matters because BM25 scores and cosine similarities are not comparable.
    fusion_k: int = Field(default=60, gt=0)
    early_exit: EarlyExitConfig = Field(default_factory=EarlyExitConfig)


class RerankTierConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    input_k: int = Field(gt=0)
    output_k: int = Field(gt=0)
    timeout_ms: int = Field(gt=0)

    @model_validator(mode="after")
    def _output_not_larger_than_input(self) -> RerankTierConfig:
        if self.output_k > self.input_k:
            raise ValueError(f"output_k ({self.output_k}) cannot exceed input_k ({self.input_k})")
        return self


class RerankingConfig(BaseModel):
    """The most expensive optional stage, and the first thing the ladder drops."""

    model_config = ConfigDict(frozen=True)

    tier_by_sla: dict[str, str] = Field(
        default_factory=lambda: {
            "interactive": "light",
            "standard": "light",
            "high_stakes": "standard",
            "batch": "llm",
        }
    )
    tiers: dict[str, RerankTierConfig] = Field(
        default_factory=lambda: {
            "light": RerankTierConfig(
                model="cross-encoder-distil-onnx-int8", input_k=30, output_k=8, timeout_ms=70
            ),
            "standard": RerankTierConfig(
                model="cross-encoder-base", input_k=40, output_k=8, timeout_ms=150
            ),
            "llm": RerankTierConfig(model="small.general", input_k=20, output_k=8, timeout_ms=500),
        }
    )


class CompressionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled_stages: tuple[str, ...] = ("drop", "parent_narrow", "extractive", "hierarchical")
    #: Off by default. A small model summarising evidence can fabricate, and a fabrication
    #: introduced during compression is indistinguishable downstream from one the answer model
    #: invented.
    abstractive_enabled: bool = False
    #: Never abstractively compressed, at any setting. These are the content types where a
    #: paraphrase that is 95 percent right is 100 percent wrong.
    abstractive_forbid_on: tuple[str, ...] = (
        "numeral",
        "date",
        "dosage",
        "money",
        "identifier",
        "code",
        "quote",
    )


class ContextValidationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    relevance_floor: float = Field(default=0.35, ge=0.0, le=1.0)
    coverage_floor: float = Field(default=0.60, ge=0.0, le=1.0)
    contradiction_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    noise_max: float = Field(default=0.40, ge=0.0, le=1.0)
    on_failure: ValidationAction = ValidationAction.RE_RETRIEVE_THEN_WARN


class ContextConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_evidence_tokens: int = Field(default=8000, gt=0)
    memory_tokens: int = Field(default=2000, ge=0)
    output_reserve_headroom: float = Field(default=0.25, ge=0.0, lt=1.0)
    ordering_mode: OrderingMode = OrderingMode.EDGE_WEIGHTED
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    validation: ContextValidationConfig = Field(default_factory=ContextValidationConfig)


class FusionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "w1": 0.40,
            "w2": 0.20,
            "w3": 0.20,
            "w4": 0.15,
            "w5": 0.05,
        }
    )
    #: Parametric knowledge gets a fixed, modest authority. It cannot be audited per claim, so
    #: it does not outrank a curated source on authority alone.
    parametric_authority: float = Field(default=0.55, ge=0.0, le=1.0)
    abstain_below_knowledge_score: float = Field(default=0.42, ge=0.0, le=1.0)
    #: The hard gate preventing the worst failure this system can produce: a confident-sounding
    #: general answer to a question about the tenant's own data.
    abstain_on_private_query_without_evidence: bool = True
    surface_conflicts_when_authority_delta_below: float = Field(default=0.15, ge=0.0, le=1.0)
    provenance_shadowing_enabled: bool = True
    provenance_entailment_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    provenance_budget_ms: int = Field(default=40, gt=0)


class GenerationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    routing: dict[str, str] = Field(
        default_factory=lambda: {
            "simple_factual": "mid.instruct",
            "reasoning": "large.reasoning",
            "domain": "mid.instruct",
        }
    )
    escalation_enabled: bool = True
    #: One escalation per request. Uncapped, a request that keeps finding reasons to escalate
    #: is how cost per request grows unnoticed while quality stays flat.
    max_escalations_per_request: int = Field(default=1, ge=0)
    fallback_chains: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    sentence_buffer: bool = True
    regenerate_on_grounding_failure: bool = True


class GuardrailsConfig(BaseModel):
    """The chain always runs. Individual guardrails are toggled here, never skipped in code."""

    model_config = ConfigDict(frozen=True)

    input: tuple[str, ...] = (
        "injection",
        "instruction_override",
        "pii",
        "policy",
        "tenant_assertion",
        "payload_limits",
    )
    retrieval: tuple[str, ...] = (
        "doc_injection",
        "acl_recheck",
        "poison_heuristics",
        "canary",
        "source_health",
    )
    output: tuple[str, ...] = (
        "grounding",
        "citation_validation",
        "leakage",
        "schema",
        "policy",
    )
    #: The defence that holds after the model has already been fooled by a document-borne
    #: injection. Retrieved evidence can never authorise a tool call.
    tool_provenance_gate: bool = True
    strict_mode: bool = False


class MemoryConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_ttl_hours: int = Field(default=24, gt=0)
    summarize_after_turns: int = Field(default=8, gt=0)
    keep_verbatim: tuple[str, ...] = ("entities", "decisions")
    long_term_enabled: bool = True
    long_term_max_items: int = Field(default=500, gt=0)
    salience_decay_half_life_days: float = Field(default=60.0, gt=0.0)
    write_sources: tuple[str, ...] = ("user_asserted", "confirmed_structured")
    #: Model-generated content never persists. It is how a system starts confidently believing
    #: things nobody ever told it, and the belief survives every subsequent session.
    forbid: tuple[str, ...] = ("model_generated",)


class CacheTierConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    ttl_s: int = Field(default=600, ge=0)
    similarity_floor: float | None = Field(default=None, ge=0.0, le=1.0)


class CachingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    tiers: dict[str, CacheTierConfig] = Field(
        default_factory=lambda: {
            "exact": CacheTierConfig(ttl_s=86_400),
            "semantic": CacheTierConfig(ttl_s=3_600, similarity_floor=0.95),
            "embedding": CacheTierConfig(ttl_s=2_592_000),
            "retrieval": CacheTierConfig(ttl_s=600),
            "rerank": CacheTierConfig(ttl_s=3_600),
            "analysis": CacheTierConfig(ttl_s=604_800),
        }
    )
    ttl_by_volatility: dict[str, int] = Field(
        default_factory=lambda: {"static": 86_400, "slow": 14_400, "fast": 900, "realtime": 0}
    )
    #: Enforced in key construction, not left to callers. A caching rule that depends on every
    #: call site remembering it gets broken once and then stays broken invisibly.
    never_cache: tuple[str, ...] = (
        "abstained",
        "coverage_warning",
        "staleness_warning",
        "validation_failed",
    )


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    online_sample_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    counterfactual_sample_rate: float = Field(default=0.01, ge=0.0, le=1.0)
    judge_model: str = "large.reasoning"
    #: No judge score gates anything until it has been calibrated against human labels. An
    #: uncalibrated judge reporting confident nonsense at scale is a real failure mode.
    judge_calibrated: bool = False
    faithfulness_min: float = Field(default=0.92, ge=0.0, le=1.0)
    citation_precision_min: float = Field(default=0.95, ge=0.0, le=1.0)
    #: The adversarial suite is a blocking gate at 100 percent. A single injection or ACL probe
    #: getting through is not a percentage, it is an incident.
    adversarial_pass_rate_min: float = Field(default=1.0, ge=0.0, le=1.0)


class ObservabilityConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    otel_endpoint: str | None = None
    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    #: Retention differs by content class, which is the point. A trace skeleton is safe to keep
    #: for a quarter; the query text inside it is not, and the audit trail must outlive both.
    retention_days: dict[str, int] = Field(
        default_factory=lambda: {"trace_skeleton": 90, "query_text": 7, "audit": 2555}
    )
    debug_trace_enabled: bool = True
    debug_trace_scope: str = "platform:debug"


#: What a tenant may override. Deliberately explicit and deliberately short: a tenant can tune
#: how cautious the system is and how it trades latency against cost, and cannot switch off a
#: guardrail, weaken isolation, or make a failed validation cacheable.
ALLOWED_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {
        "routing.utility_weights",
        "guardrails.strict_mode",
        "context.max_evidence_tokens",
        "fusion.abstain_below_knowledge_score",
        "caching.tiers.semantic.similarity_floor",
        "parametric.enabled",
    }
)


class PragSettings(BaseSettings):
    """The whole configuration contract.

    Layered at load: shipped defaults, then an environment file, then environment variables,
    then per-tenant overrides resolved at request time. Environment variables use a nested
    delimiter, so ``PRAG_SERVER__PORT=9000`` sets ``server.port``.
    """

    model_config = SettingsConfigDict(
        env_prefix="PRAG_",
        env_nested_delimiter="__",
        frozen=True,
        extra="forbid",
    )

    #: Part of every cache key, so changing config invalidates caches automatically.
    config_version: str = "2026.09.03-1"

    server: ServerConfig = Field(default_factory=ServerConfig)
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
    intelligence: IntelligenceConfig = Field(default_factory=IntelligenceConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    parametric: ParametricConfig = Field(default_factory=ParametricConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    caching: CachingConfig = Field(default_factory=CachingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    tenant_overrides_path: str = "config/tenants/"

    @model_validator(mode="after")
    def _internally_consistent(self) -> PragSettings:
        """Catch the cross-section mistakes a per-field validator cannot see."""
        if self.orchestration.default_graph not in self.orchestration.graphs:
            raise ValueError(
                f"default graph {self.orchestration.default_graph!r} is not in the graph registry"
            )

        unknown_tiers = set(self.reranking.tier_by_sla.values()) - set(self.reranking.tiers)
        if unknown_tiers - {"none"}:
            raise ValueError(f"rerank tiers referenced but not defined: {sorted(unknown_tiers)}")

        weights_total = sum(self.fusion.weights.values())
        if abs(weights_total - 1.0) > 1e-6:
            raise ValueError(f"fusion weights must sum to 1.0, got {weights_total:.6f}")

        required_source_ids = [s.id for s in self.retrieval.sources if s.required]
        if self.retrieval.sources and not required_source_ids:
            raise ValueError(
                "at least one retrieval source must be required, "
                "or a total retrieval failure cannot be detected"
            )

        # The parametric tier is gated on evaluation existing. Enabling it without a calibrated
        # judge means there is no way to tell whether an adapter helped, which is the exact
        # situation the phased roadmap exists to prevent.
        if self.parametric.enabled and not self.evaluation.judge_calibrated:
            raise ValueError(
                "parametric.enabled requires evaluation.judge_calibrated: "
                "without calibrated evaluation there is no way to tell whether an adapter helped"
            )

        return self
