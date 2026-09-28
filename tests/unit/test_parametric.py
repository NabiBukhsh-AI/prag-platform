"""The parametric tier: the eligibility gate, weight residency, selection, and revocation.

The test that matters most is ``TestTenantIsolation``: a tenant-scoped adapter must never serve
another tenant, however well it matches. Once a delta is merged into the serving weights there is
no per-request filter that can take it back out.
"""

from __future__ import annotations

import pytest

from prag.core.errors import AdapterLoadError
from prag.core.models.common import EmbeddingPurpose, VolatilityClass
from prag.core.models.identity import Principal
from prag.core.models.parametric import (
    GLOBAL_TENANT_SCOPE,
    AdapterRecord,
    AdapterStatus,
    AdapterTier,
    CompositionMode,
    DomainCoverage,
    KnowledgeClass,
    KnowledgeRecord,
    ParametricEconomics,
)
from prag.core.protocols import AdapterSelector, AdapterStore
from prag.ingestion.embedding import HashingEmbeddingProvider
from prag.parametric import (
    BLOCKERS,
    AdapterRegistry,
    CentroidAdapterSelector,
    EligibilityGate,
    InMemoryBlobStore,
    LruAdapterStore,
    blob_key,
)
from prag.storage.repositories import InMemoryAdapterRepository
from tests.unit.test_graph_engine import an_analysis

# ---------------------------------------------------------------------------
# The eligibility gate
# ---------------------------------------------------------------------------


def a_record(**overrides: object) -> KnowledgeRecord:
    fields: dict[str, object] = {
        "record_id": "k1",
        "knowledge_class": KnowledgeClass.NON_PARAMETRIC_STATIC,
        "tenant_id": "tenant-a",
        "acl_narrower_than_tenant": False,
        "volatility_class": VolatilityClass.STATIC,
        "estimated_half_life_days": 365.0,
    }
    return KnowledgeRecord(**{**fields, **overrides})


def economics(**overrides: object) -> ParametricEconomics:
    fields: dict[str, object] = {
        "training_gpu_cost_usd": 20.0,
        "storage_cost_per_period_usd": 1.0,
        "expected_queries_per_period": 10_000,
        "evidence_tokens_displaced": 3_000,
        "prefill_cost_per_token_usd": 0.000_003,
        "retrieval_infra_cost_per_query_usd": 0.000_5,
        "cluster_coherence_score": 0.8,
    }
    return ParametricEconomics(**{**fields, **overrides})


GATE = EligibilityGate(clock=lambda: 1_000.0)

#: One record per blocker, each clean except for the one property that should block it.
BLOCKING = {
    "knowledge_class_not_parameterizable": {"knowledge_class": KnowledgeClass.SESSION},
    "acl_narrower_than_tenant": {"acl_narrower_than_tenant": True},
    "revocation_sla_shorter_than_retrain_cadence": {"revocation_sla_hours": 24.0},
    "requires_exact_quotation": {"requires_exact_quotation": True},
    "half_life_below_floor": {"estimated_half_life_days": 30.0},
    "contested": {"contested": True},
    "contains_pii": {"contains_pii": True},
}


class TestEligibilityGate:
    def test_a_clean_record_with_good_economics_is_eligible(self) -> None:
        result = GATE.evaluate(a_record(), economics())
        assert result.eligible
        assert result.blocking_reasons == ()
        assert result.evaluated_at_ms == 1_000_000

    @pytest.mark.parametrize(("code", "overrides"), BLOCKING.items(), ids=list(BLOCKING))
    def test_each_blocker_blocks_alone(self, code: str, overrides: dict) -> None:
        result = GATE.evaluate(a_record(**overrides), economics())
        assert not result.eligible
        assert result.blocking_reasons == (code,)
        assert result.blocked_on_policy

    def test_per_claim_provenance_blocks_only_without_shadowing(self) -> None:
        record = a_record(requires_per_claim_provenance=True)
        assert GATE.evaluate(record, economics()).eligible
        no_shadowing = EligibilityGate(provenance_shadowing_enabled=False)
        assert no_shadowing.evaluate(record, economics()).blocking_reasons == (
            "per_claim_provenance_without_shadowing",
        )

    def test_every_blocker_is_reported_not_only_the_first(self) -> None:
        """A curator who fixes one blocker only to hit the next has learned nothing."""
        everything = {k: v for overrides in BLOCKING.values() for k, v in overrides.items()}
        result = EligibilityGate(provenance_shadowing_enabled=False).evaluate(
            a_record(**everything, requires_per_claim_provenance=True), economics()
        )
        assert result.blocking_reasons == BLOCKERS

    def test_no_economic_argument_overrides_a_blocker(self) -> None:
        free = economics(training_gpu_cost_usd=0.0, storage_cost_per_period_usd=0.0)
        result = GATE.evaluate(a_record(contains_pii=True), free)
        assert not result.eligible
        assert result.economic_reasons == ()
        assert result.amortized_cost_usd is None

    @pytest.mark.parametrize(
        ("overrides", "reason"),
        [
            ({"expected_queries_per_period": 100}, "volume_below_floor"),
            ({"training_gpu_cost_usd": 10_000.0}, "savings_below_threshold"),
            ({"cluster_coherence_score": 0.3}, "cluster_incoherent"),
        ],
    )
    def test_economic_refusals(self, overrides: dict, reason: str) -> None:
        result = GATE.evaluate(a_record(), economics(**overrides))
        assert not result.eligible
        assert reason in result.economic_reasons
        assert not result.blocked_on_policy, "worth re-evaluating as traffic grows"

    def test_zero_volume_fails_economics_rather_than_crashing(self) -> None:
        result = GATE.evaluate(a_record(), economics(expected_queries_per_period=0))
        assert "volume_below_floor" in result.economic_reasons
        assert "savings_below_threshold" in result.economic_reasons


# ---------------------------------------------------------------------------
# Registry, residency and selection
# ---------------------------------------------------------------------------

EMBEDDER = HashingEmbeddingProvider(dimensions=64)
BASE = "base-v1"


async def centroid(text: str) -> tuple[float, ...]:
    from prag.core.models.common import Deadline

    (vector,) = await EMBEDDER.embed([text], EmbeddingPurpose.DOCUMENT, Deadline.in_ms(1_000))
    return tuple(vector)


async def an_adapter(
    adapter_id: str,
    *,
    tenant_scope: str = GLOBAL_TENANT_SCOPE,
    topic: str = "incident escalation paging on-call lead",
    version: str = "1",
    base: str = BASE,
    domain: str = "ops",
    documents: tuple[str, ...] = ("doc-1",),
) -> AdapterRecord:
    return AdapterRecord(
        adapter_id=adapter_id,
        version=version,
        tier=AdapterTier.CLUSTER_KNOWLEDGE,
        base_model_id="base",
        base_model_version=base,
        tenant_scope=tenant_scope,
        centroid_embedding=await centroid(topic),
        embedding_model_version=EMBEDDER.model_version,
        domain_coverage=(DomainCoverage(domain=domain, weight=0.9),),
        rank=8,
        alpha=16.0,
        source_document_ids=documents,
        created_at_ms=0,
    )


def a_registry(capacity: int = 24) -> AdapterRegistry:
    repository = InMemoryAdapterRepository()
    blobs = InMemoryBlobStore()
    return AdapterRegistry(repository, blobs, LruAdapterStore(repository, blobs, capacity=capacity))


async def active(registry: AdapterRegistry, record: AdapterRecord) -> AdapterRecord:
    await registry.register(record, weights=f"weights:{record.adapter_id}".encode())
    await registry.promote(record.adapter_id, record.version)
    return record


def a_selector(registry: AdapterRegistry, **overrides: object) -> CentroidAdapterSelector:
    fields: dict[str, object] = {"base_model_version": BASE, "min_coverage": 0.62}
    return CentroidAdapterSelector(registry, EMBEDDER, **{**fields, **overrides})


def an_escalation_query():
    return an_analysis().model_copy(
        update={"normalized_query": "incident escalation paging on-call lead"}
    )


def principal(tenant: str) -> Principal:
    return Principal(tenant_id=tenant, user_id="u1")


class TestStore:
    def test_the_implementations_satisfy_the_protocols(self) -> None:
        registry = a_registry()
        assert isinstance(registry.store, AdapterStore)
        assert isinstance(a_selector(registry), AdapterSelector)

    async def test_cold_then_warm(self) -> None:
        registry = a_registry()
        await active(registry, await an_adapter("a1"))

        first = await registry.store.load("a1", "1")
        second = await registry.store.load("a1", "1")
        assert first.cold_loaded
        assert not second.cold_loaded
        assert await registry.store.is_resident("a1", "1")

    async def test_the_least_recently_used_adapter_is_evicted(self) -> None:
        registry = a_registry(capacity=2)
        for name in ("a1", "a2", "a3"):
            await active(registry, await an_adapter(name))
        await registry.store.load("a1", "1")
        await registry.store.load("a2", "1")
        await registry.store.load("a1", "1")  # a2 is now least recent
        await registry.store.load("a3", "1")

        assert await registry.store.is_resident("a1", "1")
        assert not await registry.store.is_resident("a2", "1")

    async def test_a_corrupted_blob_refuses_to_load(self) -> None:
        """A corrupted delta does not error at inference; it degrades answers invisibly."""
        registry = a_registry()
        await active(registry, await an_adapter("a1"))
        registry.blobs.corrupt(blob_key("a1", "1"))

        with pytest.raises(AdapterLoadError, match="checksum"):
            await registry.store.load("a1", "1")

    async def test_a_candidate_does_not_load(self) -> None:
        registry = a_registry()
        await registry.register(await an_adapter("a1"), weights=b"w")
        with pytest.raises(AdapterLoadError):
            await registry.store.load("a1", "1")

    async def test_a_shadow_loads_only_for_mirrored_traffic(self) -> None:
        registry = a_registry()
        await registry.register(await an_adapter("a1"), weights=b"w")
        await registry.set_status("a1", "1", AdapterStatus.SHADOW)

        with pytest.raises(AdapterLoadError):
            await registry.store.load("a1", "1")
        assert (await registry.store.load("a1", "1", include_shadow=True)).cold_loaded

    async def test_promotion_deprecates_the_previous_version(self) -> None:
        registry = a_registry()
        await active(registry, await an_adapter("a1", version="1"))
        await active(registry, await an_adapter("a1", version="2"))

        assert (await registry.repository.get("a1", "1")).status is AdapterStatus.DEPRECATED
        assert [r.version for r in registry.servable("tenant-a")] == ["2"]


class TestTenantIsolation:
    """The cross-tenant adapter canary. Blocking in CI through the unit suite."""

    async def test_a_tenant_adapter_never_serves_another_tenant(self) -> None:
        """A perfect match for the query, scoped to tenant A, requested by tenant B."""
        registry = a_registry()
        await active(registry, await an_adapter("private-a", tenant_scope="tenant-a"))

        selection = await a_selector(registry).select(
            an_escalation_query(), principal("tenant-b"), max_adapters=2
        )
        assert selection.is_empty
        assert selection.rejected_for_coverage == (), "filtered before scoring, not scored low"

    async def test_the_owning_tenant_gets_it(self) -> None:
        registry = a_registry()
        await active(registry, await an_adapter("private-a", tenant_scope="tenant-a"))

        selection = await a_selector(registry).select(
            an_escalation_query(), principal("tenant-a"), max_adapters=2
        )
        assert [a.adapter_id for a in selection.adapters] == ["private-a"]
        # Near 1, not exactly: query and document embeddings are deliberately asymmetric.
        assert selection.adapters[0].coverage > 0.95

    async def test_a_global_adapter_serves_everyone(self) -> None:
        registry = a_registry()
        await active(registry, await an_adapter("shared"))
        for tenant in ("tenant-a", "tenant-b"):
            selection = await a_selector(registry).select(
                an_escalation_query(), principal(tenant), max_adapters=1
            )
            assert not selection.is_empty

    async def test_the_snapshot_cannot_leak_even_if_it_is_wrong(self) -> None:
        """The selector re-asserts scope per record rather than trusting the snapshot."""
        registry = a_registry()
        record = await active(registry, await an_adapter("private-a", tenant_scope="tenant-a"))
        # Simulate a snapshot bug: the registry hands tenant B a record it should not.
        registry.servable = lambda tenant_id: (  # type: ignore[method-assign]
            record.model_copy(update={"status": AdapterStatus.ACTIVE}),
        )
        selection = await a_selector(registry).select(
            an_escalation_query(), principal("tenant-b"), max_adapters=1
        )
        assert selection.is_empty


class TestSelection:
    async def test_below_the_coverage_floor_returns_nothing(self) -> None:
        """Half-coverage is confident noise, worse than the absence it replaces."""
        registry = a_registry()
        await active(registry, await an_adapter("billing", topic="invoice refund billing cycle"))

        selection = await a_selector(registry).select(
            an_escalation_query(), principal("tenant-a"), max_adapters=2
        )
        assert selection.is_empty
        assert [r.adapter_id for r in selection.rejected_for_coverage] == ["billing"]

    async def test_single_best_keeps_one(self) -> None:
        registry = a_registry()
        await active(registry, await an_adapter("a1"))
        await active(registry, await an_adapter("a2"))

        selection = await a_selector(registry).select(
            an_escalation_query(), principal("tenant-a"), max_adapters=2
        )
        assert len(selection.adapters) == 1

    async def test_merging_modes_respect_the_concurrency_cap(self) -> None:
        registry = a_registry()
        for name in ("a1", "a2", "a3"):
            await active(registry, await an_adapter(name))
        selector = a_selector(
            registry, composition_mode=CompositionMode.WEIGHTED_MERGE, max_concurrent=2
        )

        selection = await selector.select(an_escalation_query(), principal("t"), max_adapters=4)
        assert len(selection.adapters) == 2
        assert selection.composition_mode is CompositionMode.WEIGHTED_MERGE

    async def test_another_base_model_version_is_never_selected(self) -> None:
        """A delta served against a different base degrades output with no error."""
        registry = a_registry()
        await active(registry, await an_adapter("old", base="base-v0"))

        selection = await a_selector(registry).select(
            an_escalation_query(), principal("t"), max_adapters=1
        )
        assert selection.is_empty

    async def test_another_embedding_model_is_never_compared(self) -> None:
        registry = a_registry()
        record = await an_adapter("a1")
        await active(registry, record.model_copy(update={"embedding_model_version": "v0"}))

        selection = await a_selector(registry).select(
            an_escalation_query(), principal("t"), max_adapters=1
        )
        assert selection.is_empty


class TestRevocation:
    async def test_erasing_a_document_revokes_and_evicts_every_adapter_trained_on_it(
        self,
    ) -> None:
        registry = a_registry()
        await active(registry, await an_adapter("a1", documents=("doc-erased", "doc-2")))
        await active(registry, await an_adapter("a2", documents=("doc-2",)))
        await registry.store.load("a1", "1")

        affected = await registry.revoke_documents(["doc-erased"])

        assert [r.adapter_id for r in affected] == ["a1"]
        assert (await registry.repository.get("a1", "1")).status is AdapterStatus.REVOKED
        assert not await registry.store.is_resident("a1", "1"), "evicted, not left to LRU"
        assert [r.adapter_id for r in registry.servable("t")] == ["a2"]

    async def test_a_revoked_adapter_cannot_be_loaded_by_a_stale_selection(self) -> None:
        """Revocation degrades to non-parametric, never to serving stale weights."""
        registry = a_registry()
        await active(registry, await an_adapter("a1"))
        await registry.revoke_documents(["doc-1"])

        with pytest.raises(AdapterLoadError):
            await registry.store.load("a1", "1")


class TestRouting:
    async def test_private_data_needs_a_tenant_scoped_adapter(self) -> None:
        from prag.core.models.query import FieldPrediction, KnowledgeRequirement, Strategy
        from prag.intelligence import UtilityStrategyRouter
        from tests.unit.test_intelligence import a_budget, a_policy

        registry = a_registry()
        await active(registry, await an_adapter("shared"))
        analysis = an_analysis().model_copy(
            update={
                "knowledge_requirements": {
                    KnowledgeRequirement.REQUIRES_EXTERNAL_KNOWLEDGE: FieldPrediction(
                        value=True, confidence=0.9
                    ),
                    KnowledgeRequirement.REQUIRES_PRIVATE_DATA: FieldPrediction(
                        value=True, confidence=0.9
                    ),
                }
            }
        )
        router = UtilityStrategyRouter(
            parametric_available=True,
            adapter_coverage=lambda t, d, scoped: registry.covers(t, d, tenant_scoped_only=scoped),
        )
        policy = a_policy(parametric=True)

        shared_only = router.route(analysis, principal("tenant-a"), a_budget(), policy)
        assert shared_only.eliminated[Strategy.PARAMETRIC] == (
            "requires_private_data_without_tenant_adapter"
        )

        await active(registry, await an_adapter("private-a", tenant_scope="tenant-a"))
        scoped = router.route(analysis, principal("tenant-a"), a_budget(), policy)
        assert Strategy.PARAMETRIC not in scoped.eliminated

    async def test_an_uncovered_domain_eliminates_parametric(self) -> None:
        from prag.core.models.query import Strategy
        from prag.intelligence import UtilityStrategyRouter
        from tests.unit.test_intelligence import a_budget, a_policy

        registry = a_registry()
        await active(registry, await an_adapter("billing", domain="billing"))
        router = UtilityStrategyRouter(
            parametric_available=True,
            adapter_coverage=lambda t, d, scoped: registry.covers(t, d, tenant_scoped_only=scoped),
        )
        decision = router.route(
            an_analysis(), principal("tenant-a"), a_budget(), a_policy(parametric=True)
        )
        assert decision.eliminated[Strategy.PARAMETRIC] == "no_adapter_covers_this_domain"
