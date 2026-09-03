"""Invariants the domain models are responsible for enforcing."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from prag.core.models.common import MemoryNamespace, Provenance, VolatilityClass
from prag.core.models.context import (
    ContextBundle,
    ContextRegion,
    CoverageWarning,
    RegionName,
)
from prag.core.models.fusion import (
    ConfidenceBand,
    ConfidenceBlock,
    ConflictEvent,
    ConflictKind,
    ConflictPosition,
    ConflictResolution,
    KnowledgeBasis,
    Stance,
)
from prag.core.models.generation import (
    AnswerEnvelope,
    Diagnostics,
    GroundingReport,
    ModelSpec,
)
from prag.core.models.identity import Principal, TenantPolicy, UtilityWeights
from prag.core.models.memory import MemoryItem, MemorySelector
from prag.core.models.parametric import (
    GLOBAL_TENANT_SCOPE,
    AdapterRecord,
    AdapterStatus,
    AdapterTier,
    ParametricEconomics,
)
from tests.fakes.sources import make_candidate


class TestPrincipal:
    def test_cache_discriminator_is_sorted(self) -> None:
        """Two principals with the same permissions must share a cache entry."""
        a = Principal(tenant_id="t", user_id="u", acl_hashes=("z", "a"))
        b = Principal(tenant_id="t", user_id="u2", acl_hashes=("a", "z"))
        assert a.cache_discriminator == b.cache_discriminator

    def test_tenant_leads_the_discriminator(self) -> None:
        a = Principal(tenant_id="t1", user_id="u", acl_hashes=("x",))
        b = Principal(tenant_id="t2", user_id="u", acl_hashes=("x",))
        assert a.cache_discriminator != b.cache_discriminator

    def test_is_frozen(self) -> None:
        """Recomputing permissions mid-request would let the filter and the recheck disagree."""
        principal = Principal(tenant_id="t", user_id="u")
        with pytest.raises(ValidationError):
            principal.tenant_id = "other"  # type: ignore[misc]


class TestUtilityWeights:
    def test_must_sum_to_one(self) -> None:
        with pytest.raises(ValidationError, match=r"sum to 1\.0"):
            UtilityWeights(quality=0.9, latency=0.9, cost=0.9)

    def test_valid_weights_are_accepted(self) -> None:
        assert UtilityWeights(quality=0.85, latency=0.05, cost=0.10).quality == 0.85


class TestTenantPolicy:
    def test_strict_mode_drives_conflict_abstention(self) -> None:
        weights = UtilityWeights(quality=0.6, latency=0.2, cost=0.2)
        strict = TenantPolicy(
            tenant_id="t", config_version="c", utility_weights=weights, strict_mode=True
        )
        lenient = strict.model_copy(update={"strict_mode": False})

        assert strict.abstains_on_irreconcilable_conflict
        assert not lenient.abstains_on_irreconcilable_conflict


class TestCandidate:
    def test_effective_score_prefers_the_latest_stage(self) -> None:
        """Callers wanting an ordering must not get None because a stage was skipped."""
        candidate = make_candidate("t", candidate_id="c1", score=0.4)
        assert candidate.effective_score == 0.4

        fused = candidate.model_copy(update={"fused_score": 0.7})
        assert fused.effective_score == 0.7

        reranked = fused.model_copy(update={"rerank_score": 0.9})
        assert reranked.effective_score == 0.9

    def test_context_text_prefers_the_parent(self) -> None:
        """Parent-child retrieval: match the precise child, show the surrounding parent."""
        child = make_candidate("a narrow sentence", candidate_id="c1")
        assert child.context_text == "a narrow sentence"

        with_parent = child.model_copy(update={"parent_text": "the whole section"})
        assert with_parent.context_text == "the whole section"


class TestAdapterIsolation:
    def _record(self, *, scope: str, status: AdapterStatus) -> AdapterRecord:
        return AdapterRecord(
            adapter_id="a1",
            version="v1",
            tier=AdapterTier.CLUSTER_KNOWLEDGE,
            base_model_id="m",
            base_model_version="1",
            tenant_scope=scope,
            rank=8,
            alpha=16.0,
            status=status,
            created_at_ms=0,
        )

    def test_tenant_exclusive_adapter_serves_only_its_tenant(self) -> None:
        """The check that cannot be walked back if it is wrong."""
        record = self._record(scope="tenant-a", status=AdapterStatus.ACTIVE)
        assert record.servable_for("tenant-a")
        assert not record.servable_for("tenant-b")

    def test_global_adapter_serves_anyone(self) -> None:
        record = self._record(scope=GLOBAL_TENANT_SCOPE, status=AdapterStatus.ACTIVE)
        assert record.is_global
        assert record.servable_for("tenant-a")
        assert record.servable_for("tenant-b")

    @pytest.mark.parametrize(
        "status",
        [
            AdapterStatus.CANDIDATE,
            AdapterStatus.SHADOW,
            AdapterStatus.DEPRECATED,
            AdapterStatus.REVOKED,
        ],
    )
    def test_only_active_adapters_serve(self, status: AdapterStatus) -> None:
        """Revoked especially: serving erased knowledge is not a degraded state."""
        record = self._record(scope=GLOBAL_TENANT_SCOPE, status=status)
        assert not record.servable_for("tenant-a")


class TestParametricEconomics:
    def _economics(self, **overrides: float) -> ParametricEconomics:
        base = {
            "training_gpu_cost_usd": 40.0,
            "storage_cost_per_period_usd": 2.0,
            "expected_queries_per_period": 1000.0,
            "evidence_tokens_displaced": 6000,
            "prefill_cost_per_token_usd": 0.00002,
            "retrieval_infra_cost_per_query_usd": 0.0004,
            "cluster_coherence_score": 0.7,
        }
        return ParametricEconomics(**{**base, **overrides})  # type: ignore[arg-type]

    def test_amortized_cost_falls_with_volume(self) -> None:
        low = self._economics(expected_queries_per_period=100.0)
        high = self._economics(expected_queries_per_period=10_000.0)
        assert high.amortized_parametric_cost_usd < low.amortized_parametric_cost_usd

    def test_zero_volume_is_infinite_not_an_error(self) -> None:
        """No expected traffic is a legitimate input that should fail the gate, not crash it."""
        economics = self._economics(expected_queries_per_period=0.0)
        assert economics.amortized_parametric_cost_usd == float("inf")

    def test_nonparametric_cost_includes_prefill_and_infra(self) -> None:
        economics = self._economics()
        assert economics.per_query_nonparametric_cost_usd == pytest.approx(6000 * 0.00002 + 0.0004)


class TestMemoryItem:
    def test_long_term_refuses_model_generated(self) -> None:
        item = MemoryItem(
            item_id="m1",
            namespace=MemoryNamespace.LONG_TERM,
            text="probably prefers dark mode",
            provenance=Provenance.MODEL_GENERATED,
            created_at_ms=0,
        )
        assert not item.persistable

    def test_session_accepts_anything(self) -> None:
        """Session memory is bounded by the session; the durable tier is the one that gates."""
        item = MemoryItem(
            item_id="m1",
            namespace=MemoryNamespace.SESSION,
            text="derived mid-conversation",
            provenance=Provenance.MODEL_GENERATED,
            created_at_ms=0,
        )
        assert item.persistable

    def test_empty_selector_is_detectable(self) -> None:
        assert MemorySelector().is_empty
        assert not MemorySelector(session_id="s").is_empty


class TestConflictEvent:
    def test_authority_delta_measures_the_spread(self) -> None:
        """Small spread means comparable authority, which is the condition for surfacing."""
        event = ConflictEvent(
            conflict_id="k1",
            kind=ConflictKind.SOURCE_VS_SOURCE,
            claim="the retention window is 30 days",
            positions=(
                ConflictPosition(origin_id="kb.policies", stance=Stance.SUPPORTS, authority=0.9),
                ConflictPosition(origin_id="kb.wiki", stance=Stance.CONTRADICTS, authority=0.4),
            ),
            resolution=ConflictResolution.AUTHORITY_WINS,
        )
        assert event.authority_delta == pytest.approx(0.5)

    def test_no_positions_is_zero_not_an_error(self) -> None:
        event = ConflictEvent(
            conflict_id="k1",
            kind=ConflictKind.VERSION_VS_VERSION,
            claim="x",
            positions=(),
            resolution=ConflictResolution.SURFACED,
        )
        assert event.authority_delta == 0.0


class TestContextBundle:
    def _bundle(self, **overrides: object) -> ContextBundle:
        base: dict[str, object] = {
            "bundle_id": "ctx-1",
            "regions": (
                ContextRegion(
                    name=RegionName.EVIDENCE,
                    allocated_tokens=4000,
                    used_tokens=3200,
                    trimmable=True,
                ),
                ContextRegion(
                    name=RegionName.QUERY,
                    allocated_tokens=200,
                    used_tokens=48,
                    trimmable=False,
                ),
            ),
            "rendered_prompt_hash": "abc123",
        }
        return ContextBundle(**{**base, **overrides})  # type: ignore[arg-type]

    def test_totals_and_lookup(self) -> None:
        bundle = self._bundle()
        assert bundle.total_used_tokens == 3248
        assert bundle.region(RegionName.EVIDENCE) is not None
        assert bundle.region(RegionName.TOOLS) is None

    def test_region_overflow_is_detectable(self) -> None:
        region = ContextRegion(
            name=RegionName.EVIDENCE, allocated_tokens=100, used_tokens=150, trimmable=True
        )
        assert region.overflowed
        assert region.headroom_tokens == 0

    def test_independent_evidence_is_counted_separately(self) -> None:
        """Ten groups from one document are one source, and counting them as ten is how a
        single stale page becomes a consensus."""
        from prag.core.models.retrieval import EvidenceGroup

        def group(gid: str, *, root: str, independent: bool) -> EvidenceGroup:
            candidate = make_candidate("t", candidate_id=gid, lineage_root=root)
            return EvidenceGroup(
                group_id=gid,
                members=(candidate,),
                representative=candidate,
                lineage_root=root,
                authority=0.5,
                freshness=0.9,
                independent=independent,
                citation_marker=f"E{gid}",
            )

        bundle = self._bundle(
            evidence=(
                group("1", root="r1", independent=True),
                group("2", root="r1", independent=False),
                group("3", root="r2", independent=True),
            )
        )
        assert len(bundle.evidence) == 3
        assert bundle.independent_evidence_count == 2


class TestAnswerEnvelope:
    def _envelope(self, **overrides: object) -> AnswerEnvelope:
        base: dict[str, object] = {
            "request_id": "req-1",
            "answer": "Escalate to the on-call lead within 15 minutes.",
            "confidence": ConfidenceBlock(
                score=0.88, band=ConfidenceBand.HIGH, basis=KnowledgeBasis.RETRIEVED_EVIDENCE
            ),
            "grounding": GroundingReport(claims_total=3, claims_cited=3, claims_unsourced=0),
            "diagnostics": Diagnostics(
                route_class="hybrid",
                strategy="NON_PARAMETRIC",
                model_id="mid.instruct",
                model_version="1",
                total_ms=1420,
            ),
        }
        return AnswerEnvelope(**{**base, **overrides})  # type: ignore[arg-type]

    def test_clean_answer_is_cacheable(self) -> None:
        envelope = self._envelope()
        assert not envelope.has_warnings
        assert envelope.cacheable

    def test_coverage_warning_blocks_caching(self) -> None:
        envelope = self._envelope(coverage_warning=CoverageWarning(coverage=0.4))
        assert envelope.has_warnings
        assert not envelope.cacheable

    def test_unsourced_claims_block_caching(self) -> None:
        """Caching an answer with an unsupported claim republishes it indefinitely."""
        envelope = self._envelope(
            grounding=GroundingReport(claims_total=3, claims_cited=2, claims_unsourced=1)
        )
        assert not envelope.cacheable

    def test_groundedness_of_a_claimless_answer_is_one(self) -> None:
        """A formatting request has nothing to be unfaithful about.

        Scoring it zero would drag the aggregate faithfulness metric down for the wrong reason.
        """
        report = GroundingReport(claims_total=0, claims_cited=0, claims_unsourced=0)
        assert report.groundedness == 1.0

    def test_unknown_fields_are_preserved(self) -> None:
        """Forward compatibility: a rolling deploy must not drop the newer version's data."""
        envelope = AnswerEnvelope.model_validate(
            {**self._envelope().model_dump(), "future_field": "kept"}
        )
        assert envelope.model_dump()["future_field"] == "kept"


class TestModelSpec:
    def test_cost_estimate_uses_the_routers_numbers(self) -> None:
        spec = ModelSpec(
            model_id="m",
            model_version="1",
            provider_id="p",
            profile="grounded_extraction",
            context_window=8192,
            cost_per_1k_in=0.002,
            cost_per_1k_out=0.006,
        )
        assert spec.estimated_cost_usd(1000, 500) == pytest.approx(0.002 + 0.003)


class TestValidationBounds:
    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_probabilities_are_bounded(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            ConfidenceBlock(score=bad, band=ConfidenceBand.HIGH, basis=KnowledgeBasis.PARAMETRIC)

    def test_volatility_class_is_constrained(self) -> None:
        assert VolatilityClass("static") is VolatilityClass.STATIC
        with pytest.raises(ValueError, match="not a valid"):
            VolatilityClass("occasionally")
