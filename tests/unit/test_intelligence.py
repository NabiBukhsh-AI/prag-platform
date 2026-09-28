"""The T0 rules tier, the analyzer cascade, the strategy router, and query transforms."""

from __future__ import annotations

import random

import pytest

from prag.core.models.common import Deadline, SlaTier, VolatilityClass
from prag.core.models.identity import Budget, Principal, TenantPolicy, UtilityWeights
from prag.core.models.query import KnowledgeRequirement, SessionContext, Strategy
from prag.intelligence import (
    CascadeQueryAnalyzer,
    CoreferenceTransformer,
    DecompositionTransformer,
    ExpansionTransformer,
    QualityTable,
    RewriteTransformer,
    UtilityStrategyRouter,
    apply_transforms,
    classify_rules,
)


def a_principal(tier: SlaTier = SlaTier.STANDARD) -> Principal:
    return Principal(tenant_id="tenant-a", user_id="u1", sla_tier=tier)


def a_policy(*, parametric: bool = False, tier: str = "standard") -> TenantPolicy:
    weights = {
        "standard": UtilityWeights(quality=0.60, latency=0.20, cost=0.20),
        "interactive": UtilityWeights(quality=0.50, latency=0.35, cost=0.15),
        "high_stakes": UtilityWeights(quality=0.85, latency=0.05, cost=0.10),
    }[tier]
    return TenantPolicy(
        tenant_id="tenant-a",
        config_version="c1",
        utility_weights=weights,
        parametric_enabled=parametric,
    )


def a_budget(*, wall_ms: int = 10_000, usd: float = 0.05) -> Budget:
    return Budget(
        wall_ms_total=wall_ms,
        wall_ms_remaining=wall_ms,
        usd_total=usd,
        max_tokens_in=8_000,
        max_tokens_out=1_024,
    )


async def analyze(query: str, *, session: SessionContext | None = None):
    return await CascadeQueryAnalyzer().analyze(
        query, session, a_principal(), Deadline.in_ms(200, label="t")
    )


class TestRulesTier:
    @pytest.mark.parametrize(
        "query",
        ["what is 17% of 340", "calculate 5 * 12", "what is 40 percent of 900", "12 divided by 4"],
    )
    def test_arithmetic_needs_no_retrieval(self, query: str) -> None:
        """Retrieving for arithmetic is expensive theatre, and the evidence dilutes attention."""
        verdict = classify_rules(query)
        assert "arithmetic" in verdict.fired
        assert verdict.requires_external_knowledge is False

    def test_arithmetic_is_recognised_mid_sentence(self) -> None:
        """The canonical arithmetic query opens with a question word, not a digit."""
        assert "arithmetic" in classify_rules("what is 17% of 340").fired

    def test_a_greeting_is_fully_decided(self) -> None:
        verdict = classify_rules("hi")
        assert verdict.is_decisive
        assert verdict.requires_external_knowledge is False
        assert verdict.intent == "conversational"

    def test_live_data_forbids_the_parametric_route(self) -> None:
        """Weights are a snapshot, and a snapshot answering "right now" reads as right."""
        verdict = classify_rules("what is the current status of the payments service")
        assert verdict.requires_live_data is True
        assert verdict.volatility is VolatilityClass.REALTIME

    def test_exact_quotation_is_detected(self) -> None:
        verdict = classify_rules("quote the exact wording of the retention clause")
        assert verdict.requires_exact_quotation is True
        assert verdict.requires_citation is True

    def test_possessives_mark_a_query_as_private(self) -> None:
        assert classify_rules("what is our escalation policy").requires_private_data is True

    def test_comparison_is_multi_hop(self) -> None:
        assert classify_rules("compare sev-1 and sev-2 targets").multi_hop is True

    def test_an_ordinary_question_settles_nothing(self) -> None:
        """A rule that half-matches must stay silent and let the next tier decide.

        The cascade's value is T0 being right when it speaks, not speaking often.
        """
        verdict = classify_rules("what is the retention window")
        assert verdict.fired == ()
        assert verdict.coverage == 0.0
        assert not verdict.is_decisive

    def test_rules_are_deterministic(self) -> None:
        assert classify_rules("what is our policy") == classify_rules("what is our policy")


class TestAnalyzer:
    async def test_reports_the_tier_honestly(self) -> None:
        """A trace claiming T1 ran would attribute a default's decision to a missing model."""
        analysis = await analyze("what is the retention window")
        assert analysis.classifier_tier_used == "T0"

    async def test_uncertainty_tracks_what_the_rules_left_undone(self) -> None:
        """Reporting false confidence suppresses exactly the hedging uncertainty triggers."""
        decided = await analyze("hi")
        undecided = await analyze("what is the retention window")

        assert decided.router_uncertainty < undecided.router_uncertainty
        assert undecided.router_uncertainty == pytest.approx(1.0)

    async def test_a_rule_verdict_raises_its_field_confidence(self) -> None:
        analysis = await analyze("what is our escalation policy")
        assert analysis.confidence_in(KnowledgeRequirement.REQUIRES_PRIVATE_DATA) > 0.8

    async def test_a_defaulted_field_sits_at_the_coin_flip_line(self) -> None:
        """Rounding a default up would launder a guess into a finding."""
        analysis = await analyze("what is the retention window")
        assert analysis.confidence_in(KnowledgeRequirement.REQUIRES_LIVE_DATA) == pytest.approx(0.5)

    async def test_half_life_follows_volatility(self) -> None:
        """Absolute age is meaningless without it: freshness scoring divides by this."""
        stable = await analyze("what is the retention window")
        live = await analyze("what is the current status right now")

        assert (
            live.temporality.estimated_half_life_days < stable.temporality.estimated_half_life_days
        )

    async def test_identifiers_are_extracted_as_entities(self) -> None:
        """Identifiers are where dense retrieval fails and a lexical leg earns its place."""
        analysis = await analyze("what happens on a SEV-1 incident")
        assert any("sev" in e.text.lower() for e in analysis.structure.entities)

    async def test_a_pronoun_without_context_is_ambiguous(self) -> None:
        analysis = await analyze("does it apply to them")
        assert analysis.ambiguity.is_ambiguous
        assert analysis.ambiguity.clarification_candidates

    async def test_a_follow_up_pronoun_is_not_ambiguous(self) -> None:
        """It has an antecedent; the coreference transform resolves it rather than asking."""
        session = SessionContext(session_id="s1", turn_index=2)
        analysis = await analyze("does it apply to them", session=session)
        assert not analysis.ambiguity.is_ambiguous

    async def test_escalation_is_counted(self) -> None:
        """Observable before the tiers exist, which is what says whether building them pays."""
        analyzer = CascadeQueryAnalyzer()
        await analyzer.analyze(
            "what is the retention window", None, a_principal(), Deadline.in_ms(200)
        )
        assert analyzer.escalations == 1


class TestHardConstraints:
    """Stage one can only eliminate. A high score must never resurrect a disqualified route."""

    async def test_parametric_is_off_when_the_tenant_forbids_it(self) -> None:
        analysis = await analyze("what is the retention window")
        decision = UtilityStrategyRouter().route(
            analysis, a_principal(), a_budget(), a_policy(parametric=False)
        )
        assert decision.eliminated[Strategy.PARAMETRIC] == "tenant_policy_forbids_parametric"
        assert decision.strategy is Strategy.NON_PARAMETRIC

    async def test_live_data_eliminates_parametric(self) -> None:
        analysis = await analyze("what is the current status right now")
        decision = UtilityStrategyRouter(parametric_available=True).route(
            analysis, a_principal(), a_budget(), a_policy(parametric=True)
        )
        assert decision.eliminated[Strategy.PARAMETRIC] == "requires_live_data"

    async def test_exact_quotation_eliminates_parametric(self) -> None:
        """Weights reproduce meaning, not spans."""
        analysis = await analyze("quote the exact wording of the clause")
        decision = UtilityStrategyRouter(parametric_available=True).route(
            analysis, a_principal(), a_budget(), a_policy(parametric=True)
        )
        assert decision.eliminated[Strategy.PARAMETRIC] == "requires_exact_quotation"

    async def test_private_data_eliminates_parametric(self) -> None:
        """The failure that cannot be walked back: parameters cannot be filtered per request."""
        analysis = await analyze("what is our internal escalation policy")
        decision = UtilityStrategyRouter(parametric_available=True).route(
            analysis, a_principal(), a_budget(), a_policy(parametric=True)
        )
        assert "private_data" in decision.eliminated[Strategy.PARAMETRIC]

    async def test_hybrid_inherits_the_parametric_disqualification(self) -> None:
        """Hybrid contains the parametric route, so it would reintroduce it under another name."""
        analysis = await analyze("what is the current status right now")
        decision = UtilityStrategyRouter(parametric_available=True).route(
            analysis, a_principal(), a_budget(), a_policy(parametric=True)
        )
        assert Strategy.HYBRID in decision.eliminated

    async def test_arithmetic_eliminates_retrieval(self) -> None:
        analysis = await analyze("what is 17% of 340")
        decision = UtilityStrategyRouter().route(analysis, a_principal(), a_budget(), a_policy())
        assert decision.eliminated[Strategy.NON_PARAMETRIC] == "no_external_knowledge_required"

    async def test_every_elimination_records_a_reason(self) -> None:
        """A trace showing only the winner cannot answer "why didn't it retrieve"."""
        analysis = await analyze("what is 17% of 340")
        decision = UtilityStrategyRouter().route(analysis, a_principal(), a_budget(), a_policy())
        assert all(reason for reason in decision.eliminated.values())


class TestUtilityScoring:
    async def test_survivors_are_scored(self) -> None:
        analysis = await analyze("what is the retention window")
        decision = UtilityStrategyRouter().route(analysis, a_principal(), a_budget(), a_policy())
        assert decision.utility_scores
        assert Strategy.PARAMETRIC not in decision.utility_scores, "eliminated routes are unscored"

    async def test_a_tight_budget_penalises_the_slow_route(self) -> None:
        """Normalising by the budget is what lets one weight set serve every tier."""
        analysis = await analyze("what is the retention window")
        router = UtilityStrategyRouter(parametric_available=True)
        policy = a_policy(parametric=True, tier="interactive")

        generous = router.route(analysis, a_principal(), a_budget(wall_ms=30_000), policy)
        tight = router.route(analysis, a_principal(), a_budget(wall_ms=600), policy)

        assert (
            tight.utility_scores[Strategy.NON_PARAMETRIC]
            < generous.utility_scores[Strategy.NON_PARAMETRIC]
        )

    async def test_routing_is_deterministic_without_exploration(self) -> None:
        analysis = await analyze("what is the retention window")
        router = UtilityStrategyRouter()
        first = router.route(analysis, a_principal(), a_budget(), a_policy())
        second = router.route(analysis, a_principal(), a_budget(), a_policy())
        assert first.strategy is second.strategy

    async def test_exploration_routes_against_the_argmax(self) -> None:
        """Without a counterfactual sample the table only confirms what it already believes."""
        analysis = await analyze("what is the retention window")
        router = UtilityStrategyRouter(
            parametric_available=True, exploration_fraction=1.0, rng=random.Random(0)
        )
        decision = router.route(analysis, a_principal(), a_budget(), a_policy(parametric=True))
        assert decision.exploration

    async def test_high_uncertainty_hedges(self) -> None:
        """Hedging costs a retrieval; not hedging costs the answer when the router was wrong."""
        analysis = await analyze("what is the retention window")
        assert analysis.router_uncertainty > 0.35

        router = UtilityStrategyRouter(
            parametric_available=True,
            profiles={
                Strategy.PARAMETRIC: __import__(
                    "prag.intelligence.strategy_router", fromlist=["StrategyProfile"]
                ).StrategyProfile(expected_latency_ms=1.0, expected_cost_usd=0.0),
                Strategy.NON_PARAMETRIC: __import__(
                    "prag.intelligence.strategy_router", fromlist=["StrategyProfile"]
                ).StrategyProfile(expected_latency_ms=9_000.0, expected_cost_usd=0.04),
                Strategy.HYBRID: __import__(
                    "prag.intelligence.strategy_router", fromlist=["StrategyProfile"]
                ).StrategyProfile(expected_latency_ms=9_500.0, expected_cost_usd=0.045),
            },
        )
        decision = router.route(
            analysis, a_principal(), a_budget(), a_policy(parametric=True, tier="interactive")
        )

        assert decision.strategy is Strategy.PARAMETRIC
        assert decision.hedged, "an uncertain parametric lean must still run retrieval"
        assert decision.speculative_retrieval_started


class TestQualityTable:
    def test_a_cold_bucket_returns_the_prior(self) -> None:
        """Scoring an unobserved strategy zero would make the table self-confirming."""
        table = QualityTable(prior_quality=0.62)
        assert table.expected_quality(
            domain="ops", complexity="simple_factual", intent="lookup", strategy=Strategy.PARAMETRIC
        ) == pytest.approx(0.62)

    def test_observations_shift_the_estimate(self) -> None:
        table = QualityTable(prior_quality=0.5, prior_weight=2.0)
        for _ in range(20):
            table.record(
                domain="ops",
                complexity="simple_factual",
                intent="lookup",
                strategy=Strategy.NON_PARAMETRIC,
                quality=0.95,
            )
        estimate = table.expected_quality(
            domain="ops",
            complexity="simple_factual",
            intent="lookup",
            strategy=Strategy.NON_PARAMETRIC,
        )
        assert estimate > 0.85

    def test_a_small_sample_is_shrunk_toward_the_prior(self) -> None:
        """The safer direction: a strategy's first few requests may simply have been lucky."""
        table = QualityTable(prior_quality=0.5, prior_weight=20.0)
        table.record(
            domain="ops",
            complexity="simple_factual",
            intent="lookup",
            strategy=Strategy.PARAMETRIC,
            quality=1.0,
        )
        estimate = table.expected_quality(
            domain="ops", complexity="simple_factual", intent="lookup", strategy=Strategy.PARAMETRIC
        )
        assert 0.5 < estimate < 0.6


class TestTransforms:
    async def test_rewrite_strips_framing_not_meaning(self) -> None:
        analysis = await analyze("could you please tell me what the retention window is thanks")
        variants = await RewriteTransformer().transform(analysis, Deadline.in_ms(60))

        assert variants.rewritten is not None
        assert "retention window" in variants.rewritten
        assert "please" not in variants.rewritten.lower()

    async def test_expansion_adds_aliases_for_the_lexical_leg(self) -> None:
        """ "sev1" and "sev-1" are neighbours to a human and unrelated tokens to BM25."""
        analysis = await analyze("what is the sev-1 escalation target")
        variants = await ExpansionTransformer().transform(analysis, Deadline.in_ms(5))

        assert variants.expanded is not None
        assert "sev1" in variants.expanded

    async def test_coreference_declines_without_an_antecedent(self) -> None:
        """A wrong antecedent retrieves confidently for the wrong subject."""
        analysis = await analyze("does it apply")
        variants = await CoreferenceTransformer().transform(analysis, Deadline.in_ms(25))
        assert variants.coreference_resolved is None

    async def test_decomposition_splits_a_comparison(self) -> None:
        """Independent hops execute in parallel, which is the latency win."""
        analysis = await analyze("compare sev-1 and sev-2 response targets")
        variants = await DecompositionTransformer().transform(analysis, Deadline.in_ms(150))

        assert len(variants.sub_queries) == 2
        assert all(sq.depends_on == () for sq in variants.sub_queries), "independent hops"

    async def test_gating_keeps_transforms_from_running_needlessly(self) -> None:
        """A rewrite on a clear query costs latency and can lose precision."""
        analysis = await analyze("what is the documented incident retention period")
        assert not DecompositionTransformer().applies_to(analysis)
        assert not CoreferenceTransformer().applies_to(analysis)

    async def test_apply_transforms_merges_variants(self) -> None:
        analysis = await analyze("please compare sev-1 and sev-2 response targets")
        variants = await apply_transforms(analysis, Deadline.in_ms(500))

        assert variants.raw
        assert variants.expanded or variants.rewritten
        assert variants.sub_queries

    async def test_a_failing_transform_is_skipped_not_fatal(self) -> None:
        """Every transform is an optimisation; the raw query is always retrievable."""

        class Exploding:
            name = "exploding"

            def applies_to(self, analysis: object) -> bool:
                return True

            async def transform(self, analysis: object, deadline: object) -> object:
                raise RuntimeError("boom")

        analysis = await analyze("what is the retention window")
        variants = await apply_transforms(
            analysis, Deadline.in_ms(200), transformers=(Exploding(),)
        )
        assert variants.raw == analysis.normalized_query

    async def test_an_expired_deadline_stops_the_chain(self) -> None:
        analysis = await analyze("please compare sev-1 and sev-2")
        variants = await apply_transforms(analysis, Deadline.in_ms(0))
        assert variants.raw == analysis.normalized_query
        assert variants.sub_queries == ()
