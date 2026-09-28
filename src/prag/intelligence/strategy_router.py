"""Strategy selection: parametric, non-parametric, or hybrid.

Two stages, and the order is the whole design.

**Stage one applies hard constraints and can only eliminate.** A strategy disqualified here
cannot be selected however well it scores, because the catastrophic failures are all
disqualifications. Live data cannot come from weights that were frozen last month. Exact
quotation cannot come from a model that reproduces meaning rather than spans. A question about
the tenant's own data cannot be answered from general knowledge — that is the single most
damaging output this system can produce, because it looks exactly like a correct one.

**Stage two scores the survivors on expected utility**, using the tenant's SLA weights over
quality, latency and cost. Expected quality comes from a table of historical outcomes rather
than from a heuristic, which makes routing empirically driven and gives a natural place to
measure regret.

Every eliminated strategy is recorded with its reason. A trace showing only the winner cannot
answer "why didn't it retrieve", which is the question actually asked during an incident.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from prag.core.models.query import KnowledgeRequirement, Strategy, StrategyDecision

if TYPE_CHECKING:
    from collections.abc import Callable

    from prag.core.models.identity import Budget, Principal, TenantPolicy
    from prag.core.models.query import QueryAnalysis

    #: ``(tenant_id, domain, tenant_scoped_only) -> covered``. Synchronous by contract: it runs
    #: on every request, so it reads a snapshot rather than a database.
    AdapterCoverage = Callable[[str, str, bool], bool]

__all__ = ["QualityTable", "StrategyProfile", "UtilityStrategyRouter"]


@dataclass(frozen=True, slots=True)
class StrategyProfile:
    """What a strategy is expected to cost, before quality is considered.

    Latency and cost are estimates the deployment measures and updates. They are stated per
    strategy rather than per request because the routing decision happens before anything has
    run, and an estimate that needed the outcome would be useless at the moment of choosing.
    """

    expected_latency_ms: float
    expected_cost_usd: float


#: Shipped estimates. Parametric is cheap and fast per query because its knowledge cost is
#: amortised into training; non-parametric pays prefill on every request. Hybrid pays both.
DEFAULT_PROFILES: dict[Strategy, StrategyProfile] = {
    Strategy.PARAMETRIC: StrategyProfile(expected_latency_ms=420.0, expected_cost_usd=0.0016),
    Strategy.NON_PARAMETRIC: StrategyProfile(expected_latency_ms=1_150.0, expected_cost_usd=0.0043),
    Strategy.HYBRID: StrategyProfile(expected_latency_ms=1_380.0, expected_cost_usd=0.0057),
}


class QualityTable:
    """Historical per-strategy quality, bucketed by ``(domain, complexity, intent)``.

    Empirical rather than heuristic: the numbers come from evaluation outcomes, refreshed
    nightly. A cold bucket falls back to a Bayesian prior instead of to zero, because a strategy
    with no observations is unknown rather than bad, and scoring it zero would make the table
    self-confirming — the strategy would never be chosen, so it would never accumulate the
    evidence that might have vindicated it.
    """

    def __init__(
        self,
        observations: dict[tuple[str, str, str, Strategy], tuple[float, int]] | None = None,
        *,
        prior_quality: float = 0.62,
        prior_weight: float = 20.0,
    ) -> None:
        #: Bucket to (mean quality, observation count).
        self._observations = dict(observations or {})
        self._prior_quality = prior_quality
        #: How many observations it takes for measured quality to outweigh the prior. Higher
        #: means slower to trust a small sample, which is the safer direction when a strategy's
        #: first few requests may have been lucky.
        self._prior_weight = prior_weight

    def expected_quality(
        self, *, domain: str, complexity: str, intent: str, strategy: Strategy
    ) -> float:
        """Shrink the bucket's measured quality toward the prior by its sample size."""
        observed, count = self._observations.get(
            (domain, complexity, intent, strategy), (self._prior_quality, 0)
        )
        weight = count / (count + self._prior_weight)
        return weight * observed + (1.0 - weight) * self._prior_quality

    def record(
        self,
        *,
        domain: str,
        complexity: str,
        intent: str,
        strategy: Strategy,
        quality: float,
    ) -> None:
        """Fold one outcome into its bucket, as a running mean."""
        key = (domain, complexity, intent, strategy)
        mean, count = self._observations.get(key, (self._prior_quality, 0))
        self._observations[key] = ((mean * count + quality) / (count + 1), count + 1)

    @property
    def bucket_count(self) -> int:
        return len(self._observations)


class UtilityStrategyRouter:
    """Hard constraints, then expected-utility scoring over what survives."""

    def __init__(
        self,
        *,
        quality_table: QualityTable | None = None,
        profiles: dict[Strategy, StrategyProfile] | None = None,
        exploration_fraction: float = 0.0,
        hedge_above_uncertainty: float = 0.35,
        parametric_available: bool = False,
        adapter_coverage: AdapterCoverage | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._quality = quality_table or QualityTable()
        self._profiles = profiles or DEFAULT_PROFILES
        self._exploration = exploration_fraction
        self._hedge_above = hedge_above_uncertainty
        #: Whether the parametric tier is switched on at all. Distinct from the tenant's policy,
        #: and from coverage: the tier may be on while this tenant has no adapter for anything.
        self._parametric_available = parametric_available
        #: Per-tenant, per-domain coverage. Without it, "available" is taken to cover every
        #: domain except private data, which needs a tenant-scoped adapter the router cannot see.
        self._coverage = adapter_coverage
        self._rng = rng or random.Random()

    def route(
        self,
        analysis: QueryAnalysis,
        principal: Principal,
        budget: Budget,
        policy: TenantPolicy,
    ) -> StrategyDecision:
        eliminated = self._hard_constraints(analysis, policy, principal.tenant_id)
        survivors = [s for s in Strategy if s not in eliminated]

        if not survivors:
            # Every strategy disqualified. Non-parametric is the honest fallback: it is the one
            # that can still abstain with evidence of having tried, and abstaining is a better
            # outcome than answering from a route that was ruled out.
            return StrategyDecision(
                strategy=Strategy.NON_PARAMETRIC,
                eliminated=eliminated,
                hedged=True,
            )

        scores = {
            strategy: self._utility(strategy, analysis, budget, policy) for strategy in survivors
        }
        best = max(scores, key=lambda s: (scores[s], s.value))

        # A small fraction of traffic is routed against the argmax on purpose. Those requests
        # are the counterfactual sample that keeps the quality table honest — without them it
        # only ever confirms what it already believes, and routing regret is unmeasurable.
        exploring = len(survivors) > 1 and self._rng.random() < self._exploration
        if exploring:
            best = min(scores, key=lambda s: (scores[s], s.value))

        # High uncertainty means retrieval runs regardless of a parametric lean, so the more
        # expensive path is available if fusion turns out to need it. Hedging costs a retrieval;
        # not hedging costs the answer when the router was wrong.
        hedged = (
            analysis.router_uncertainty > self._hedge_above
            and best is Strategy.PARAMETRIC
            and Strategy.NON_PARAMETRIC not in eliminated
        )

        return StrategyDecision(
            strategy=best,
            utility_scores=scores,
            eliminated=eliminated,
            hedged=hedged,
            speculative_retrieval_started=hedged,
            exploration=exploring,
        )

    def _hard_constraints(
        self, analysis: QueryAnalysis, policy: TenantPolicy, tenant_id: str
    ) -> dict[Strategy, str]:
        """Eliminate strategies that cannot serve this query, with the reason for each.

        Reasons are stable strings rather than prose: they land in traces and in regret
        analysis, and a reason that is rephrased between versions cannot be grouped over time.
        """
        eliminated: dict[Strategy, str] = {}
        domain = str(analysis.domain.value)

        def covered(tenant_scoped_only: bool) -> bool:
            if self._coverage is None:
                return not tenant_scoped_only
            return self._coverage(tenant_id, domain, tenant_scoped_only)

        if not policy.parametric_enabled:
            eliminated[Strategy.PARAMETRIC] = "tenant_policy_forbids_parametric"
        elif not self._parametric_available or not covered(False):
            eliminated[Strategy.PARAMETRIC] = "no_adapter_covers_this_domain"
        elif analysis.requires(KnowledgeRequirement.REQUIRES_LIVE_DATA):
            # Weights are a snapshot. A snapshot answering "what is it right now" is wrong in a
            # way that reads as right, which is worse than declining.
            eliminated[Strategy.PARAMETRIC] = "requires_live_data"
        elif analysis.requires(KnowledgeRequirement.REQUIRES_EXACT_QUOTATION):
            eliminated[Strategy.PARAMETRIC] = "requires_exact_quotation"
        elif analysis.requires(KnowledgeRequirement.REQUIRES_PRIVATE_DATA) and not covered(True):
            # Parameters cannot be filtered per request, so private data is only answerable from
            # an adapter scoped to this tenant. A global adapter cannot know the tenant's data,
            # and a confident general answer to a question about it is the most damaging
            # failure this system can produce.
            eliminated[Strategy.PARAMETRIC] = "requires_private_data_without_tenant_adapter"

        if Strategy.PARAMETRIC in eliminated:
            # Hybrid contains the parametric route, so whatever disqualifies one disqualifies
            # the other. Letting hybrid through would reintroduce the eliminated path under a
            # different name.
            eliminated[Strategy.HYBRID] = eliminated[Strategy.PARAMETRIC]

        if not analysis.requires(KnowledgeRequirement.REQUIRES_EXTERNAL_KNOWLEDGE):
            # Arithmetic, transformation of user-supplied text, and follow-ups answerable from
            # session context. Retrieving for these is expensive theatre, and the evidence it
            # returns dilutes attention on a task that needed none.
            eliminated[Strategy.NON_PARAMETRIC] = "no_external_knowledge_required"

        return eliminated

    def _utility(
        self,
        strategy: Strategy,
        analysis: QueryAnalysis,
        budget: Budget,
        policy: TenantPolicy,
    ) -> float:
        """Expected utility: quality, minus latency and cost, normalised by their budgets.

        Normalising by the budget rather than by an absolute is what lets one weight set serve
        every tier. A 400 ms strategy is cheap against a 30 s batch budget and expensive against
        a 650 ms interactive one, and the same number should not mean both.
        """
        weights = policy.utility_weights
        profile = self._profiles[strategy]

        quality = self._quality.expected_quality(
            domain=str(analysis.domain.value),
            complexity=str(analysis.complexity.value),
            intent=str(analysis.intent.value),
            strategy=strategy,
        )
        latency_fraction = profile.expected_latency_ms / max(1, budget.wall_ms_remaining)
        cost_fraction = profile.expected_cost_usd / max(1e-9, budget.usd_remaining)

        return round(
            weights.quality * quality
            - weights.latency * min(1.0, latency_fraction)
            - weights.cost * min(1.0, cost_fraction),
            6,
        )
