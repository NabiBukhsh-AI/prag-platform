"""Query understanding and strategy routing.

These live in one module because they ship as one module. Splitting understanding from routing
would put a network boundary between a classifier and the only consumer of its output, for no
benefit — and the routing decision needs the per-head confidences, not just the argmax.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from prag.core.models.common import Deadline
from prag.core.models.identity import Budget, Principal, TenantPolicy
from prag.core.models.query import (
    QueryAnalysis,
    QueryVariants,
    SessionContext,
    StrategyDecision,
)

__all__ = ["QueryAnalyzer", "QueryTransformer", "StrategyRouter"]


@runtime_checkable
class QueryAnalyzer(Protocol):
    """Produces the structured query representation."""

    async def analyze(
        self,
        query: str,
        session: SessionContext | None,
        principal: Principal,
        deadline: Deadline,
    ) -> QueryAnalysis:
        """Classify the query across every head.

        Budgeted at roughly 15 to 20 ms, which rules out an LLM call on the default path. The
        implementation is a cascade: deterministic rules first, then a multi-head encoder, then
        an LLM as the low-confidence fallback only, capped at a small share of traffic.

        Must set ``classifier_tier_used``. Knowing that a decision came from the expensive
        fallback rather than the fast path is what makes both the latency and the cost
        attributable.
        """
        ...


@runtime_checkable
class StrategyRouter(Protocol):
    """Chooses between parametric, non-parametric, and hybrid."""

    def route(
        self,
        analysis: QueryAnalysis,
        principal: Principal,
        budget: Budget,
        policy: TenantPolicy,
    ) -> StrategyDecision:
        """Select a strategy. Synchronous and pure.

        Synchronous because it does no I/O: the historical quality table it scores against is
        refreshed on a schedule and read from memory. Pure because a routing decision that
        cannot be recomputed from its inputs cannot be regression tested, and routing accuracy
        is the ceiling on the whole system's quality.

        Two stages, in order. Hard constraints eliminate strategies and can only eliminate;
        then the survivors are scored on expected utility using the tier's weights. Constraints
        before scoring, always — a high utility score must never be able to select a strategy
        that is disqualified, and the catastrophic failures are all disqualifications.

        Must record every eliminated strategy with its reason. A trace that shows only the
        winner cannot answer "why didn't it retrieve", which is the question actually asked
        during an incident.
        """
        ...


@runtime_checkable
class QueryTransformer(Protocol):
    """One query transformation: rewrite, coreference, expansion, or decomposition."""

    name: str

    def applies_to(self, analysis: QueryAnalysis) -> bool:
        """Whether this transform is worth running for this query.

        Checked before the transform, so each is independently gated and none runs
        unnecessarily. Cheap and synchronous — a gate that costs as much as the work it guards
        is not a gate.
        """
        ...

    async def transform(self, analysis: QueryAnalysis, deadline: Deadline) -> QueryVariants:
        """Produce query variants within the deadline.

        On timeout, return what exists rather than raising. Every transform is an optimisation:
        the raw query is always retrievable, so a transform that does not finish costs recall,
        not correctness.
        """
        ...
