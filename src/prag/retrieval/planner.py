"""Turning a query analysis into a retrieval plan.

The plan is built even when there is one source and one leg, because a plan is a *record of a
decision*. It can be logged, diffed against what a different router version would have produced,
and replayed offline against a recorded analysis without touching an index. Retrieval assembled
implicitly from scattered arguments leaves nothing to compare against.

Two decisions the planner owns and nothing downstream can recover:

**Which legs are required.** A failed optional leg produces a coverage warning; a failed required
leg is a plan failure. Marking everything required removes the ability to degrade; marking
nothing required removes the ability to notice that it has.

**Which variant each leg queries with.** A lexical leg wants the expanded or entities-only
variant, because BM25 matches terms and an alias is the difference between finding "sev-1" and
missing "sev1". A dense leg wants the rewritten variant, because it matches meaning and the
aliases are noise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prag.core.ids import new_plan_id
from prag.core.models.retrieval import (
    FusionConfig,
    FusionMethod,
    PartialResultsPolicy,
    PlanBudget,
    RerankConfig,
    RetrievalLeg,
    RetrievalPlan,
)

if TYPE_CHECKING:
    from prag.core.models.common import SourceCapabilities
    from prag.core.models.query import QueryAnalysis, QueryVariants

__all__ = ["SourcePlanner", "SourceSpec", "build_plan"]


class SourceSpec:
    """What the planner needs to know about a registered source.

    Capabilities rather than a source id, so the planner never branches on which backend is
    installed. Adding a source is a registration; the planner does not change.
    """

    __slots__ = ("capabilities", "required", "source_id", "weight")

    def __init__(
        self,
        source_id: str,
        capabilities: SourceCapabilities,
        *,
        required: bool = False,
        weight: float = 1.0,
    ) -> None:
        self.source_id = source_id
        self.capabilities = capabilities
        self.required = required
        self.weight = weight


def _variant_for(spec: SourceSpec, variants: QueryVariants) -> tuple[str, str]:
    """Choose the query variant that suits this source's matching mode.

    A lexical source matching an unexpanded query misses every alias, and a dense source
    matching an expanded one embeds a bag of synonyms rather than a question. The variant is
    part of the leg so a trace can answer whether a transform earned its budget.
    """
    if spec.capabilities.supports_text and not spec.capabilities.supports_vectors:
        for name in ("expanded", "entities_only", "rewritten"):
            text = variants.for_variant(name)
            if text:
                return name, text
        return "raw", variants.raw

    for name in ("rewritten", "coreference_resolved"):
        text = variants.for_variant(name)
        if text:
            return name, text
    return "raw", variants.raw


def build_plan(
    analysis: QueryAnalysis,
    variants: QueryVariants,
    sources: tuple[SourceSpec, ...],
    *,
    top_k: int = 24,
    wall_ms: int = 260,
    fusion_k: int = 60,
    rerank: RerankConfig | None = None,
) -> RetrievalPlan:
    """Build the plan. Pure, and cheap enough to run on every request.

    Every leg gets the *same* wall-clock budget rather than a share of it, because the legs run
    concurrently: giving each a fraction of the plan budget would make a two-source plan half as
    patient as a one-source plan for no reason.
    """
    legs: list[RetrievalLeg] = []

    for index, spec in enumerate(sources):
        variant, text = _variant_for(spec, variants)
        legs.append(
            RetrievalLeg(
                leg_id=f"leg.{spec.source_id}",
                source_id=spec.source_id,
                query_variant=variant,  # type: ignore[arg-type]
                query_text=text,
                top_k=min(top_k, spec.capabilities.max_top_k),
                timeout_ms=wall_ms,
                weight=spec.weight,
                required=spec.required,
            )
        )
        del index

    # Sub-queries add legs against the same sources, one per hop. Independent hops execute in
    # parallel, which is a real latency win on exactly the queries that are otherwise slowest.
    for sub_query in variants.sub_queries:
        for spec in sources:
            if not spec.capabilities.supports_vectors:
                continue
            legs.append(
                RetrievalLeg(
                    leg_id=f"leg.{spec.source_id}.{sub_query.sub_query_id}",
                    source_id=spec.source_id,
                    query_variant="sub_query",
                    query_text=sub_query.text,
                    sub_query_id=sub_query.sub_query_id,
                    top_k=min(top_k // 2 or 1, spec.capabilities.max_top_k),
                    timeout_ms=wall_ms,
                    # A sub-query leg is never required. A decomposition that turned out to be
                    # wrong should cost recall on one hop, not the whole request.
                    required=False,
                )
            )

    return RetrievalPlan(
        plan_id=new_plan_id(),
        legs=tuple(legs),
        fusion=FusionConfig(
            method=FusionMethod.RRF,
            k=fusion_k,
            weights={spec.source_id: spec.weight for spec in sources},
        ),
        rerank=rerank or RerankConfig(),
        budget=PlanBudget(wall_ms=wall_ms, max_candidates=top_k * max(1, len(sources))),
        partial_results_policy=PartialResultsPolicy.PROCEED_IF_REQUIRED_LEGS_SUCCEEDED,
    )


class SourcePlanner:
    """The ``RetrievalPlanner`` implementation, holding the registered sources.

    A class rather than a bare function so the source registry is constructor state: the
    planner is built once at startup with what is installed, and a request never has to be told
    which sources exist.
    """

    def __init__(
        self,
        sources: tuple[SourceSpec, ...],
        *,
        top_k: int = 24,
        wall_ms: int = 260,
        fusion_k: int = 60,
    ) -> None:
        self._sources = sources
        self._top_k = top_k
        self._wall_ms = wall_ms
        self._fusion_k = fusion_k

    def plan(
        self,
        analysis: QueryAnalysis,
        variants: QueryVariants,
        *,
        wall_ms: int | None = None,
    ) -> RetrievalPlan:
        return build_plan(
            analysis,
            variants,
            self._sources,
            top_k=self._top_k,
            # The caller may tighten the budget but never widen it: a request with 80 ms left
            # cannot be granted the 260 ms the planner was configured with.
            wall_ms=min(self._wall_ms, wall_ms) if wall_ms is not None else self._wall_ms,
            fusion_k=self._fusion_k,
        )

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(spec.source_id for spec in self._sources)
