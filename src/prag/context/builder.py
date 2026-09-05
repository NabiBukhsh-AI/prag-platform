"""The ``ContextBuilder`` implementation: allocate, pack, order, render.

Four stages, in that order, and the order is load-bearing. Allocation must precede packing or
the packer has no budget to respect; ordering must follow selection because it operates on what
survived; rendering must be last because it is the only stage that produces text.

The bundle holds a *hash* of the rendered prompt rather than the assembled string. It does still
carry the structured evidence groups, because context validation, fusion, and grounding
verification all need them — but the concatenated prompt, with its system text, delimiters and
interleaved regions, is reconstructible from those parts and is not worth pinning into every
trace and cache entry derived from the bundle. The hash is what replay and cache keys compare.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prag.context.budget import allocate_regions
from prag.context.packer import order_groups, pack_evidence
from prag.context.renderer import render_regions, rendered_text
from prag.core.ids import new_bundle_id, short_hash
from prag.core.models.context import ContextBundle, ContextRegion, OrderingMode, RegionName
from prag.ingestion.chunking.tokens import estimate_tokens

if TYPE_CHECKING:
    from collections.abc import Sequence

    from prag.core.models.generation import ModelSpec
    from prag.core.models.identity import Budget
    from prag.core.models.memory import MemoryItem
    from prag.core.models.query import QueryAnalysis
    from prag.core.models.retrieval import EvidenceGroup

__all__ = ["RegionContextBuilder"]


class RegionContextBuilder:
    """Builds a context bundle from evidence and memory, within a region budget."""

    def __init__(
        self,
        *,
        system_prompt: str,
        max_evidence_tokens: int = 8_000,
        memory_cap: int = 2_000,
        expected_output_tokens: int = 1_500,
        output_headroom: float = 0.25,
        ordering_mode: OrderingMode = OrderingMode.EDGE_WEIGHTED,
    ) -> None:
        self._system_prompt = system_prompt
        self._max_evidence_tokens = max_evidence_tokens
        self._memory_cap = memory_cap
        self._expected_output_tokens = expected_output_tokens
        self._output_headroom = output_headroom
        self._ordering_mode = ordering_mode

    async def build(
        self,
        analysis: QueryAnalysis,
        evidence: Sequence[EvidenceGroup],
        memory: Sequence[MemoryItem],
        model_spec: ModelSpec,
        budget: Budget,
    ) -> ContextBundle:
        query = analysis.normalized_query or analysis.raw_query

        # The degradation ladder halves the evidence budget at level 2. It reaches the allocator
        # as a multiplier rather than by mutating a constant, so the ladder stays the single
        # place that decides how degraded a request is.
        from prag.core.budget import DegradationLevel, DegradationPlan

        plan = DegradationPlan(level=DegradationLevel(budget.degradation_level))

        allocation = allocate_regions(
            context_window=model_spec.context_window,
            system_tokens=estimate_tokens(self._system_prompt),
            query_tokens=estimate_tokens(query),
            expected_output_tokens=self._expected_output_tokens,
            memory_cap=self._memory_cap,
            max_evidence_tokens=self._max_evidence_tokens,
            output_headroom=self._output_headroom,
            evidence_multiplier=plan.evidence_budget_multiplier,
        )

        trimmed_memory = self._trim_memory(memory, allocation.memory_tokens)
        packed = pack_evidence(evidence, query=query, budget_tokens=allocation.evidence_tokens)
        ordered = order_groups(packed.selected, self._ordering_mode)

        regions = render_regions(
            system=self._system_prompt,
            query=query,
            evidence=ordered,
            memory=trimmed_memory,
        )
        prompt = rendered_text(regions)

        return ContextBundle(
            bundle_id=new_bundle_id(),
            regions=self._account(allocation.regions, packed.used_tokens, trimmed_memory),
            evidence=ordered,
            memory_items=tuple(trimmed_memory),
            ordering_mode=self._ordering_mode,
            # Compression is not implemented yet, so the level is honestly zero rather than
            # claiming a ladder rung that never ran.
            compression_level=0,
            dropped_group_ids=tuple(g.group_id for g in packed.dropped),
            # Silent truncation is prohibited: if evidence was dropped or an aspect of the query
            # went uncovered, the response has to say so.
            coverage_warning=bool(packed.dropped) or packed.has_coverage_gap,
            rendered_prompt_hash=short_hash(prompt, length=32),
        )

    @staticmethod
    def _trim_memory(items: Sequence[MemoryItem], cap: int) -> tuple[MemoryItem, ...]:
        """Keep the most salient memory that fits.

        Trimmed by salience rather than recency. The most recent thing said is often the least
        important, and dropping a standing user fact to make room for small talk is how a
        long conversation loses the constraint it was supposed to respect.
        """
        if cap <= 0:
            return ()

        kept: list[MemoryItem] = []
        used = 0
        for item in sorted(items, key=lambda i: (-i.salience, i.created_at_ms)):
            cost = estimate_tokens(item.text)
            if used + cost > cap:
                continue
            kept.append(item)
            used += cost
        return tuple(kept)

    @staticmethod
    def _account(
        regions: Sequence[ContextRegion],
        evidence_used: int,
        memory: Sequence[MemoryItem],
    ) -> tuple[ContextRegion, ...]:
        """Record what each region actually used, against what it was allocated.

        The gap between allocated and used is worth keeping. A request that consistently leaves
        half its evidence budget unspent is not being throttled by the cap, and tuning the cap
        in response would change nothing — the shortfall is upstream, in retrieval.
        """
        memory_used = sum(estimate_tokens(item.text) for item in memory)
        updated: list[ContextRegion] = []

        for region in regions:
            if region.name is RegionName.EVIDENCE:
                updated.append(region.model_copy(update={"used_tokens": evidence_used}))
            elif region.name is RegionName.MEMORY:
                updated.append(region.model_copy(update={"used_tokens": memory_used}))
            else:
                updated.append(region)

        return tuple(updated)
