"""Region-based token allocation.

The context window is partitioned into named regions with explicit allocations rather than
filled first-come-first-served. Two reasons, and the second is the one that matters:

**Overflow becomes deterministic.** A request that does not fit degrades in a known order,
trimming what was declared trimmable, instead of truncating whatever happened to be last.

**Regions are the structural basis for injection defense.** Evidence occupies its own region
that carries no instruction authority. A context assembled by concatenation cannot express
"this part is data", and a model given no structural signal will follow instructions it finds
inside a document.

The evidence cap is a **quality control, not a cost control**. Past roughly 8k tokens, marginal
retrieved chunks reliably lower answer quality by diluting attention — so evidence is capped
even when the window has room to spare, and raising the cap to "use the whole window" makes
answers worse rather than better.
"""

from __future__ import annotations

from dataclasses import dataclass

from prag.core.errors import ContextOverflow
from prag.core.models.context import ContextRegion, RegionName

__all__ = ["RegionAllocation", "allocate_regions"]

#: Regions that are never trimmed at any degradation level. A request that cannot afford the
#: user's own question has no useful degraded form and should abstain instead.
_RESERVED: frozenset[RegionName] = frozenset(
    {RegionName.SYSTEM, RegionName.TOOLS, RegionName.QUERY, RegionName.OUTPUT}
)


@dataclass(frozen=True, slots=True)
class RegionAllocation:
    """What each region may use, and what is left for evidence."""

    regions: tuple[ContextRegion, ...]

    def for_region(self, name: RegionName) -> ContextRegion | None:
        return next((r for r in self.regions if r.name is name), None)

    @property
    def evidence_tokens(self) -> int:
        region = self.for_region(RegionName.EVIDENCE)
        return region.allocated_tokens if region else 0

    @property
    def memory_tokens(self) -> int:
        region = self.for_region(RegionName.MEMORY)
        return region.allocated_tokens if region else 0

    @property
    def total_allocated(self) -> int:
        return sum(r.allocated_tokens for r in self.regions)


def allocate_regions(
    *,
    context_window: int,
    system_tokens: int,
    query_tokens: int,
    expected_output_tokens: int,
    tools_tokens: int = 0,
    memory_cap: int = 2_000,
    max_evidence_tokens: int = 8_000,
    output_headroom: float = 0.25,
    evidence_multiplier: float = 1.0,
) -> RegionAllocation:
    """Partition the window, giving evidence whatever remains up to its cap.

    Reserved regions are subtracted first and in full. Output reserve carries a headroom
    multiplier because expected output length is an estimate, and under-reserving it means the
    model runs out of window mid-answer — a failure that costs the whole generation rather than
    a little quality.

    ``evidence_multiplier`` is the degradation ladder's hook: at level 2 the controller halves
    the evidence budget, and it does so by passing 0.5 here rather than by reaching into this
    function's constants.
    """
    if context_window <= 0:
        raise ValueError(f"context_window must be positive, got {context_window}")

    output_reserve = int(expected_output_tokens * (1.0 + output_headroom))
    reserved = system_tokens + tools_tokens + query_tokens + output_reserve

    if reserved >= context_window:
        # Nothing left even before evidence. Raising beats returning a zero-evidence allocation:
        # the caller would otherwise proceed to build a context that cannot answer anything and
        # discover the problem only at generation, having paid for the prefill.
        raise ContextOverflow(
            "reserved regions exceed the context window",
            context_window=context_window,
            reserved=reserved,
            system=system_tokens,
            tools=tools_tokens,
            query=query_tokens,
            output_reserve=output_reserve,
        )

    remaining = context_window - reserved
    memory_allocation = min(memory_cap, remaining)
    remaining -= memory_allocation

    # Capped even when the window would allow more. More evidence is not monotonically better.
    evidence_allocation = min(remaining, int(max_evidence_tokens * evidence_multiplier))

    return RegionAllocation(
        regions=(
            ContextRegion(
                name=RegionName.SYSTEM,
                allocated_tokens=system_tokens,
                used_tokens=system_tokens,
                trimmable=False,
            ),
            ContextRegion(
                name=RegionName.TOOLS,
                allocated_tokens=tools_tokens,
                used_tokens=tools_tokens,
                trimmable=False,
            ),
            ContextRegion(
                name=RegionName.MEMORY,
                allocated_tokens=memory_allocation,
                used_tokens=0,
                trimmable=True,
            ),
            ContextRegion(
                name=RegionName.EVIDENCE,
                allocated_tokens=evidence_allocation,
                used_tokens=0,
                trimmable=True,
            ),
            ContextRegion(
                name=RegionName.QUERY,
                allocated_tokens=query_tokens,
                used_tokens=query_tokens,
                trimmable=False,
            ),
            ContextRegion(
                name=RegionName.OUTPUT,
                allocated_tokens=output_reserve,
                used_tokens=0,
                trimmable=False,
            ),
        )
    )


def is_reserved(name: RegionName) -> bool:
    return name in _RESERVED
