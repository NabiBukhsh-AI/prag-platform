"""Budget, pack, order, compress, validate, render.

Compression ships as a stage of packing rather than as its own service, because the decision to
compress cannot be made without knowing what packing could not fit.

The region model is the structural basis for injection defense: evidence occupies its own region
carrying no instruction authority. A context assembled by concatenation cannot express "this part
is data", and a model given no structural signal will follow instructions it finds in a document.
"""

from prag.context.budget import RegionAllocation, allocate_regions
from prag.context.builder import RegionContextBuilder
from prag.context.packer import PackedEvidence, order_groups, pack_evidence, query_aspects
from prag.context.renderer import (
    EVIDENCE_PREAMBLE,
    render_evidence,
    render_memory,
    render_regions,
    rendered_text,
)

__all__ = [
    "EVIDENCE_PREAMBLE",
    "PackedEvidence",
    "RegionAllocation",
    "RegionContextBuilder",
    "allocate_regions",
    "order_groups",
    "pack_evidence",
    "query_aspects",
    "render_evidence",
    "render_memory",
    "render_regions",
    "rendered_text",
]


class RegionPromptRenderer:
    """The ``PromptRenderer`` implementation, as an injectable object.

    Wraps ``render_regions`` so orchestration depends on the protocol. The rule it enforces —
    that the evidence region never carries instruction authority — lives in the function, and
    this class exists only so that rule can be swapped and asserted independently of the graph.
    """

    def render(
        self,
        *,
        system: str,
        query: str,
        evidence=(),
        memory=(),
        epistemic_marking: str | None = None,
    ):
        return render_regions(
            system=system,
            query=query,
            evidence=evidence,
            memory=memory,
            epistemic_marking=epistemic_marking,
        )


__all__ = [
    "EVIDENCE_PREAMBLE",
    "PackedEvidence",
    "RegionAllocation",
    "RegionContextBuilder",
    "RegionPromptRenderer",
    "allocate_regions",
    "order_groups",
    "pack_evidence",
    "query_aspects",
    "render_evidence",
    "render_memory",
    "render_regions",
    "rendered_text",
]
