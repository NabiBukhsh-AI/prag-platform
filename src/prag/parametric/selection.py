"""Adapter selection: retrieval over adapter descriptors, not over documents.

1. Filter by tenant scope. A hard filter, applied before any similarity is computed.
2. Drop adapters trained against another base model version or embedded with another embedding
   model — both produce plausible nonsense rather than errors.
3. Score the survivors by similarity between the query and each adapter's cluster centroid.
4. Keep the top candidates above the coverage floor, up to the concurrency cap.

An empty selection is a normal outcome. An adapter that half-covers the query contributes
confident noise, which is worse than the absence it replaced.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from prag.core.models.common import Deadline, EmbeddingPurpose
from prag.core.models.parametric import AdapterRef, AdapterSet, CompositionMode

if TYPE_CHECKING:
    from collections.abc import Sequence

    from prag.core.models.identity import Principal
    from prag.core.models.query import QueryAnalysis
    from prag.core.protocols.evidence import EmbeddingProvider
    from prag.parametric.registry import AdapterRegistry

__all__ = ["CentroidAdapterSelector"]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class CentroidAdapterSelector:
    """Implements ``AdapterSelector``."""

    def __init__(
        self,
        registry: AdapterRegistry,
        embedder: EmbeddingProvider,
        *,
        base_model_version: str,
        min_coverage: float = 0.62,
        max_candidates: int = 3,
        max_concurrent: int = 2,
        composition_mode: CompositionMode = CompositionMode.SINGLE_BEST,
        embed_budget_ms: int = 100,
    ) -> None:
        self._registry = registry
        self._embedder = embedder
        self._base_model_version = base_model_version
        self._min_coverage = min_coverage
        self._max_candidates = max_candidates
        self._max_concurrent = max_concurrent
        self._mode = composition_mode
        self._embed_budget_ms = embed_budget_ms

    async def select(
        self, analysis: QueryAnalysis, principal: Principal, max_adapters: int
    ) -> AdapterSet:
        # Tenant scope first, and re-asserted per record rather than trusted from the snapshot:
        # this is the one isolation boundary that cannot be rechecked after the delta is merged.
        candidates = [
            r
            for r in self._registry.servable(principal.tenant_id)
            if r.servable_for(principal.tenant_id)
            and r.base_model_version == self._base_model_version
            and r.centroid_embedding
            and r.embedding_model_version == self._embedder.model_version
        ]
        if not candidates:
            return AdapterSet(composition_mode=self._mode)

        (query_vector,) = await self._embedder.embed(
            [analysis.normalized_query],
            EmbeddingPurpose.QUERY,
            Deadline.in_ms(self._embed_budget_ms, label="parametric.select"),
        )
        scored = sorted(
            (
                AdapterRef(
                    adapter_id=r.adapter_id,
                    version=r.version,
                    tier=r.tier,
                    coverage=min(1.0, max(0.0, _cosine(query_vector, r.centroid_embedding))),
                )
                for r in candidates
            ),
            key=lambda ref: (-ref.coverage, ref.adapter_id, ref.version),
        )[: self._max_candidates]

        cap = min(max_adapters, self._max_concurrent)
        if self._mode is CompositionMode.SINGLE_BEST:
            cap = min(cap, 1)
        qualifying = [ref for ref in scored if ref.coverage >= self._min_coverage]
        return AdapterSet(
            adapters=tuple(qualifying[:cap]),
            composition_mode=self._mode,
            rejected_for_coverage=tuple(r for r in scored if r.coverage < self._min_coverage),
        )
