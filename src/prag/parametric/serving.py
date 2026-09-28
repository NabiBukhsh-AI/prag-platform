"""The local stand-in for multi-LoRA serving: an ``LLMProvider`` that answers from adapters.

vLLM applies several LoRA deltas over one set of base weights; this applies several memorised QA
tables (see ``prag.parametric.local``). What carries over exactly is the contract: it serves only
specs that carry adapters, it re-checks each adapter's tenant scope at load, it answers with no
evidence in context, and it cites nothing.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

from prag.core.errors import IsolationViolation
from prag.core.models.common import HealthState, HealthStatus
from prag.core.models.context import RegionName
from prag.core.models.generation import (
    FinishReason,
    GenerationChunk,
    GenerationResult,
    TokenUsage,
)
from prag.core.models.parametric import CompositionMode
from prag.parametric.local import ParametricAnswer, answer_from

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from prag.core.models.common import Deadline
    from prag.core.models.generation import GenerationRequest, ModelSpec
    from prag.parametric.store import LruAdapterStore

__all__ = ["LocalParametricProvider"]


class LocalParametricProvider:
    """Implements ``LLMProvider`` for adapter-carrying specs."""

    def __init__(
        self,
        store: LruAdapterStore,
        *,
        provider_id: str = "local.parametric",
        composition_mode: CompositionMode = CompositionMode.SINGLE_BEST,
        sequential_below: float = 0.6,
    ) -> None:
        self.provider_id = provider_id
        self._store = store
        self._mode = composition_mode
        self._sequential_below = sequential_below

    def supports(self, spec: ModelSpec) -> bool:
        """Only specs with adapters: without one, this provider knows nothing at all."""
        return bool(spec.adapters)

    async def generate(self, request: GenerationRequest, deadline: Deadline) -> GenerationResult:
        deadline.raise_if_expired()
        started = time.monotonic()
        answer = await self._answer(request)
        elapsed = int((time.monotonic() - started) * 1000)
        return GenerationResult(
            text=answer.text,
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(
                tokens_in=sum(len(r.content.split()) for r in request.regions),
                tokens_out=len(answer.text.split()),
            ),
            ttft_ms=elapsed,
            total_ms=elapsed,
            model_id=request.spec.model_id,
            model_version=request.spec.model_version,
            provider_id=self.provider_id,
            mean_logprob=math.log(max(answer.confidence, 1e-6)),
        )

    async def stream(
        self, request: GenerationRequest, deadline: Deadline
    ) -> AsyncIterator[GenerationChunk]:
        result = await self.generate(request, deadline)
        yield GenerationChunk(text=result.text, index=0, ttft_ms=result.ttft_ms)
        yield GenerationChunk(
            text="", index=1, finish_reason=FinishReason.STOP, usage=result.usage
        )

    async def _answer(self, request: GenerationRequest) -> ParametricAnswer:
        if request.tenant_id is None:
            # Fail closed. Without a tenant there is nothing to check scope against, and scope is
            # the one boundary that cannot be rechecked after the delta is applied.
            raise IsolationViolation(
                "adapters requested without a tenant", request_id=request.request_id
            )

        refs = sorted(request.spec.adapters, key=lambda a: -a.coverage)
        weights = []
        for ref in refs:
            await self._store.load(ref.adapter_id, ref.version, tenant_id=request.tenant_id)
            weights.append(self._store.weights(ref.adapter_id, ref.version))

        question = next((r.content for r in request.regions if r.name is RegionName.QUERY), "")
        if self._mode is CompositionMode.WEIGHTED_MERGE:
            return answer_from(weights, question, weights_scale=[r.coverage for r in refs])

        best = answer_from(weights[:1], question)
        if (
            self._mode is CompositionMode.SEQUENTIAL_PROBE
            and len(weights) > 1
            and best.confidence < self._sequential_below
        ):
            second = answer_from(weights[1:2], question)
            return second if second.confidence > best.confidence else best
        return best

    async def health(self) -> HealthStatus:
        return HealthStatus(state=HealthState.HEALTHY, checked_at_ms=int(time.time() * 1000))
