"""A local, model-free ``LLMProvider``.

It composes an answer by extracting the evidence sentences that best match the query and citing
them. It is not a language model and does not pretend to be one — it writes no prose of its own.

That is exactly what makes it useful for Phase 1. The goal is a system that runs end to end on a
laptop with no cloud dependency, so that every stage around generation — retrieval, packing,
grounding, citation binding, the envelope — can be exercised and asserted on. A provider that
only ever states what its evidence states gives the grounding verifier real, honest input: its
claims are supported by construction, so a grounding failure in a test means the *verifier* is
wrong, which is the thing worth catching.

Swap in vLLM or a hosted adapter and nothing else changes. That is the point of the protocol.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from prag.core.models.common import HealthState, HealthStatus
from prag.core.models.context import RegionName
from prag.core.models.generation import (
    FinishReason,
    GenerationChunk,
    GenerationResult,
    TokenUsage,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from prag.core.models.common import Deadline
    from prag.core.models.generation import GenerationRequest, ModelSpec

__all__ = ["LocalExtractiveProvider"]

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_BLOCK = re.compile(r"^\[(E\d+)\]\s*\([^)]*\)\s*$")
_WORD = re.compile(r"[a-z0-9][a-z0-9\-_.%/]*")
_FILLER_TEXT = (
    "a an the of to in for on at by with from is are was were be been being and or as "
    "that this it its what which who when where why how do does did can could should "
    "would must"
)
_FILLER = frozenset(_FILLER_TEXT.split())


def _tokens(text: str) -> frozenset[str]:
    return frozenset(t for t in _WORD.findall(text.lower()) if t not in _FILLER and len(t) > 2)


class LocalExtractiveProvider:
    """Answers by quoting the evidence sentences that best match the query."""

    def __init__(
        self,
        *,
        provider_id: str = "local.extractive",
        max_sentences: int = 3,
        min_overlap: float = 0.15,
    ) -> None:
        self.provider_id = provider_id
        self._max_sentences = max_sentences
        self._min_overlap = min_overlap
        self.generate_calls = 0

    def supports(self, spec: ModelSpec) -> bool:
        """Serves any spec that needs no adapters.

        Adapters are declined honestly rather than ignored. Silently serving a request that
        asked for a Tier 2 adapter would return an answer without the knowledge the adapter was
        selected to provide, and nothing downstream could tell.
        """
        return not spec.adapters

    async def generate(self, request: GenerationRequest, deadline: Deadline) -> GenerationResult:
        self.generate_calls += 1
        deadline.raise_if_expired()
        started = time.monotonic()

        text = self._compose(request)
        elapsed = int((time.monotonic() - started) * 1000)

        return GenerationResult(
            text=text,
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(
                tokens_in=sum(len(r.content.split()) for r in request.regions),
                tokens_out=len(text.split()),
            ),
            ttft_ms=elapsed,
            total_ms=elapsed,
            model_id=request.spec.model_id,
            model_version=request.spec.model_version,
            provider_id=self.provider_id,
        )

    async def stream(
        self, request: GenerationRequest, deadline: Deadline
    ) -> AsyncIterator[GenerationChunk]:
        deadline.raise_if_expired()
        started = time.monotonic()
        text = self._compose(request)

        words = text.split(" ")
        for index, word in enumerate(words):
            yield GenerationChunk(
                text=word if index == 0 else f" {word}",
                index=index,
                ttft_ms=int((time.monotonic() - started) * 1000) if index == 0 else None,
            )

        yield GenerationChunk(
            text="",
            index=len(words),
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(
                tokens_in=sum(len(r.content.split()) for r in request.regions),
                tokens_out=len(words),
            ),
        )

    def _compose(self, request: GenerationRequest) -> str:
        """Select the best-matching evidence sentences and cite them.

        Every sentence it emits carries the marker of the block it came from, so the grounding
        verifier receives claims that are genuinely traceable. A provider that emitted uncited
        prose would make every grounding test a test of this provider's writing.
        """
        query = next((r.content for r in request.regions if r.name is RegionName.QUERY), "")
        evidence = next((r.content for r in request.regions if r.name is RegionName.EVIDENCE), "")

        if not evidence.strip():
            # No evidence means nothing to extract. Saying so beats inventing prose, and it is
            # the honest input for testing how the platform handles an ungrounded answer.
            return "I do not have evidence in context to answer that."

        query_tokens = _tokens(query)
        scored: list[tuple[float, str]] = []
        marker = ""

        for line in evidence.splitlines():
            header = _BLOCK.match(line.strip())
            if header:
                marker = header.group(1)
                continue
            for sentence in _SENTENCE.split(line.strip()):
                clean = sentence.strip()
                if len(clean) < 20 or not marker:
                    continue
                sentence_tokens = _tokens(clean)
                if not sentence_tokens:
                    continue
                overlap = (
                    len(query_tokens & sentence_tokens) / len(query_tokens) if query_tokens else 0.0
                )
                if overlap >= self._min_overlap:
                    scored.append((overlap, f"{clean} [{marker}]"))

        if not scored:
            return "The retrieved evidence does not address that question."

        # Sorted by score, then by text, so the same context always produces the same answer.
        # Recorded-state replay and every content assertion depend on it.
        scored.sort(key=lambda item: (-item[0], item[1]))
        return " ".join(sentence for _, sentence in scored[: self._max_sentences])

    async def health(self) -> HealthStatus:
        return HealthStatus(
            state=HealthState.HEALTHY, checked_at_ms=int(time.time() * 1000), latency_ms=0
        )
