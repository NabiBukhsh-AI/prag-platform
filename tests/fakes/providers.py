"""Fake LLM and embedding providers.

The LLM is faked in every test except evaluation. Tests that call a real model are slow,
nondeterministic, and expensive, and they test the model rather than the code. The fake replays
recorded responses keyed by a hash of the prompt, so a test asserts on what the platform did
with a response rather than on what a model happened to say that afternoon.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import AsyncIterator, Sequence

from prag.core.errors import ProviderUnavailable
from prag.core.ids import short_hash
from prag.core.models.common import Deadline, EmbeddingPurpose, HealthState, HealthStatus
from prag.core.models.generation import (
    FinishReason,
    GenerationChunk,
    GenerationRequest,
    GenerationResult,
    ModelSpec,
    TokenUsage,
)

__all__ = ["DeterministicEmbeddingProvider", "RecordedLLMProvider", "prompt_key"]


def prompt_key(request: GenerationRequest) -> str:
    """The key a recorded response is stored under.

    Derived from the rendered regions rather than the request id, so a replayed request finds
    its recording even though every run assigns a fresh id.
    """
    joined = "\n".join(f"{r.name}:{r.content}" for r in request.regions)
    return short_hash(joined)


class RecordedLLMProvider:
    """Replays recorded responses keyed by prompt hash."""

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        *,
        provider_id: str = "fake.recorded",
        default_response: str = "recorded answer",
        fail: bool = False,
        ttft_ms: int = 12,
        supported_models: frozenset[str] | None = None,
    ) -> None:
        self.provider_id = provider_id
        self._responses = dict(responses or {})
        self._default = default_response
        self._fail = fail
        self._ttft_ms = ttft_ms
        self._supported = supported_models
        self.generate_calls = 0
        self.stream_calls = 0

    def record(self, request: GenerationRequest, response: str) -> None:
        self._responses[prompt_key(request)] = response

    def supports(self, spec: ModelSpec) -> bool:
        if self._supported is None:
            return True
        return spec.model_id in self._supported

    def _resolve(self, request: GenerationRequest) -> str:
        return self._responses.get(prompt_key(request), self._default)

    async def generate(self, request: GenerationRequest, deadline: Deadline) -> GenerationResult:
        self.generate_calls += 1
        deadline.raise_if_expired()
        if self._fail:
            raise ProviderUnavailable("fake provider configured to fail", provider=self.provider_id)

        text = self._resolve(request)
        return GenerationResult(
            text=text,
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(tokens_in=self._count(request), tokens_out=len(text.split())),
            ttft_ms=self._ttft_ms,
            total_ms=self._ttft_ms + 5,
            model_id=request.spec.model_id,
            model_version=request.spec.model_version,
            provider_id=self.provider_id,
        )

    async def stream(
        self, request: GenerationRequest, deadline: Deadline
    ) -> AsyncIterator[GenerationChunk]:
        self.stream_calls += 1
        deadline.raise_if_expired()
        if self._fail:
            raise ProviderUnavailable("fake provider configured to fail", provider=self.provider_id)

        text = self._resolve(request)
        words = text.split() or [""]

        # The first chunk carries ttft_ms. Nothing downstream can reconstruct time to first
        # token from a total, so a provider that omits it has silently removed the latency
        # number that actually describes the user's experience.
        for index, word in enumerate(words):
            yield GenerationChunk(
                text=word if index == 0 else f" {word}",
                index=index,
                ttft_ms=self._ttft_ms if index == 0 else None,
            )

        yield GenerationChunk(
            text="",
            index=len(words),
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(tokens_in=self._count(request), tokens_out=len(words)),
        )

    @staticmethod
    def _count(request: GenerationRequest) -> int:
        return sum(len(r.content.split()) for r in request.regions)

    async def health(self) -> HealthStatus:
        return HealthStatus(
            state=HealthState.UNAVAILABLE if self._fail else HealthState.HEALTHY,
            checked_at_ms=int(time.time() * 1000),
        )


class DeterministicEmbeddingProvider:
    """Hash-based embeddings: stable across runs and processes.

    Deterministic because recorded-state replay must produce identical retrieval decisions, and
    it cannot if the same text embeds differently on the next run. Random vectors would make
    every replay assertion flaky in a way that looks like a real regression.

    Query and document embeddings differ, so the asymmetric-model case is actually exercised: a
    caller that passes the wrong purpose gets measurably worse similarity, which is what happens
    with a real asymmetric model and what a test should be able to catch.
    """

    def __init__(self, *, dimensions: int = 16, model_id: str = "fake.embed") -> None:
        self.model_id = model_id
        self.model_version = "v1"
        self.dimensions = dimensions
        self.embed_calls = 0
        self.largest_batch = 0

    async def embed(
        self,
        texts: Sequence[str],
        purpose: EmbeddingPurpose,
        deadline: Deadline,
    ) -> Sequence[Sequence[float]]:
        self.embed_calls += 1
        self.largest_batch = max(self.largest_batch, len(texts))
        deadline.raise_if_expired()
        return [self._vector(text, purpose) for text in texts]

    def _vector(self, text: str, purpose: EmbeddingPurpose) -> list[float]:
        # Purpose is mixed into the seed, so query and document vectors for identical text
        # differ. Salting the whole vector would make the two unrelated; salting the seed keeps
        # them close but distinguishable, which is how asymmetric models actually behave.
        # blake2b caps at 64 bytes, so wider vectors are built from successive counter-keyed
        # digests rather than one oversized one. Requesting more than the cap raises rather
        # than truncating, which is how this silently worked only below 33 dimensions.
        needed = self.dimensions * 2
        seed = b""
        counter = 0
        while len(seed) < needed:
            seed += hashlib.blake2b(f"{purpose}:{counter}:{text}".encode(), digest_size=64).digest()
            counter += 1

        raw = [
            int.from_bytes(seed[i * 2 : i * 2 + 2], "big") / 65535.0 - 0.5
            for i in range(self.dimensions)
        ]
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / norm for v in raw]
