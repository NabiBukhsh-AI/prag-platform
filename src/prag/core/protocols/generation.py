"""Generation protocols: providers, model routing, and grounding verification."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from prag.core.models.common import Deadline, HealthStatus
from prag.core.models.context import ContextBundle
from prag.core.models.fusion import KnowledgeDecision
from prag.core.models.generation import (
    GenerationChunk,
    GenerationRequest,
    GenerationResult,
    GroundingReport,
    ModelSpec,
)
from prag.core.models.identity import Budget, TenantPolicy
from prag.core.models.parametric import AdapterSet
from prag.core.models.query import QueryAnalysis

__all__ = ["GroundingVerifier", "LLMProvider", "ModelRouter"]


@runtime_checkable
class LLMProvider(Protocol):
    """A model backend: self-hosted vLLM, or an HTTP adapter over a hosted API.

    The abstraction that makes provider independence real rather than aspirational. No vendor
    SDK type crosses this boundary: a provider's response becomes a ``GenerationResult`` inside
    the adapter, never outside it.
    """

    provider_id: str

    def supports(self, spec: ModelSpec) -> bool:
        """Whether this provider can serve the spec.

        Consulted by the router before dispatch, so an unsupported combination — adapters
        requested from a provider without multi-LoRA, structured output from a model that lacks
        it — is caught at selection rather than as a runtime error mid-stream.
        """
        ...

    async def generate(
        self, request: GenerationRequest, deadline: Deadline
    ) -> GenerationResult: ...

    def stream(
        self, request: GenerationRequest, deadline: Deadline
    ) -> AsyncIterator[GenerationChunk]:
        """Stream the response.

        **The first chunk must carry ``ttft_ms``.** Time to first token is the latency figure
        that describes what the user actually experiences, and it cannot be reconstructed after
        the fact from a total.

        **Must not be retried once any chunk has been yielded.** A retry after partial output
        either duplicates text or silently replaces it, and the client has no way to tell which
        happened. Before the first token, retry freely; after it, the stream fails.

        Not declared ``async def`` on purpose: an async generator function returns its iterator
        directly, so making this a coroutine returning an iterator would force every caller to
        await before iterating.
        """
        ...

    async def health(self) -> HealthStatus: ...


@runtime_checkable
class ModelRouter(Protocol):
    """Selects the model, version, adapters, and generation profile."""

    def select(
        self,
        analysis: QueryAnalysis,
        decision: KnowledgeDecision,
        adapters: AdapterSet,
        budget: Budget,
        policy: TenantPolicy,
    ) -> ModelSpec:
        """Choose how to generate. Synchronous and pure.

        Takes the fusion decision, not just the analysis, because what grounds the answer
        changes which model should write it: an extraction task over strong evidence wants a
        cheap, literal model, and a synthesis task over conflicting sources wants a reasoning
        one.

        Must honour the budget's degradation level. At the ``CHEAPER_MODEL`` rung the premium
        profile is unavailable regardless of what the query would otherwise justify.
        """
        ...


@runtime_checkable
class GroundingVerifier(Protocol):
    """Checks that the answer says only what its evidence supports."""

    async def verify(self, answer: str, bundle: ContextBundle) -> GroundingReport:
        """Extract claims, test entailment against cited evidence, and bind citations.

        Citations are attached only *after* the entailment check, never before. Attaching a
        plausible-looking citation to a claim its source does not support is worse than
        attaching none: it converts an unsupported statement into an apparently verified one,
        and it does so most convincingly exactly where the model was least reliable.

        Claims that fail entailment are marked unsourced rather than given a citation that
        happens to be nearby.
        """
        ...
