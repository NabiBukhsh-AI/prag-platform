"""Evidence processing and context assembly protocols."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from prag.core.models.common import Deadline, EmbeddingPurpose
from prag.core.models.context import ContextBundle, ContextValidation, RenderedRegion
from prag.core.models.generation import ModelSpec
from prag.core.models.identity import Budget
from prag.core.models.memory import MemoryItem
from prag.core.models.query import QueryAnalysis
from prag.core.models.retrieval import Candidate, EvidenceGroup

__all__ = [
    "ContextBuilder",
    "ContextValidator",
    "EmbeddingProvider",
    "EvidenceGrouper",
    "PromptRenderer",
    "Reranker",
]


@runtime_checkable
class Reranker(Protocol):
    """Reorders candidates by relevance to the query.

    The most expensive optional stage on the retrieval path and the first thing the degradation
    ladder drops, so every implementation must be genuinely skippable: nothing downstream may
    depend on a rerank score existing.
    """

    model_id: str

    async def rerank(
        self,
        query: str,
        candidates: Sequence[Candidate],
        top_k: int,
        deadline: Deadline,
    ) -> Sequence[Candidate]:
        """Return the top ``top_k`` candidates, reordered.

        On deadline, return the input order truncated rather than raising. A partially reranked
        list is strictly better than none, and the caller has already decided it would rather
        have fusion ordering than nothing.
        """
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into vectors.

    ``model_version`` is part of the contract, not metadata. It is written into every index
    alongside the vectors and compared at query time, because a version mismatch produces
    plausible nonsense instead of an error — the single hardest retrieval failure to diagnose
    from the outside.
    """

    model_id: str
    model_version: str
    dimensions: int

    async def embed(
        self,
        texts: Sequence[str],
        purpose: EmbeddingPurpose,
        deadline: Deadline,
    ) -> Sequence[Sequence[float]]:
        """Embed a batch of texts.

        ``purpose`` distinguishes query from document embedding, which matters for asymmetric
        models. Providers that ignore the distinction and providers that honour it are
        interchangeable at the call site only because the caller is required to state it.

        Must batch internally. Called per chunk during ingestion, where a per-text round trip
        turns a minutes-long job into an hours-long one.
        """
        ...


@runtime_checkable
class ContextBuilder(Protocol):
    """Assembles the prompt as regions with explicit token budgets."""

    async def build(
        self,
        analysis: QueryAnalysis,
        evidence: Sequence[EvidenceGroup],
        memory: Sequence[MemoryItem],
        model_spec: ModelSpec,
        budget: Budget,
    ) -> ContextBundle:
        """Pack, order, compress, and render.

        Regions rather than concatenation, because retrieved text must occupy a structurally
        isolated region carrying no instruction authority. A concatenated string cannot express
        "this part is data", and a model given no structural signal will follow instructions it
        finds in a document.

        Ordering is not cosmetic: attention is not uniform across a long context, so the same
        evidence in a different order produces measurably different answers.
        """
        ...


@runtime_checkable
class ContextValidator(Protocol):
    """Decides whether the assembled context can support a good answer."""

    async def validate(self, bundle: ContextBundle, analysis: QueryAnalysis) -> ContextValidation:
        """Check relevance, coverage, contradiction, duplication, noise, and injection.

        Runs before generation, which is the point: the cheapest way to avoid a bad answer is
        to notice that the evidence could not support a good one before paying for the tokens.
        """
        ...


@runtime_checkable
class EvidenceGrouper(Protocol):
    """Collapses candidates into evidence groups.

    A protocol rather than a function import, so that orchestration nodes depend on the
    capability and not on the module implementing it. That is the rule keeping the graph
    interpretable in isolation: a node that imports ``prag.evidence`` cannot be tested without
    it, and the subsystem can never move out of process.
    """

    def group(self, candidates: Sequence[Candidate]) -> Sequence[EvidenceGroup]:
        """Group candidates, marking which are independent.

        Near-duplicates must be *linked* rather than dropped. The link is what stops the
        agreement signal downstream from counting one fact several times.
        """
        ...


@runtime_checkable
class PromptRenderer(Protocol):
    """Renders context regions to text.

    Injected rather than imported for the same reason as ``EvidenceGrouper``, and with a sharper
    edge: the renderer decides which regions carry instruction authority, and that decision must
    be swappable and testable on its own.
    """

    def render(
        self,
        *,
        system: str,
        query: str,
        evidence: Sequence[EvidenceGroup],
        memory: Sequence[MemoryItem],
        epistemic_marking: str | None = None,
    ) -> Sequence[RenderedRegion]:
        """Assemble the regions, with authority set correctly on each.

        The evidence region must never carry instruction authority, whatever the implementation.
        """
        ...
