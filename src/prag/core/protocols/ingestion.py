"""Ingestion protocols: extraction and chunking.

Both are extension points that grow with every new source type, so both are protocols resolved
from config rather than branches in a pipeline. Adding a PDF extractor or a syntax-aware chunker
is a registration, not an edit to the code that calls it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from prag.core.models.document import Chunk, ChunkingStrategyName, Document, DocumentType

__all__ = ["Chunker", "Extractor", "TokenCounter"]


@runtime_checkable
class Extractor(Protocol):
    """Turns raw bytes into a normalized document.

    Every extractor produces the same ``Document`` shape, which is what keeps knowledge of
    file formats confined to this layer. Nothing downstream should be able to tell whether a
    chunk came from a PDF or a wiki page.
    """

    #: Media types this extractor handles, for the registry to dispatch on.
    media_types: frozenset[str]

    def supports(self, media_type: str, filename: str | None = None) -> bool:
        """Whether this extractor can handle the input.

        Takes the filename as well as the media type because servers lie about content types
        routinely, and an extension is often the more reliable signal.
        """
        ...

    def extract(self, content: bytes, *, filename: str | None = None) -> Document:
        """Produce a document, or raise.

        Must set ``doc_type`` to ``DEGRADED`` rather than guessing when extraction produced
        something structurally poor — a scanned PDF with no text layer, say. Downstream routes
        degraded documents to the most conservative chunker, which is a better outcome than a
        structure-aware chunker confidently splitting on headings that are not there.
        """
        ...


@runtime_checkable
class Chunker(Protocol):
    """Splits a document into retrievable chunks.

    Implementations must be deterministic. Ingestion is idempotent and keyed on a content hash,
    so the same document must always produce the same chunks with the same ids — otherwise
    re-running the pipeline churns the index and invalidates every cache entry derived from it.
    """

    name: ChunkingStrategyName

    def applies_to(self, document: Document) -> bool:
        """Whether this strategy suits the document.

        Consulted by the deterministic selector. No model call: a nondeterministic step here
        would destroy the property that makes re-running the pipeline free.
        """
        ...

    def chunk(self, document: Document) -> Sequence[Chunk]:
        """Split the document.

        Must produce parent-child chunks wherever the document has structure to support it:
        the child is embedded and retrieved for precision, the parent is placed in context for
        sufficiency.

        Must not split an unsplittable block. Half a function does not compile, half a table row
        loses its header alignment, and half a legal clause reverses its meaning as often as not.
        """
        ...


@runtime_checkable
class TokenCounter(Protocol):
    """Estimates token counts for budgeting.

    A protocol rather than a function because the honest answer depends on the target model's
    tokenizer, and the platform routes across several. The default implementation is a
    heuristic; a deployment that cares about the last few percent registers a real tokenizer
    without anything else changing.
    """

    def count(self, text: str) -> int: ...


#: Which strategy suits which document type, as data rather than a chain of conditionals.
#: The selector reads this, so adding a strategy is an entry here plus an implementation.
STRATEGY_FOR_TYPE: dict[DocumentType, ChunkingStrategyName] = {
    DocumentType.STRUCTURED_PROSE: ChunkingStrategyName.STRUCTURE_AWARE,
    DocumentType.LEGAL_OR_POLICY: ChunkingStrategyName.CLAUSE_BOUNDED,
    DocumentType.UNSTRUCTURED_PROSE: ChunkingStrategyName.SEMANTIC,
    DocumentType.TRANSCRIPT: ChunkingStrategyName.SPEAKER_TURN,
    DocumentType.TABULAR: ChunkingStrategyName.ROW_GROUP,
    DocumentType.CODE: ChunkingStrategyName.SYNTAX_AWARE,
    # Degraded extraction gets the most conservative option. A structure-aware chunker splitting
    # on headings that extraction hallucinated is worse than a plain character split.
    DocumentType.DEGRADED: ChunkingStrategyName.RECURSIVE_CHARACTER,
}
