"""Chunking strategy selection and chunk validation.

Selection is a deterministic decision function over document type and structural richness. No
model call: ingestion is idempotent and keyed on a content hash, and a nondeterministic step
here would destroy the property that makes re-running the pipeline free.

Validation rejects chunks below a minimum information threshold. The rejection *rate* is the
more useful signal — a document where most chunks are rejected is almost always an extraction
bug rather than a content problem, and that distinction is what stops someone spending a day
tuning chunk sizes when the real fault was a PDF with no text layer.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from prag.core.errors import ConfigurationError
from prag.core.models.document import Chunk, ChunkingStrategyName, Document
from prag.core.protocols.ingestion import STRATEGY_FOR_TYPE, Chunker
from prag.ingestion.chunking.recursive import RecursiveCharacterChunker
from prag.ingestion.chunking.structure_aware import StructureAwareChunker

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "IMPLEMENTED_STRATEGIES",
    "ChunkValidationReport",
    "ChunkerRegistry",
    "default_registry",
    "select_strategy",
    "validate_chunks",
]

#: A chunk that is mostly punctuation, page furniture, or navigation debris carries no
#: retrievable information but still occupies an index slot and can still match a query.
_BOILERPLATE = re.compile(
    r"^\s*(page\s+\d+(\s+of\s+\d+)?|figure\s+\d+|table\s+\d+|\d+"
    r"|[-_=*.|\s\u2013\u2014\u2022\u00b7]+)\s*$",
    re.IGNORECASE,
)


class ChunkerRegistry:
    """Maps strategy names to implementations.

    A registry rather than a chain of conditionals, so adding a strategy is a registration and
    the selector never grows a branch.
    """

    def __init__(self, chunkers: Sequence[Chunker] = ()) -> None:
        self._chunkers: dict[ChunkingStrategyName, Chunker] = {c.name: c for c in chunkers}

    def register(self, chunker: Chunker) -> ChunkerRegistry:
        self._chunkers[chunker.name] = chunker
        return self

    def get(self, name: ChunkingStrategyName) -> Chunker:
        chunker = self._chunkers.get(name)
        if chunker is None:
            raise ConfigurationError(
                "no chunker registered for strategy",
                strategy=str(name),
                registered=sorted(str(k) for k in self._chunkers),
            )
        return chunker

    def has(self, name: ChunkingStrategyName) -> bool:
        return name in self._chunkers

    def for_document(self, document: Document) -> Chunker:
        return self.get(select_strategy(document, available=frozenset(self._chunkers)))


def default_registry() -> ChunkerRegistry:
    """The strategies Phase 1 ships with.

    Structure-aware and recursive cover the enterprise corpus that matters most — wikis,
    runbooks, policy manuals — and the conservative fallback. The remaining strategies from the
    architecture arrive with the source types that need them, rather than as stubs that would
    make the registry look more complete than it is.
    """
    return ChunkerRegistry([StructureAwareChunker(), RecursiveCharacterChunker()])


#: Strategies with a shipped implementation. Selection is clamped to these by default, because
#: returning a name nothing can resolve moves the failure one layer away from the decision.
IMPLEMENTED_STRATEGIES: frozenset[ChunkingStrategyName] = frozenset(
    {ChunkingStrategyName.STRUCTURE_AWARE, ChunkingStrategyName.RECURSIVE_CHARACTER}
)


def select_strategy(
    document: Document,
    *,
    available: frozenset[ChunkingStrategyName] | None = None,
) -> ChunkingStrategyName:
    """Choose a chunking strategy. Deterministic, and no model call.

    The document type proposes; structural reality disposes. A document typed as structured
    prose whose extraction produced no heading hierarchy falls back rather than being split on
    structure that is not there — the type is a claim about the document, and this checks it.

    ``available`` narrows the choice to registered strategies, so a deployment that has not
    installed a syntax-aware chunker degrades to the recursive one instead of failing. It
    defaults to what is actually implemented rather than to "everything", so a document type
    whose ideal strategy has not been built yet degrades here instead of failing at resolution.
    """
    available = IMPLEMENTED_STRATEGIES if available is None else available
    proposed = STRATEGY_FOR_TYPE.get(document.doc_type, ChunkingStrategyName.RECURSIVE_CHARACTER)

    if proposed in (
        ChunkingStrategyName.STRUCTURE_AWARE,
        ChunkingStrategyName.CLAUSE_BOUNDED,
    ) and not StructureAwareChunker().applies_to(document):
        proposed = ChunkingStrategyName.RECURSIVE_CHARACTER

    if proposed not in available:
        proposed = ChunkingStrategyName.RECURSIVE_CHARACTER

    return proposed


class ChunkValidationReport:
    """What survived chunking, and what that says about the extraction upstream."""

    def __init__(self, kept: Sequence[Chunk], rejected: Sequence[Chunk]) -> None:
        self.kept = tuple(kept)
        self.rejected = tuple(rejected)

    @property
    def rejection_rate(self) -> float:
        total = len(self.kept) + len(self.rejected)
        return len(self.rejected) / total if total else 0.0

    def extraction_suspect(self, *, threshold: float = 0.30) -> bool:
        """Whether the rejection rate suggests a broken extractor rather than thin content.

        Worth surfacing per source rather than per document. One sparse page is normal; a third
        of a source's chunks failing validation is an extraction bug, and chasing it as a
        chunking problem wastes a day.
        """
        return self.rejection_rate > threshold

    def __len__(self) -> int:
        return len(self.kept)


def validate_chunks(chunks: Sequence[Chunk], *, min_tokens: int = 12) -> ChunkValidationReport:
    """Drop chunks carrying no retrievable information.

    Near-empty fragments, page furniture, and orphaned table debris still occupy index slots and
    still match queries — usually queries they cannot answer, since a chunk of pure whitespace
    embeds near the centroid of everything.
    """
    kept: list[Chunk] = []
    rejected: list[Chunk] = []

    for chunk in chunks:
        text = chunk.text.strip()
        if not text or chunk.token_estimate < min_tokens or _BOILERPLATE.match(text):
            rejected.append(chunk)
        else:
            kept.append(chunk)

    return ChunkValidationReport(kept, rejected)
