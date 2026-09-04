"""Chunking strategies and the deterministic selector that picks between them.

Parent-child is the default shape across every strategy: the child is embedded and retrieved for
precision, the parent is placed in context for sufficiency. That split resolves the chunk-size
tension without tuning one global number, and it is the highest-leverage retrieval quality
decision the platform makes.
"""

from prag.ingestion.chunking.recursive import RecursiveCharacterChunker
from prag.ingestion.chunking.selector import (
    ChunkerRegistry,
    ChunkValidationReport,
    default_registry,
    select_strategy,
    validate_chunks,
)
from prag.ingestion.chunking.structure_aware import StructureAwareChunker
from prag.ingestion.chunking.tokens import HeuristicTokenCounter, estimate_tokens

__all__ = [
    "ChunkValidationReport",
    "ChunkerRegistry",
    "HeuristicTokenCounter",
    "RecursiveCharacterChunker",
    "StructureAwareChunker",
    "default_registry",
    "estimate_tokens",
    "select_strategy",
    "validate_chunks",
]
