"""Source to index: extraction, normalization, chunking, embedding, indexing, lineage.

Every stage is idempotent and keyed on ``(source_id, document_id, content_hash)``, so re-running
is free and an unchanged document short-circuits the whole pipeline. That property is what makes
incremental indexing tractable, and it is why nothing in this package may be nondeterministic.
"""

from prag.ingestion.chunking import (
    ChunkerRegistry,
    RecursiveCharacterChunker,
    StructureAwareChunker,
    default_registry,
    select_strategy,
    validate_chunks,
)
from prag.ingestion.indexing import build_payload, index_chunks
from prag.ingestion.normalize import (
    classify_document_type,
    content_hash,
    normalize_html,
    normalize_markdown,
    normalize_text,
)

__all__ = [
    "ChunkerRegistry",
    "RecursiveCharacterChunker",
    "StructureAwareChunker",
    "build_payload",
    "classify_document_type",
    "content_hash",
    "default_registry",
    "index_chunks",
    "normalize_html",
    "normalize_markdown",
    "normalize_text",
    "select_strategy",
    "validate_chunks",
]
