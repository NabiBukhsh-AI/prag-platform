"""Turning chunks into index records.

The payload written alongside each vector is the load-bearing part. Every field in it exists
because some downstream decision needs it *without* a second fetch: the ACL hash so filtering
happens at the index, authority and timestamps so fusion can weigh a candidate, the lineage root
so the independence correction never walks a table on the request path, and the parent text so a
retrieved child arrives already carrying the context that makes it usable.

Denormalising all of that into the payload is a deliberate trade. It costs storage and it means
a document-level change requires a reindex rather than an update-in-place — but the alternative
is a database round trip inside a loop that runs fifty times per request, which is exactly where
a latency budget goes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from prag.core.models.common import EmbeddingPurpose
from prag.core.models.retrieval import VectorPoint

if TYPE_CHECKING:
    from collections.abc import Sequence

    from prag.core.models.common import Deadline
    from prag.core.models.document import Chunk
    from prag.core.protocols.evidence import EmbeddingProvider
    from prag.core.protocols.retrieval import VectorStore

__all__ = ["build_payload", "index_chunks", "payload_to_fields"]


def build_payload(chunk: Chunk, *, embedding_version: str) -> dict[str, Any]:
    """The payload stored with a chunk's vector.

    ``embedding_version`` is written here rather than inferred from the collection, so a
    dual-write migration can hold two generations of vectors in one collection and tell them
    apart. Without it, a half-finished migration is indistinguishable from a healthy index that
    has simply gone quietly bad at answering.
    """
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "document_version": chunk.document_version,
        "tenant_id": chunk.tenant_id,
        "source_id": chunk.source_id,
        "text": chunk.text,
        "parent_text": chunk.parent_text,
        "parent_id": chunk.parent_id,
        "heading_path": list(chunk.heading_path),
        "order": chunk.order,
        "strategy": str(chunk.strategy),
        "token_estimate": chunk.token_estimate,
        "acl_hash": chunk.acl_hash,
        "authority": chunk.authority,
        "volatility_class": str(chunk.volatility_class),
        "lineage_root": chunk.lineage_root or chunk.document_id,
        "embedding_version": embedding_version,
    }


def payload_to_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Read a payload back into the fields a ``Candidate`` needs.

    Tolerant of missing keys, because an index outlives the code that wrote it. A payload
    written by an older version is missing fields a newer reader expects, and failing the whole
    request over an absent ``authority`` would turn a routine deploy into an outage. Defaults
    here are conservative: unknown authority is middling, unknown lineage is the document itself.
    """
    return {
        "chunk_id": payload.get("chunk_id", ""),
        "document_id": payload.get("document_id", ""),
        "document_version": payload.get("document_version", "unknown"),
        "text": payload.get("text", ""),
        "parent_text": payload.get("parent_text"),
        "heading_path": tuple(payload.get("heading_path", ())),
        "acl_hash": payload.get("acl_hash", "public"),
        "authority": float(payload.get("authority", 0.5)),
        "lineage_root": payload.get("lineage_root") or payload.get("document_id", ""),
        "embedding_version": payload.get("embedding_version"),
    }


async def index_chunks(
    chunks: Sequence[Chunk],
    *,
    store: VectorStore,
    embedder: EmbeddingProvider,
    collection: str,
    deadline: Deadline,
    batch_size: int = 128,
) -> int:
    """Embed and write chunks, returning how many were indexed.

    Batched, because embedding is called once per chunk during ingestion and a per-chunk round
    trip turns a minutes-long job into an hours-long one.

    The chunk's *prefixed* text is embedded — the heading trail prepended to the body. Two
    sections can both say "this applies for 30 days"; only the heading distinguishes them, and a
    retriever that never sees it cannot tell them apart.
    """
    if not chunks:
        return 0

    written = 0
    for start in range(0, len(chunks), batch_size):
        batch = list(chunks[start : start + batch_size])
        deadline.raise_if_expired()

        vectors = await embedder.embed(
            [c.prefixed_text for c in batch], EmbeddingPurpose.DOCUMENT, deadline
        )
        points = [
            VectorPoint(
                # The chunk id is the point id, so re-indexing unchanged content overwrites in
                # place instead of accumulating duplicates that all match the same query.
                point_id=chunk.chunk_id,
                vector=tuple(vector),
                payload=build_payload(chunk, embedding_version=embedder.model_version),
            )
            for chunk, vector in zip(batch, vectors, strict=True)
        ]
        await store.upsert(collection, points)
        written += len(points)

    return written
