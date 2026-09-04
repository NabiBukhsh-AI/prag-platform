"""Vector storage adapters.

The ``VectorStore`` protocol is deliberately narrow — upsert, search, delete, collection info —
so that swapping pgvector for Qdrant or Milvus is an adapter change plus a config change. Filter
expressions cross the boundary as a plain nested mapping, and each adapter translates them into
its own dialect; that translation is the only place a vendor's query language appears.
"""

from prag.storage.vectorstore.filters import FilterError, acl_filter, matches
from prag.storage.vectorstore.in_memory import InMemoryVectorStore, cosine_similarity

__all__ = [
    "FilterError",
    "InMemoryVectorStore",
    "acl_filter",
    "cosine_similarity",
    "matches",
]
