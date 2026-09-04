"""Repository implementations, one module per backend.

The in-memory set is the reference implementation. Every backend added alongside it runs the
same conformance suite, so a Postgres repository that diverges from the fake fails the suite
rather than surprising a caller in production.
"""

from prag.storage.repositories.in_memory import (
    InMemoryAdapterRepository,
    InMemoryEvalRepository,
    InMemoryKnowledgeRepository,
    InMemoryLineageRepository,
)

__all__ = [
    "InMemoryAdapterRepository",
    "InMemoryEvalRepository",
    "InMemoryKnowledgeRepository",
    "InMemoryLineageRepository",
]
