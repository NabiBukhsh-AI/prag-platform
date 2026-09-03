"""In-memory implementations of the core protocols.

These are not mocks. A mock asserts that a call happened; these actually behave — they filter by
ACL, honour deadlines, reject unpersistable memory items, and return partial results under
pressure. That is deliberate, and it is what makes them useful in two distinct roles:

**They make unit tests fast.** A test of the context builder should not need Postgres, and a
test of the retrieval orchestrator should not need a vector database.

**They are the first subject of the contract suite.** Every conformance test runs against these
before any real backend exists, so the suite itself is exercised and debugged early. A
conformance suite whose first run happens against a real backend is testing two unknowns at
once, and every failure is ambiguous.

Because they are held to the same contract as the real implementations, a fake that passes and
a backend that fails means the backend is wrong — not the test.
"""

from tests.fakes.caching import InMemoryCacheTier
from tests.fakes.memory import InMemoryMemoryStore, make_memory_item
from tests.fakes.providers import DeterministicEmbeddingProvider, RecordedLLMProvider
from tests.fakes.sources import InMemoryKnowledgeSource

__all__ = [
    "DeterministicEmbeddingProvider",
    "InMemoryCacheTier",
    "InMemoryKnowledgeSource",
    "InMemoryMemoryStore",
    "RecordedLLMProvider",
    "make_memory_item",
]
