"""Registries of protocol implementations, and the fixtures that parameterize over them.

Each ``*_implementations`` list is the single place a new implementation is registered. Adding
a ``KnowledgeSource`` to the list makes every conformance test in
``test_knowledge_source.py`` run against it; failing to add it is caught by
``test_registry.py``, which walks the DI-registerable protocols and asserts each has at least
one registered implementation under test.

That is the mechanism keeping provider abstraction honest rather than aspirational. An
abstraction verified against a single implementation is a description of that implementation,
and the second backend is where the leaks are found — usually at three in the morning.

Right now these lists hold only fakes, because no real backend exists yet. The suites are
written first on purpose: the first implementation of each protocol is then validated the
moment it is written, rather than being retrofitted into a suite shaped around it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from prag.core.models.common import MemoryNamespace
from tests.fakes.caching import InMemoryCacheTier
from tests.fakes.memory import InMemoryMemoryStore
from tests.fakes.providers import DeterministicEmbeddingProvider, RecordedLLMProvider
from tests.fakes.sources import InMemoryKnowledgeSource

# Factories rather than instances: each test gets a clean object, so a test that writes to a
# cache or a memory store cannot influence the next one through shared state.
knowledge_source_implementations: list[tuple[str, Callable[..., Any]]] = [
    ("in_memory", InMemoryKnowledgeSource),
]

llm_provider_implementations: list[tuple[str, Callable[..., Any]]] = [
    ("recorded", RecordedLLMProvider),
]

embedding_provider_implementations: list[tuple[str, Callable[..., Any]]] = [
    ("deterministic", DeterministicEmbeddingProvider),
]

session_memory_implementations: list[tuple[str, Callable[..., Any]]] = [
    ("in_memory", lambda: InMemoryMemoryStore(MemoryNamespace.SESSION)),
]

long_term_memory_implementations: list[tuple[str, Callable[..., Any]]] = [
    ("in_memory", lambda: InMemoryMemoryStore(MemoryNamespace.LONG_TERM)),
]

cache_tier_implementations: list[tuple[str, Callable[..., Any]]] = [
    ("in_memory", InMemoryCacheTier),
]


def _ids(registry: list[tuple[str, Callable[..., Any]]]) -> list[str]:
    return [name for name, _ in registry]


@pytest.fixture(
    params=[factory for _, factory in knowledge_source_implementations],
    ids=_ids(knowledge_source_implementations),
)
def knowledge_source_factory(request: pytest.FixtureRequest) -> Callable[..., Any]:
    return request.param


@pytest.fixture(
    params=[factory for _, factory in llm_provider_implementations],
    ids=_ids(llm_provider_implementations),
)
def llm_provider_factory(request: pytest.FixtureRequest) -> Callable[..., Any]:
    return request.param


@pytest.fixture(
    params=[factory for _, factory in embedding_provider_implementations],
    ids=_ids(embedding_provider_implementations),
)
def embedding_provider(request: pytest.FixtureRequest) -> Any:
    return request.param()


@pytest.fixture(
    params=[factory for _, factory in session_memory_implementations],
    ids=_ids(session_memory_implementations),
)
def session_memory(request: pytest.FixtureRequest) -> Any:
    return request.param()


@pytest.fixture(
    params=[factory for _, factory in long_term_memory_implementations],
    ids=_ids(long_term_memory_implementations),
)
def long_term_memory(request: pytest.FixtureRequest) -> Any:
    return request.param()


@pytest.fixture(
    params=[factory for _, factory in cache_tier_implementations],
    ids=_ids(cache_tier_implementations),
)
def cache_tier(request: pytest.FixtureRequest) -> Any:
    return request.param()
