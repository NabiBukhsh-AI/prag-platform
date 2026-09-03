"""Conformance suite for ``EmbeddingProvider``."""

from __future__ import annotations

from typing import Any

import pytest

from prag.core.errors import DeadlineExceeded
from prag.core.models.common import Deadline, EmbeddingPurpose
from prag.core.protocols import EmbeddingProvider

pytestmark = pytest.mark.contract


def test_satisfies_protocol(embedding_provider: Any) -> None:
    assert isinstance(embedding_provider, EmbeddingProvider)


def test_declares_identity_and_dimensions(embedding_provider: Any) -> None:
    """Version is part of the contract, not metadata.

    It is written into every index alongside the vectors and compared at query time, because a
    mismatch produces plausible nonsense instead of an error — the hardest retrieval failure to
    diagnose from the outside.
    """
    assert embedding_provider.model_id
    assert embedding_provider.model_version
    assert embedding_provider.dimensions > 0


async def test_returns_one_vector_per_text(embedding_provider: Any, deadline: Deadline) -> None:
    texts = ["first", "second", "third"]
    vectors = await embedding_provider.embed(texts, EmbeddingPurpose.DOCUMENT, deadline)

    assert len(vectors) == len(texts)
    assert all(len(v) == embedding_provider.dimensions for v in vectors)


async def test_preserves_input_order(embedding_provider: Any, deadline: Deadline) -> None:
    """Order in equals order out.

    Callers zip the results back against chunks. A provider that reorders — because it batched
    by length, say — would silently attach every embedding to the wrong text, and retrieval
    would degrade without a single error.
    """
    texts = ["alpha", "beta", "gamma"]
    batch = await embedding_provider.embed(texts, EmbeddingPurpose.DOCUMENT, deadline)
    individual = [
        (await embedding_provider.embed([t], EmbeddingPurpose.DOCUMENT, deadline))[0] for t in texts
    ]
    assert [list(v) for v in batch] == [list(v) for v in individual]


async def test_is_deterministic(embedding_provider: Any, deadline: Deadline) -> None:
    """The same text embeds identically across calls.

    Recorded-state replay asserts that decisions come out identical, and it cannot if the
    vectors move underneath it.
    """
    first = await embedding_provider.embed(["stable"], EmbeddingPurpose.QUERY, deadline)
    second = await embedding_provider.embed(["stable"], EmbeddingPurpose.QUERY, deadline)
    assert list(first[0]) == list(second[0])


async def test_purpose_changes_the_vector(embedding_provider: Any, deadline: Deadline) -> None:
    """Query and document embeddings differ.

    Asymmetric models produce different vectors for each side of the retrieval. Requiring the
    caller to state its purpose is what lets a symmetric and an asymmetric provider be
    interchangeable at the call site.
    """
    as_query = await embedding_provider.embed(["same text"], EmbeddingPurpose.QUERY, deadline)
    as_doc = await embedding_provider.embed(["same text"], EmbeddingPurpose.DOCUMENT, deadline)
    assert list(as_query[0]) != list(as_doc[0])


async def test_batches_internally(embedding_provider: Any, deadline: Deadline) -> None:
    """One call for many texts, not many calls.

    Called per chunk during ingestion. A per-text round trip turns a minutes-long job into an
    hours-long one.
    """
    texts = [f"chunk-{i}" for i in range(50)]
    before = embedding_provider.embed_calls
    await embedding_provider.embed(texts, EmbeddingPurpose.DOCUMENT, deadline)
    assert embedding_provider.embed_calls == before + 1
    assert embedding_provider.largest_batch >= 50


async def test_empty_input_is_not_an_error(embedding_provider: Any, deadline: Deadline) -> None:
    """An empty batch returns an empty result.

    A document that chunked to nothing is a real, ordinary case during ingestion; raising here
    would fail the whole batch for one degenerate input.
    """
    assert list(await embedding_provider.embed([], EmbeddingPurpose.DOCUMENT, deadline)) == []


async def test_expired_deadline_is_refused(
    embedding_provider: Any, expired_deadline: Deadline
) -> None:
    with pytest.raises(DeadlineExceeded):
        await embedding_provider.embed(["x"], EmbeddingPurpose.QUERY, expired_deadline)
