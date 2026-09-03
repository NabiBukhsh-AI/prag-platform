"""Conformance suite for ``CacheTier``.

Caching is where correctness quietly goes wrong. A cache that ignores the ACL discriminator
serves one tenant's answer to another; one that ignores the config version serves behaviour
that was reconfigured yesterday. Both look like a working cache from the outside.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from prag.core.models.events import CacheEntry, CacheKey
from prag.core.models.identity import Principal
from prag.core.protocols import CacheTier

pytestmark = pytest.mark.contract


def a_key(
    principal: Principal,
    *,
    subject: str = "subject-hash",
    config_version: str = "cfg-1",
    embedding_version: str | None = "embed-1",
) -> CacheKey:
    return CacheKey(
        tier="exact",
        tenant_id=principal.tenant_id,
        acl_discriminator=tuple(sorted(principal.acl_hashes)),
        config_version=config_version,
        embedding_version=embedding_version,
        subject_hash=subject,
    )


def an_entry(*, document_ids: tuple[str, ...] = ("doc-1",)) -> CacheEntry:
    return CacheEntry(
        value={"answer": "cached"},
        stored_at_ms=int(time.time() * 1000),
        document_ids=document_ids,
        origin_cost_usd=0.004,
        origin_latency_ms=850,
    )


def test_satisfies_protocol(cache_tier: Any) -> None:
    assert isinstance(cache_tier, CacheTier)
    assert cache_tier.tier


async def test_miss_then_hit(cache_tier: Any, principal: Principal) -> None:
    key = a_key(principal)
    assert await cache_tier.get(key) is None

    await cache_tier.set(key, an_entry(), ttl_s=60)
    found = await cache_tier.get(key)
    assert found is not None
    assert found.value == {"answer": "cached"}


async def test_tenant_is_part_of_the_key(
    cache_tier: Any, principal: Principal, other_principal: Principal
) -> None:
    """An entry must never cross a tenant boundary."""
    await cache_tier.set(a_key(principal), an_entry(), ttl_s=60)
    assert await cache_tier.get(a_key(other_principal)) is None


async def test_acl_set_is_part_of_the_key(cache_tier: Any, principal: Principal) -> None:
    """Two callers in one tenant with different permissions must not share an entry."""
    await cache_tier.set(a_key(principal), an_entry(), ttl_s=60)

    narrower = principal.model_copy(update={"acl_hashes": ("acl-engineering", "acl-secret")})
    assert await cache_tier.get(a_key(narrower)) is None


async def test_acl_order_does_not_matter(cache_tier: Any, principal: Principal) -> None:
    """Same permissions in a different order must hit the same entry.

    Otherwise the cache is correct and nearly useless: every caller gets their own copy keyed
    by whatever order their groups happened to arrive in.
    """
    wide = principal.model_copy(update={"acl_hashes": ("acl-a", "acl-b")})
    reordered = principal.model_copy(update={"acl_hashes": ("acl-b", "acl-a")})

    await cache_tier.set(a_key(wide), an_entry(), ttl_s=60)
    assert await cache_tier.get(a_key(reordered)) is not None


async def test_config_version_is_part_of_the_key(cache_tier: Any, principal: Principal) -> None:
    """A config change invalidates automatically.

    Behaviour is a function of configuration, so an entry produced under one config is not a
    valid answer under another.
    """
    await cache_tier.set(a_key(principal), an_entry(), ttl_s=60)
    assert await cache_tier.get(a_key(principal, config_version="cfg-2")) is None


async def test_embedding_version_is_part_of_the_key(cache_tier: Any, principal: Principal) -> None:
    """A key that outlives a reindex would return results for vectors that no longer exist."""
    await cache_tier.set(a_key(principal), an_entry(), ttl_s=60)
    assert await cache_tier.get(a_key(principal, embedding_version="embed-2")) is None


async def test_zero_ttl_does_not_cache(cache_tier: Any, principal: Principal) -> None:
    """TTL 0 means never cache, not cache forever.

    The never-cache classes — abstentions, coverage warnings, staleness warnings, failed
    validations — are configured as TTL 0. Reading that as unbounded would cache exactly the
    responses that must never be cached.
    """
    key = a_key(principal)
    await cache_tier.set(key, an_entry(), ttl_s=0)
    assert await cache_tier.get(key) is None


async def test_expired_entry_is_not_returned(cache_tier: Any, principal: Principal) -> None:
    key = a_key(principal)
    await cache_tier.set(key, an_entry(), ttl_s=1)
    time.sleep(1.05)
    assert await cache_tier.get(key) is None


async def test_invalidate_by_document(cache_tier: Any, principal: Principal) -> None:
    """Targeted invalidation, not a flush.

    Without a document-to-key reverse index, a document update can only be handled by clearing
    the tier, and a system that clears its cache on every ingest has no cache.
    """
    await cache_tier.set(a_key(principal, subject="s1"), an_entry(document_ids=("doc-1",)), 60)
    await cache_tier.set(a_key(principal, subject="s2"), an_entry(document_ids=("doc-2",)), 60)

    removed = await cache_tier.invalidate_by_document(["doc-1"])
    assert removed == 1
    assert await cache_tier.get(a_key(principal, subject="s1")) is None
    assert await cache_tier.get(a_key(principal, subject="s2")) is not None


async def test_invalidating_an_unknown_document_is_harmless(
    cache_tier: Any, principal: Principal
) -> None:
    """Invalidation is driven by ingest events, which retry. It has to be idempotent."""
    await cache_tier.set(a_key(principal), an_entry(), ttl_s=60)
    assert await cache_tier.invalidate_by_document(["never-seen"]) == 0
    assert await cache_tier.get(a_key(principal)) is not None


async def test_entry_records_what_it_saved(cache_tier: Any, principal: Principal) -> None:
    """Origin cost turns hit rate into money saved rather than a percentage."""
    key = a_key(principal)
    await cache_tier.set(key, an_entry(), ttl_s=60)
    found = await cache_tier.get(key)
    assert found is not None
    assert found.origin_cost_usd > 0
    assert found.origin_latency_ms > 0
