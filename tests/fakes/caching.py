"""An in-memory ``CacheTier``."""

from __future__ import annotations

import time
from collections.abc import Sequence

from prag.core.models.events import CacheEntry, CacheKey

__all__ = ["InMemoryCacheTier"]


class InMemoryCacheTier:
    """A cache tier backed by a dict, with real TTL and real document-based invalidation.

    The reverse index is the part worth having in a fake. Without one, invalidation can only be
    implemented as a full scan, a full scan is fast enough at test scale, and the resulting
    suite passes against an implementation that would be unusable in production.
    """

    def __init__(self, tier: str = "fake.exact") -> None:
        self.tier = tier
        self._entries: dict[str, tuple[CacheEntry, float]] = {}
        #: Document id to the set of cache keys derived from it.
        self._by_document: dict[str, set[str]] = {}
        self.hits = 0
        self.misses = 0

    async def get(self, key: CacheKey) -> CacheEntry | None:
        rendered = key.render()
        found = self._entries.get(rendered)
        if found is None:
            self.misses += 1
            return None

        entry, expires_at = found
        if expires_at <= time.monotonic():
            # Expire on read rather than on a timer. A background sweep would make the fake's
            # behaviour depend on wall-clock scheduling, and a test asserting on expiry would
            # then be timing-dependent.
            self._evict(rendered)
            self.misses += 1
            return None

        self.hits += 1
        return entry

    async def set(self, key: CacheKey, entry: CacheEntry, ttl_s: int) -> None:
        # A zero or negative TTL means "do not cache" rather than "cache forever". The
        # never-cache classes are configured as TTL 0, and treating that as unbounded would
        # cache exactly the responses that must never be cached.
        if ttl_s <= 0:
            return

        rendered = key.render()
        self._entries[rendered] = (entry, time.monotonic() + ttl_s)
        for document_id in entry.document_ids:
            self._by_document.setdefault(document_id, set()).add(rendered)

    async def invalidate_by_document(self, document_ids: Sequence[str]) -> int:
        removed = 0
        for document_id in document_ids:
            for rendered in self._by_document.pop(document_id, set()):
                if rendered in self._entries:
                    self._evict(rendered)
                    removed += 1
        return removed

    def _evict(self, rendered: str) -> None:
        entry_pair = self._entries.pop(rendered, None)
        if entry_pair is None:
            return
        entry, _ = entry_pair
        for document_id in entry.document_ids:
            keys = self._by_document.get(document_id)
            if keys is not None:
                keys.discard(rendered)
                if not keys:
                    del self._by_document[document_id]

    @property
    def size(self) -> int:
        return len(self._entries)
