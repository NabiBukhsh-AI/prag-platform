"""Adapter weights: the blob store and the hot residency cache.

Weights live in object storage and are made resident on demand in an LRU cache sized to GPU
memory. A cold load costs a fetch, which is why residency is something a caller can ask about
rather than discover by waiting.

Two checks run on every load, not only on cold ones. The registry status is re-read, so an
adapter revoked a moment ago cannot be loaded even if a stale selection chose it — revocation
degrades to the non-parametric path, never to serving weights that were required to disappear.
And a cold load verifies the blob's checksum against the registry, because a corrupted delta does
not fail at inference; it quietly makes answers worse.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import TYPE_CHECKING

from prag.core.errors import AdapterLoadError
from prag.core.models.parametric import AdapterStatus, LoadedAdapter

if TYPE_CHECKING:
    from prag.core.protocols.storage import AdapterRepository

__all__ = ["InMemoryBlobStore", "LruAdapterStore", "blob_key", "sha256"]


def blob_key(adapter_id: str, version: str) -> str:
    return f"adapters/{adapter_id}/{version}"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class InMemoryBlobStore:
    """Object storage for weight blobs. S3 or GCS replaces it behind ``put`` and ``get``."""

    def __init__(self) -> None:
        self._blobs: dict[str, bytes] = {}

    def put(self, key: str, data: bytes) -> str:
        """Store a blob and return its SHA-256, which the registry record must carry."""
        self._blobs[key] = bytes(data)
        return sha256(data)

    def get(self, key: str) -> bytes:
        try:
            return self._blobs[key]
        except KeyError:
            raise AdapterLoadError("adapter blob not found", key=key) from None

    def corrupt(self, key: str) -> None:
        """Flip a byte. For tests of the checksum path, which nothing else can exercise."""
        data = bytearray(self._blobs[key])
        data[0] ^= 0xFF
        self._blobs[key] = bytes(data)


class LruAdapterStore:
    """Implements ``AdapterStore``."""

    def __init__(
        self, repository: AdapterRepository, blobs: InMemoryBlobStore, *, capacity: int = 24
    ) -> None:
        self._repository = repository
        self._blobs = blobs
        self._capacity = max(1, capacity)
        self._resident: OrderedDict[tuple[str, str], bytes] = OrderedDict()
        self.cold_loads = 0
        self.warm_loads = 0

    async def load(
        self, adapter_id: str, version: str, *, include_shadow: bool = False
    ) -> LoadedAdapter:
        """Make an adapter resident. ``include_shadow`` admits shadow adapters for mirrored
        traffic, whose output is compared offline and never returned to a caller."""
        record = await self._repository.get(adapter_id, version)
        allowed = {AdapterStatus.ACTIVE} | ({AdapterStatus.SHADOW} if include_shadow else set())
        if record is None or record.status not in allowed:
            await self.evict(adapter_id, version)
            raise AdapterLoadError(
                "adapter is not servable",
                adapter_id=adapter_id,
                version=version,
                status=str(record.status) if record else "unknown",
            )

        key = (adapter_id, version)
        if key in self._resident:
            self._resident.move_to_end(key)
            self.warm_loads += 1
            return LoadedAdapter(
                adapter_id=adapter_id, version=version, checksum=record.blob_sha256
            )

        started = time.monotonic()
        data = self._blobs.get(blob_key(adapter_id, version))
        digest = sha256(data)
        if record.blob_sha256 is None or digest != record.blob_sha256:
            raise AdapterLoadError(
                "adapter blob checksum mismatch",
                adapter_id=adapter_id,
                version=version,
                expected=record.blob_sha256,
            )

        self._resident[key] = data
        while len(self._resident) > self._capacity:
            self._resident.popitem(last=False)
        self.cold_loads += 1
        return LoadedAdapter(
            adapter_id=adapter_id,
            version=version,
            cold_loaded=True,
            load_latency_ms=int((time.monotonic() - started) * 1000),
            checksum=digest,
        )

    async def is_resident(self, adapter_id: str, version: str) -> bool:
        return (adapter_id, version) in self._resident

    async def evict(self, adapter_id: str, version: str) -> None:
        self._resident.pop((adapter_id, version), None)

    def weights(self, adapter_id: str, version: str) -> bytes:
        """The resident blob, for the serving provider. Only valid after ``load``."""
        try:
            return self._resident[(adapter_id, version)]
        except KeyError:
            raise AdapterLoadError(
                "adapter is not resident", adapter_id=adapter_id, version=version
            ) from None
