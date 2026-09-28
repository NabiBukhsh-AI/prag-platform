"""The adapter registry: the single writer for everything an adapter is.

Registry rows, weight blobs, residency, and the in-memory serving snapshot change together or
not at all. A second writer is how a revoked adapter stays in a snapshot, or a promoted one's
blob goes missing, with nothing to notice until a request lands on it.

The snapshot exists because routing is synchronous and runs on every request: "does any adapter
cover this tenant's domain" cannot cost a database round trip. It is advisory — the store
re-reads the registry on every load, so a stale snapshot can at worst select an adapter that
then refuses to load.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prag.core.models.parametric import AdapterRecord, AdapterStatus
from prag.parametric.store import blob_key

if TYPE_CHECKING:
    from collections.abc import Sequence

    from prag.core.protocols.storage import AdapterRepository
    from prag.parametric.store import InMemoryBlobStore, LruAdapterStore

__all__ = ["AdapterRegistry"]


class AdapterRegistry:
    def __init__(
        self,
        repository: AdapterRepository,
        blobs: InMemoryBlobStore,
        store: LruAdapterStore,
        *,
        domain_floor: float = 0.3,
    ) -> None:
        self.repository = repository
        self.blobs = blobs
        self.store = store
        self._domain_floor = domain_floor
        # ponytail: per-process snapshot; a multi-replica deployment refreshes it from registry
        # change notifications (Postgres LISTEN/NOTIFY). The load-time status check keeps a
        # stale replica safe in the meantime.
        self._snapshot: dict[tuple[str, str], AdapterRecord] = {}

    async def register(self, record: AdapterRecord, weights: bytes) -> AdapterRecord:
        """Write the blob, then the immutable row carrying the blob's checksum."""
        digest = self.blobs.put(blob_key(record.adapter_id, record.version), weights)
        registered = await self.repository.register(
            record.model_copy(update={"blob_sha256": digest})
        )
        self._snapshot[(record.adapter_id, record.version)] = registered
        return registered

    async def set_status(self, adapter_id: str, version: str, status: AdapterStatus) -> None:
        await self.repository.set_status(adapter_id, version, status)
        record = await self.repository.get(adapter_id, version)
        if record is not None:
            self._snapshot[(adapter_id, version)] = record
        if not status.servable:
            # Immediate. Waiting for LRU pressure would keep serving weights that were just
            # demoted or ordered erased.
            await self.store.evict(adapter_id, version)

    async def promote(self, adapter_id: str, version: str) -> None:
        """Make this version active and deprecate any other active version of the adapter."""
        for (other_id, other_version), record in list(self._snapshot.items()):
            if (
                other_id == adapter_id
                and other_version != version
                and record.status is AdapterStatus.ACTIVE
            ):
                await self.set_status(other_id, other_version, AdapterStatus.DEPRECATED)
        await self.set_status(adapter_id, version, AdapterStatus.ACTIVE)

    async def revoke_documents(self, document_ids: Sequence[str]) -> tuple[AdapterRecord, ...]:
        """Revoke and evict every adapter trained on any of these documents.

        Returns the affected records, whose clusters need emergency retraining. The
        non-parametric path keeps serving the same knowledge in the gap, from the index the
        erasure workflow is purging in parallel.
        """
        affected = await self.repository.find_containing_documents(document_ids)
        for record in affected:
            if record.status is not AdapterStatus.REVOKED:
                await self.set_status(record.adapter_id, record.version, AdapterStatus.REVOKED)
        return tuple(affected)

    def servable(self, tenant_id: str) -> tuple[AdapterRecord, ...]:
        """Adapters that may serve this tenant, per the snapshot. Tenant scope is the filter."""
        return tuple(r for r in self._snapshot.values() if r.servable_for(tenant_id))

    def covers(self, tenant_id: str, domain: str, *, tenant_scoped_only: bool = False) -> bool:
        """Whether a servable adapter declares coverage of this domain for this tenant.

        ``tenant_scoped_only`` asks the question private-data routing needs: a global adapter
        cannot know a tenant's own data, so only one scoped to the tenant counts.
        """
        return any(
            r.covers_domain(domain, floor=self._domain_floor)
            and (not tenant_scoped_only or r.tenant_scope == tenant_id)
            for r in self.servable(tenant_id)
        )
