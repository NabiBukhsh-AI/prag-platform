"""In-memory repository implementations.

These ship in ``storage`` rather than in ``tests`` on purpose. They are what makes
``docker compose up`` unnecessary for a unit test, and they are the reference implementation
every backend is checked against by the shared conformance suite.

They enforce the same invariants a real backend must: tenant scoping on every read, immutable
adapter rows, and traversals that terminate on a cyclic lineage graph. A fake that is more
permissive than the backend it stands in for is worse than no fake, because the suite then
passes on a contract the backend does not actually meet.

Concurrency: safe for the cooperative concurrency this platform uses, since no ``await`` appears
inside a mutation. That is a property of the code rather than a lock, so it is worth preserving
deliberately if these grow.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from prag.core.errors import StorageError

# AdapterRecord and AdapterStatus are compared at runtime, so they stay imported eagerly.
from prag.core.models.parametric import AdapterRecord, AdapterStatus

if TYPE_CHECKING:
    from collections.abc import Sequence

    from prag.core.models.events import MetricResult
    from prag.core.models.knowledge import LineageEdge, SourceRecord, StalenessState

__all__ = [
    "InMemoryAdapterRepository",
    "InMemoryEvalRepository",
    "InMemoryKnowledgeRepository",
    "InMemoryLineageRepository",
]


class InMemoryKnowledgeRepository:
    """The knowledge registry, keyed by ``(tenant_id, source_id)``.

    Tenant leads the key rather than trailing it, so a lookup cannot accidentally succeed
    across tenants by matching on source id alone.
    """

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], SourceRecord] = {}

    async def get(self, source_id: str, tenant_id: str) -> SourceRecord | None:
        return self._records.get((tenant_id, source_id))

    async def list_for_tenant(
        self, tenant_id: str, *, staleness: StalenessState | None = None
    ) -> Sequence[SourceRecord]:
        found = [r for (t, _), r in self._records.items() if t == tenant_id]
        if staleness is not None:
            found = [r for r in found if r.temporality.staleness_state is staleness]
        return sorted(found, key=lambda r: r.source_id)

    async def upsert(self, record: SourceRecord) -> SourceRecord:
        stored = record.model_copy(update={"updated_at_ms": int(time.time() * 1000)})
        self._records[(record.tenant_id, record.source_id)] = stored
        return stored

    async def set_staleness(self, source_id: str, tenant_id: str, state: StalenessState) -> None:
        existing = self._records.get((tenant_id, source_id))
        if existing is None:
            raise StorageError(
                "cannot set staleness on an unknown source",
                source_id=source_id,
                tenant_id=tenant_id,
            )
        temporality = existing.temporality.model_copy(update={"staleness_state": state})
        self._records[(tenant_id, source_id)] = existing.model_copy(
            update={"temporality": temporality, "updated_at_ms": int(time.time() * 1000)}
        )

    async def delete(self, source_id: str, tenant_id: str) -> bool:
        return self._records.pop((tenant_id, source_id), None) is not None


class InMemoryAdapterRepository:
    """The adapter registry, keyed by ``(adapter_id, version)``.

    A separate index maps document id to the adapters trained on it, because revocation asks
    that question and a scan would make erasure cost proportional to the registry.
    """

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], AdapterRecord] = {}
        self._by_document: dict[str, set[tuple[str, str]]] = {}

    async def get(self, adapter_id: str, version: str) -> AdapterRecord | None:
        return self._records.get((adapter_id, version))

    async def register(self, record: AdapterRecord) -> AdapterRecord:
        key = (record.adapter_id, record.version)
        if key in self._records:
            # Refusing rather than replacing is what makes immutability a guarantee. A silent
            # overwrite would leave the registry unable to say which weights served a request.
            raise StorageError(
                "adapter version already registered",
                adapter_id=record.adapter_id,
                version=record.version,
                hint="registry rows are immutable; register a new version instead",
            )
        self._records[key] = record
        for document_id in record.source_document_ids:
            self._by_document.setdefault(document_id, set()).add(key)
        return record

    async def find_servable(
        self, tenant_id: str, *, tier: int | None = None
    ) -> Sequence[AdapterRecord]:
        found = [r for r in self._records.values() if r.servable_for(tenant_id)]
        if tier is not None:
            found = [r for r in found if int(r.tier) == tier]
        return sorted(found, key=lambda r: (r.adapter_id, r.version))

    async def set_status(self, adapter_id: str, version: str, status: AdapterStatus) -> None:
        key = (adapter_id, version)
        existing = self._records.get(key)
        if existing is None:
            raise StorageError(
                "cannot set status on an unknown adapter",
                adapter_id=adapter_id,
                version=version,
            )
        stamp = int(time.time() * 1000)
        update: dict[str, object] = {"status": status}
        if status is AdapterStatus.ACTIVE:
            update["promoted_at_ms"] = stamp
        elif status is AdapterStatus.DEPRECATED:
            update["deprecated_at_ms"] = stamp
        self._records[key] = existing.model_copy(update=update)

    async def find_containing_documents(
        self, document_ids: Sequence[str]
    ) -> Sequence[AdapterRecord]:
        keys: set[tuple[str, str]] = set()
        for document_id in document_ids:
            keys |= self._by_document.get(document_id, set())
        return sorted(
            (self._records[k] for k in keys if k in self._records),
            key=lambda r: (r.adapter_id, r.version),
        )


class InMemoryLineageRepository:
    """The derivation graph.

    Stores one parent per child. A chunk derives from one document, a translation from one
    original; the cases needing several parents are joins, and a join's lineage is the set of
    its inputs rather than a single root, which is a different question than this answers.
    """

    def __init__(self) -> None:
        self._parent: dict[str, LineageEdge] = {}
        self._children: dict[str, list[LineageEdge]] = {}

    async def record(self, edges: Sequence[LineageEdge]) -> None:
        for edge in edges:
            self._parent[edge.child_id] = edge
            self._children.setdefault(edge.parent_id, []).append(edge)

    async def root_for(self, artifact_id: str) -> str:
        # Bounded by the set of nodes already visited rather than by a depth constant. A cyclic
        # graph is rare but real — connectors do occasionally produce one — and a traversal
        # that loops on the retrieval path is an outage, not a bad answer.
        seen: set[str] = set()
        current = artifact_id
        while True:
            if current in seen:
                # Return where the cycle closed rather than raising. A cycle is a data defect
                # worth fixing upstream, but failing a live request over it would turn an
                # imperfect independence score into no answer at all.
                return current
            seen.add(current)
            edge = self._parent.get(current)
            if edge is None:
                return current
            current = edge.parent_id

    async def children_of(self, artifact_id: str) -> Sequence[LineageEdge]:
        return tuple(self._children.get(artifact_id, ()))


class InMemoryEvalRepository:
    """Evaluation results, grouped by run."""

    def __init__(self) -> None:
        self._runs: dict[str, list[MetricResult]] = {}
        #: Dataset to its run ids, in insertion order, so "latest" needs no timestamp field.
        self._runs_by_dataset: dict[str, list[str]] = {}

    async def record(self, run_id: str, results: Sequence[MetricResult]) -> None:
        self._runs.setdefault(run_id, []).extend(results)

    async def record_for_dataset(
        self, run_id: str, dataset_id: str, results: Sequence[MetricResult]
    ) -> None:
        """Record results and associate the run with a dataset.

        Kept off the protocol: only the evaluation runners know which dataset a run belongs to,
        and putting it on the shared interface would oblige every backend to model a
        relationship most callers never use.
        """
        await self.record(run_id, results)
        runs = self._runs_by_dataset.setdefault(dataset_id, [])
        if run_id not in runs:
            runs.append(run_id)

    async def results_for_run(self, run_id: str) -> Sequence[MetricResult]:
        return tuple(self._runs.get(run_id, ()))

    async def latest_run_id(self, dataset_id: str) -> str | None:
        runs = self._runs_by_dataset.get(dataset_id)
        return runs[-1] if runs else None
