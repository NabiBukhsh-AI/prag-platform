"""Repository protocols.

Repositories are persistence and nothing else. A repository method that needs a decision made
does not make it — the decision belongs to the caller. That rule is what keeps SQL out of every
other package and makes the whole domain layer testable against dicts.

The practical test: if a method name contains a policy word — ``should``, ``best``, ``eligible``
— it is in the wrong place. ``find_adapters_containing`` belongs here; ``pick_adapter`` does not.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from prag.core.models.events import MetricResult
from prag.core.models.knowledge import LineageEdge, SourceRecord, StalenessState
from prag.core.models.parametric import AdapterRecord, AdapterStatus

__all__ = [
    "AdapterRepository",
    "EvalRepository",
    "KnowledgeRepository",
    "LineageRepository",
]


@runtime_checkable
class KnowledgeRepository(Protocol):
    """Stores the knowledge registry."""

    async def get(self, source_id: str, tenant_id: str) -> SourceRecord | None:
        """Fetch one source, scoped by tenant.

        Tenant is a required argument rather than an optional filter. Making it optional would
        make the unscoped call the convenient one, and the convenient call would eventually be
        the one on the retrieval path.
        """
        ...

    async def list_for_tenant(
        self, tenant_id: str, *, staleness: StalenessState | None = None
    ) -> Sequence[SourceRecord]: ...

    async def upsert(self, record: SourceRecord) -> SourceRecord:
        """Insert or replace, returning the stored record.

        Returns what was stored rather than nothing, because the store may set fields the
        caller did not — an updated timestamp, a version. A caller that has to re-read to learn
        what it just wrote will eventually skip the re-read and use stale values.
        """
        ...

    async def set_staleness(self, source_id: str, tenant_id: str, state: StalenessState) -> None:
        """Move a source to a staleness state.

        A state transition rather than a timestamp write. The transitions are what trigger
        action — entering STALE enqueues a reindex — and a caller computing the state itself
        would put that policy in as many places as there are callers.
        """
        ...

    async def delete(self, source_id: str, tenant_id: str) -> bool:
        """Remove a source, returning whether it existed.

        Part of the right-to-erasure workflow, so it must be idempotent and must report
        honestly: the audit log records what was actually erased, not what was requested.
        """
        ...


@runtime_checkable
class AdapterRepository(Protocol):
    """Stores the adapter registry.

    Rows are immutable. A change produces a new version, because an audit that cannot establish
    which weights answered a given request is not an audit.
    """

    async def get(self, adapter_id: str, version: str) -> AdapterRecord | None: ...

    async def register(self, record: AdapterRecord) -> AdapterRecord:
        """Write a new adapter version.

        Must reject an attempt to overwrite an existing ``(adapter_id, version)`` rather than
        replacing it. Silently overwriting would make the registry's immutability a convention
        instead of a guarantee.
        """
        ...

    async def find_servable(
        self, tenant_id: str, *, tier: int | None = None
    ) -> Sequence[AdapterRecord]:
        """Every adapter that may serve this tenant right now.

        The tenant filter is applied by the store, not by the caller. It is the isolation
        boundary that cannot be rechecked after the fact — once a delta is merged there is no
        per-request filter that takes it back out — so it is enforced at the lowest layer that
        can enforce it.
        """
        ...

    async def set_status(self, adapter_id: str, version: str, status: AdapterStatus) -> None:
        """Move an adapter through its lifecycle.

        Status is the one mutable field on an otherwise immutable row, because promotion and
        revocation are transitions of the same artifact rather than new artifacts.
        """
        ...

    async def find_containing_documents(
        self, document_ids: Sequence[str]
    ) -> Sequence[AdapterRecord]:
        """Every adapter trained on any of these documents.

        The query that makes revocation tractable. When a document must be erased, this
        identifies what has to be revoked and retrained; without it, erasure could only be
        honoured by rebuilding every adapter.
        """
        ...


@runtime_checkable
class LineageRepository(Protocol):
    """Stores the derivation graph.

    Read on the retrieval path to compute independence, so lookups must be cheap. Everything
    here is either a single-hop read or a bounded traversal; unbounded ancestor walks belong in
    a background job.
    """

    async def record(self, edges: Sequence[LineageEdge]) -> None: ...

    async def root_for(self, artifact_id: str) -> str:
        """The ultimate ancestor of an artifact.

        Returns the artifact's own id when it has no parent, so callers never special-case a
        root. Must terminate even on a cyclic graph: ingestion connectors do occasionally
        produce cycles, and a retrieval-path traversal that loops is an outage.
        """
        ...

    async def children_of(self, artifact_id: str) -> Sequence[LineageEdge]: ...


@runtime_checkable
class EvalRepository(Protocol):
    """Stores evaluation results.

    Written by three different triggers — offline runs, the CI gate, online sampling — against
    one schema, so that offline numbers and online numbers are comparable. Separate schemas
    would drift, and the offline suite would stop predicting anything.
    """

    async def record(self, run_id: str, results: Sequence[MetricResult]) -> None: ...

    async def results_for_run(self, run_id: str) -> Sequence[MetricResult]: ...

    async def latest_run_id(self, dataset_id: str) -> str | None:
        """The most recent run against a dataset, for diffing a scorecard against its baseline.

        ``None`` when the dataset has never been run, which is the cold-start case a regression
        gate must handle by passing rather than by failing on a missing baseline.
        """
        ...
