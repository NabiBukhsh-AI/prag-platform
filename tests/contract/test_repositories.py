"""Conformance suites for the four repositories.

Every backend runs these. The assertions are the invariants that are cheap to hold in a dict
and easy to lose in SQL: tenant scoping on every read, immutable adapter rows, honest deletion
counts, and a lineage traversal that terminates on a cyclic graph.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from prag.core.errors import StorageError
from prag.core.models.common import VolatilityClass
from prag.core.models.events import MetricResult
from prag.core.models.knowledge import (
    AccessPolicy,
    AclMode,
    Authority,
    IndexingProfile,
    LineageEdge,
    SourceRecord,
    SourceTemporality,
    SourceType,
    StalenessState,
)
from prag.core.models.parametric import (
    GLOBAL_TENANT_SCOPE,
    AdapterRecord,
    AdapterStatus,
    AdapterTier,
)
from prag.core.protocols import (
    AdapterRepository,
    EvalRepository,
    KnowledgeRepository,
    LineageRepository,
)
from prag.storage.repositories.in_memory import (
    InMemoryAdapterRepository,
    InMemoryEvalRepository,
    InMemoryKnowledgeRepository,
    InMemoryLineageRepository,
)

pytestmark = pytest.mark.contract

# One registry per repository. Adding a backend here runs every assertion below against it.
knowledge_repositories = [InMemoryKnowledgeRepository]
adapter_repositories = [InMemoryAdapterRepository]
lineage_repositories = [InMemoryLineageRepository]
eval_repositories = [InMemoryEvalRepository]


@pytest.fixture(params=knowledge_repositories, ids=lambda f: f.__name__)
def knowledge_repo(request: pytest.FixtureRequest) -> Any:
    return request.param()


@pytest.fixture(params=adapter_repositories, ids=lambda f: f.__name__)
def adapter_repo(request: pytest.FixtureRequest) -> Any:
    return request.param()


@pytest.fixture(params=lineage_repositories, ids=lambda f: f.__name__)
def lineage_repo(request: pytest.FixtureRequest) -> Any:
    return request.param()


@pytest.fixture(params=eval_repositories, ids=lambda f: f.__name__)
def eval_repo(request: pytest.FixtureRequest) -> Any:
    return request.param()


def a_source(
    *,
    source_id: str = "kb.policies",
    tenant_id: str = "tenant-a",
    staleness: StalenessState = StalenessState.FRESH,
    domain_overrides: dict[str, float] | None = None,
) -> SourceRecord:
    return SourceRecord(
        source_id=source_id,
        tenant_id=tenant_id,
        type=SourceType.DOCUMENT_COLLECTION,
        authority=Authority(
            base_score=0.9,
            domain_overrides=domain_overrides or {},
            rationale="system_of_record",
        ),
        temporality=SourceTemporality(
            volatility_class=VolatilityClass.SLOW,
            expected_half_life_days=180.0,
            staleness_state=staleness,
        ),
        indexing=IndexingProfile(
            embedding_model="text-embedding-3-large",
            embedding_model_version="v1",
            embedding_dim=3072,
            chunking_strategy="structure_aware_clause",
            index_targets=("vector.primary", "lexical.primary"),
        ),
        access=AccessPolicy(acl_mode=AclMode.GROUP, acl_groups=("hr",), acl_hash="sha256:x"),
        created_at_ms=int(time.time() * 1000),
    )


def an_adapter(
    *,
    adapter_id: str = "ad-1",
    version: str = "v1",
    scope: str = GLOBAL_TENANT_SCOPE,
    status: AdapterStatus = AdapterStatus.ACTIVE,
    tier: AdapterTier = AdapterTier.CLUSTER_KNOWLEDGE,
    documents: tuple[str, ...] = (),
) -> AdapterRecord:
    return AdapterRecord(
        adapter_id=adapter_id,
        version=version,
        tier=tier,
        base_model_id="mid.instruct",
        base_model_version="1",
        tenant_scope=scope,
        rank=8,
        alpha=16.0,
        status=status,
        source_document_ids=documents,
        created_at_ms=int(time.time() * 1000),
    )


class TestKnowledgeRepository:
    def test_satisfies_protocol(self, knowledge_repo: Any) -> None:
        assert isinstance(knowledge_repo, KnowledgeRepository)

    async def test_upsert_returns_what_was_stored(self, knowledge_repo: Any) -> None:
        """The store may set fields the caller did not, so it returns the stored row.

        A caller that has to re-read to learn what it just wrote will eventually skip the
        re-read and carry stale values forward.
        """
        stored = await knowledge_repo.upsert(a_source())
        assert stored.updated_at_ms is not None

    async def test_reads_are_tenant_scoped(self, knowledge_repo: Any) -> None:
        """Matching on source id alone must never succeed across tenants."""
        await knowledge_repo.upsert(a_source(tenant_id="tenant-a"))

        assert await knowledge_repo.get("kb.policies", "tenant-a") is not None
        assert await knowledge_repo.get("kb.policies", "tenant-b") is None

    async def test_same_source_id_in_two_tenants_is_two_records(self, knowledge_repo: Any) -> None:
        """Source ids are not globally unique; two tenants may both have `kb.policies`."""
        await knowledge_repo.upsert(a_source(tenant_id="tenant-a"))
        await knowledge_repo.upsert(a_source(tenant_id="tenant-b"))

        assert len(await knowledge_repo.list_for_tenant("tenant-a")) == 1
        assert len(await knowledge_repo.list_for_tenant("tenant-b")) == 1

    async def test_upsert_replaces(self, knowledge_repo: Any) -> None:
        await knowledge_repo.upsert(a_source())
        await knowledge_repo.upsert(a_source(domain_overrides={"legal": 0.98}))

        found = await knowledge_repo.get("kb.policies", "tenant-a")
        assert found is not None
        assert found.authority_for("legal") == pytest.approx(0.98)
        assert found.authority_for("engineering") == pytest.approx(0.9), "falls back to base"

    async def test_list_filters_by_staleness(self, knowledge_repo: Any) -> None:
        await knowledge_repo.upsert(a_source(source_id="fresh.one"))
        await knowledge_repo.upsert(a_source(source_id="stale.one", staleness=StalenessState.STALE))

        stale = await knowledge_repo.list_for_tenant("tenant-a", staleness=StalenessState.STALE)
        assert [r.source_id for r in stale] == ["stale.one"]

    async def test_set_staleness_transitions(self, knowledge_repo: Any) -> None:
        await knowledge_repo.upsert(a_source())
        await knowledge_repo.set_staleness("kb.policies", "tenant-a", StalenessState.STALE)

        found = await knowledge_repo.get("kb.policies", "tenant-a")
        assert found is not None
        assert found.temporality.staleness_state is StalenessState.STALE
        assert found.is_stale

    async def test_set_staleness_on_unknown_source_raises(self, knowledge_repo: Any) -> None:
        with pytest.raises(StorageError):
            await knowledge_repo.set_staleness("nope", "tenant-a", StalenessState.STALE)

    async def test_delete_reports_honestly(self, knowledge_repo: Any) -> None:
        """The erasure audit records what was erased, not what was requested."""
        await knowledge_repo.upsert(a_source())

        assert await knowledge_repo.delete("kb.policies", "tenant-a") is True
        assert await knowledge_repo.delete("kb.policies", "tenant-a") is False

    async def test_delete_is_tenant_scoped(self, knowledge_repo: Any) -> None:
        await knowledge_repo.upsert(a_source(tenant_id="tenant-a"))
        assert await knowledge_repo.delete("kb.policies", "tenant-b") is False
        assert await knowledge_repo.get("kb.policies", "tenant-a") is not None


class TestAdapterRepository:
    def test_satisfies_protocol(self, adapter_repo: Any) -> None:
        assert isinstance(adapter_repo, AdapterRepository)

    async def test_register_and_get(self, adapter_repo: Any) -> None:
        await adapter_repo.register(an_adapter())
        assert await adapter_repo.get("ad-1", "v1") is not None

    async def test_rows_are_immutable(self, adapter_repo: Any) -> None:
        """Re-registering a version must be refused rather than silently replacing it.

        A registry that can be overwritten cannot answer which weights served a given request,
        which is the whole reason it exists.
        """
        await adapter_repo.register(an_adapter())
        with pytest.raises(StorageError, match="already registered"):
            await adapter_repo.register(an_adapter())

    async def test_versions_coexist(self, adapter_repo: Any) -> None:
        await adapter_repo.register(an_adapter(version="v1"))
        await adapter_repo.register(an_adapter(version="v2"))

        assert await adapter_repo.get("ad-1", "v1") is not None
        assert await adapter_repo.get("ad-1", "v2") is not None

    async def test_find_servable_enforces_tenant_scope(self, adapter_repo: Any) -> None:
        """The isolation boundary that cannot be rechecked after the fact.

        Applied by the store rather than left to the caller, because once a delta is merged
        into the serving weights there is no per-request filter that takes it back out.
        """
        await adapter_repo.register(an_adapter(adapter_id="global", scope=GLOBAL_TENANT_SCOPE))
        await adapter_repo.register(an_adapter(adapter_id="private-a", scope="tenant-a"))
        await adapter_repo.register(an_adapter(adapter_id="private-b", scope="tenant-b"))

        for_a = {r.adapter_id for r in await adapter_repo.find_servable("tenant-a")}
        assert for_a == {"global", "private-a"}
        assert "private-b" not in for_a

    @pytest.mark.parametrize(
        "status",
        [
            AdapterStatus.CANDIDATE,
            AdapterStatus.SHADOW,
            AdapterStatus.DEPRECATED,
            AdapterStatus.REVOKED,
        ],
    )
    async def test_only_active_adapters_are_servable(
        self, adapter_repo: Any, status: AdapterStatus
    ) -> None:
        await adapter_repo.register(an_adapter(status=status))
        assert await adapter_repo.find_servable("tenant-a") == []

    async def test_find_servable_filters_by_tier(self, adapter_repo: Any) -> None:
        await adapter_repo.register(an_adapter(adapter_id="form", tier=AdapterTier.DOMAIN_FORM))
        await adapter_repo.register(
            an_adapter(adapter_id="knowledge", tier=AdapterTier.CLUSTER_KNOWLEDGE)
        )

        found = await adapter_repo.find_servable("tenant-a", tier=int(AdapterTier.DOMAIN_FORM))
        assert [r.adapter_id for r in found] == ["form"]

    async def test_promotion_stamps_the_time(self, adapter_repo: Any) -> None:
        await adapter_repo.register(an_adapter(status=AdapterStatus.CANDIDATE))
        await adapter_repo.set_status("ad-1", "v1", AdapterStatus.ACTIVE)

        found = await adapter_repo.get("ad-1", "v1")
        assert found is not None
        assert found.status is AdapterStatus.ACTIVE
        assert found.promoted_at_ms is not None

    async def test_revocation_removes_it_from_service(self, adapter_repo: Any) -> None:
        """Serving knowledge that was legally required to disappear is not a degraded state."""
        await adapter_repo.register(an_adapter())
        assert len(await adapter_repo.find_servable("tenant-a")) == 1

        await adapter_repo.set_status("ad-1", "v1", AdapterStatus.REVOKED)
        assert await adapter_repo.find_servable("tenant-a") == []

    async def test_set_status_on_unknown_adapter_raises(self, adapter_repo: Any) -> None:
        with pytest.raises(StorageError):
            await adapter_repo.set_status("nope", "v1", AdapterStatus.ACTIVE)

    async def test_find_by_document_makes_revocation_tractable(self, adapter_repo: Any) -> None:
        """When a document must be erased, this is what says which adapters contain it."""
        await adapter_repo.register(an_adapter(adapter_id="a", documents=("doc-1", "doc-2")))
        await adapter_repo.register(an_adapter(adapter_id="b", documents=("doc-2",)))
        await adapter_repo.register(an_adapter(adapter_id="c", documents=("doc-9",)))

        affected = {r.adapter_id for r in await adapter_repo.find_containing_documents(["doc-2"])}
        assert affected == {"a", "b"}

    async def test_find_by_document_spans_several_documents(self, adapter_repo: Any) -> None:
        await adapter_repo.register(an_adapter(adapter_id="a", documents=("doc-1",)))
        await adapter_repo.register(an_adapter(adapter_id="b", documents=("doc-2",)))

        affected = await adapter_repo.find_containing_documents(["doc-1", "doc-2"])
        assert len(affected) == 2

    async def test_unknown_document_matches_nothing(self, adapter_repo: Any) -> None:
        await adapter_repo.register(an_adapter(documents=("doc-1",)))
        assert await adapter_repo.find_containing_documents(["absent"]) == []


class TestLineageRepository:
    def test_satisfies_protocol(self, lineage_repo: Any) -> None:
        assert isinstance(lineage_repo, LineageRepository)

    def _edge(self, child: str, parent: str) -> LineageEdge:
        return LineageEdge(
            child_id=child,
            parent_id=parent,
            relation="derived_from",
            recorded_at_ms=int(time.time() * 1000),
        )

    async def test_root_of_an_orphan_is_itself(self, lineage_repo: Any) -> None:
        """So callers never special-case a root."""
        assert await lineage_repo.root_for("doc-1") == "doc-1"

    async def test_walks_to_the_root(self, lineage_repo: Any) -> None:
        """Three documents quoting one press release share a root, and are one source."""
        await lineage_repo.record(
            [
                self._edge("chunk-1", "doc-1"),
                self._edge("doc-1", "press-release"),
                self._edge("doc-2", "press-release"),
            ]
        )

        assert await lineage_repo.root_for("chunk-1") == "press-release"
        assert await lineage_repo.root_for("doc-2") == "press-release"

    async def test_cycle_terminates(self, lineage_repo: Any) -> None:
        """Connectors do occasionally produce cycles.

        A traversal that loops on the retrieval path is an outage, so this must return rather
        than hang — an imperfect independence score beats no answer.
        """
        await lineage_repo.record([self._edge("a", "b"), self._edge("b", "a")])
        assert await lineage_repo.root_for("a") in {"a", "b"}

    async def test_children_are_listed(self, lineage_repo: Any) -> None:
        await lineage_repo.record([self._edge("c1", "doc"), self._edge("c2", "doc")])
        assert {e.child_id for e in await lineage_repo.children_of("doc")} == {"c1", "c2"}

    async def test_childless_artifact_returns_empty(self, lineage_repo: Any) -> None:
        assert await lineage_repo.children_of("leaf") == ()


class TestEvalRepository:
    def test_satisfies_protocol(self, eval_repo: Any) -> None:
        assert isinstance(eval_repo, EvalRepository)

    def _result(self, metric: str = "faithfulness", score: float = 0.94) -> MetricResult:
        return MetricResult(metric_id=metric, sample_id="s1", score=score, passed=score >= 0.92)

    async def test_record_and_read_back(self, eval_repo: Any) -> None:
        await eval_repo.record("run-1", [self._result()])
        results = await eval_repo.results_for_run("run-1")

        assert len(results) == 1
        assert results[0].metric_id == "faithfulness"

    async def test_results_accumulate_within_a_run(self, eval_repo: Any) -> None:
        """Runners write in batches; a second write must extend rather than replace."""
        await eval_repo.record("run-1", [self._result("faithfulness")])
        await eval_repo.record("run-1", [self._result("citation_precision")])

        assert len(await eval_repo.results_for_run("run-1")) == 2

    async def test_unknown_run_is_empty_not_an_error(self, eval_repo: Any) -> None:
        assert await eval_repo.results_for_run("never-ran") == ()

    async def test_no_baseline_returns_none(self, eval_repo: Any) -> None:
        """The cold-start case a regression gate must handle by passing, not by failing."""
        assert await eval_repo.latest_run_id("golden-v1") is None
