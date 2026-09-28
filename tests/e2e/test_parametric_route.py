"""The parametric route end to end: train, promote, serve, shadow, conflict, demote, revoke.

Everything runs through ``Platform.answer`` with the local stand-in model (see
``prag.parametric.local``). The assertions are about routing, isolation, provenance and the
conflict loop — never about how well a real adapter would learn.

Coverage floor: the local hashing embedder scores query-to-centroid similarity around 0.1-0.5, so
the production floor of 0.62 (tuned for semantic embeddings) is lowered to 0.35 here. The
confidence floor still stops weak matches.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prag.api import build_platform
from prag.config import PragSettings
from prag.config.schema import (
    AdapterSelectionConfig,
    EvaluationConfig,
    ParametricConfig,
    RoutingConfig,
)
from prag.core.errors import IsolationViolation
from prag.core.models.common import Deadline, EmbeddingPurpose
from prag.core.models.events import EventKind
from prag.core.models.fusion import KnowledgeBasis
from prag.fusion import ConflictMonitor
from prag.parametric import (
    EligibilityGate,
    MemorizingTrainer,
    Passage,
    cluster_passages,
    parameterize_cluster,
    promote_from_shadow,
)
from tests.unit.test_parametric import a_record, economics

pytestmark = pytest.mark.e2e

SEED = Path(__file__).resolve().parents[2] / "eval" / "seed" / "corpus" / "tenant-local" / "public"
RUNBOOK = (SEED / "runbook.incident-response.md").read_text(encoding="utf-8")
POLICY = (SEED / "policy.access-control.md").read_text(encoding="utf-8")
TENANT = "tenant-local"
ESCALATION = "within how many minutes must the on-call lead be paged for a sev-1"


def settings() -> PragSettings:
    return PragSettings(
        parametric=ParametricConfig(
            enabled=True, selection=AdapterSelectionConfig(min_coverage_similarity=0.35)
        ),
        # The config refuses the parametric tier without calibrated evaluation. These tests use
        # no judge scores; the flag acknowledges the gate rather than claiming a judge exists.
        evaluation=EvaluationConfig(judge_calibrated=True),
        routing=RoutingConfig(exploration_fraction=0.0),
    )


async def train(platform, text: str, *, document_id: str = "runbook.incident-response"):
    """Parameterize one document for the tenant and promote it through shadow."""
    paragraphs = [p for p in text.split("\n\n") if len(p.split()) > 12]
    vectors = await platform.embedder.embed(
        paragraphs, EmbeddingPurpose.DOCUMENT, Deadline.in_ms(5_000)
    )
    passages = [
        Passage(f"{document_id}#{i}", document_id, p, tuple(v))
        for i, (p, v) in enumerate(zip(paragraphs, vectors, strict=True))
    ]
    (cluster,) = cluster_passages(passages, threshold=0.0)
    report = await parameterize_cluster(
        cluster,
        knowledge=a_record(tenant_id=TENANT),
        economics=economics(),
        gate=EligibilityGate(),
        registry=platform.parametric.registry,
        tenant_scope=TENANT,
        base_model_id="base.lora",
        base_model_version="1",
        embedding_model_version=platform.embedder.model_version,
        domain="general",
        trainer=MemorizingTrainer(),
    )
    assert report.record is not None, report.refused
    assert not await promote_from_shadow(
        platform.parametric.registry,
        report.record,
        parametric_faithfulness=1.0,
        baseline_faithfulness=1.0,
    )
    return report.record


async def a_platform(*, adapter_text: str | None = RUNBOOK):
    platform = build_platform(settings())
    await platform.ingest_markdown(
        RUNBOOK, document_id="runbook.incident-response", tenant_id=TENANT
    )
    await platform.ingest_markdown(POLICY, document_id="policy.access-control", tenant_id=TENANT)
    record = await train(platform, adapter_text) if adapter_text is not None else None
    return platform, record


async def ask(platform, query: str, *, tenant: str = TENANT):
    state = platform.request_state(query, {"x-tenant-id": tenant, "x-user-id": "u1"})
    return await platform.answer(state)


def events(run, kind: EventKind):
    return [e for e in run.state.events if e.kind is kind]


class TestParametricAnswer:
    async def test_a_covered_question_is_answered_from_the_adapter(self) -> None:
        platform, record = await a_platform()
        run = await ask(platform, ESCALATION)

        assert run.path == ("analyze", "parametric", "retrieve", "shadow")
        envelope = run.state.result
        assert envelope is not None
        assert "15 minutes" in envelope.answer
        assert envelope.confidence.basis is KnowledgeBasis.PARAMETRIC
        assert [a.adapter_id for a in envelope.diagnostics.adapters] == [record.adapter_id]
        assert events(run, EventKind.PARAMETRIC_SERVED)

    async def test_the_parametric_answer_is_shadow_cited_where_the_corpus_confirms_it(
        self,
    ) -> None:
        """Weights cannot cite; shadowing cites only what the corpus entails."""
        platform, _ = await a_platform()
        run = await ask(platform, ESCALATION)
        envelope = run.state.result

        assert envelope is not None
        assert envelope.citations
        assert all(c.document_id == "runbook.incident-response" for c in envelope.citations)
        assert envelope.grounding.claims_cited == envelope.grounding.claims_total

    async def test_the_answer_says_it_came_from_learned_knowledge(self) -> None:
        platform, _ = await a_platform()
        run = await ask(platform, ESCALATION)

        assert run.state.decision is not None
        assert run.state.decision.basis is KnowledgeBasis.PARAMETRIC
        assert "learned knowledge" in (run.state.decision.epistemic_marking or "")

    async def test_an_uncovered_question_takes_the_grounded_path(self) -> None:
        platform, _ = await a_platform()
        run = await ask(platform, "which roles does the platform define")

        assert run.path[-2:] == ("build_context", "generate")
        assert run.state.result is not None
        assert run.state.result.confidence.basis is KnowledgeBasis.RETRIEVED_EVIDENCE

    async def test_with_the_tier_disabled_nothing_changes(self) -> None:
        platform = build_platform(PragSettings())
        assert platform.parametric is None
        assert platform.engine._definition.graph_id == "standard_answer"


class TestEvidenceWins:
    STALE = RUNBOOK.replace("within 15 minutes of detection", "within 45 minutes of detection")

    async def test_a_contradicted_adapter_yields_to_the_corpus(self) -> None:
        """The corpus is the system of record. A stale adapter's number never reaches the user."""
        platform, record = await a_platform(adapter_text=self.STALE)
        run = await ask(platform, ESCALATION)

        assert run.path == (
            "analyze", "parametric", "retrieve", "shadow", "build_context", "generate"
        )
        envelope = run.state.result
        assert envelope is not None
        assert "45 minutes" not in envelope.answer
        assert "15 minutes" in envelope.answer
        assert envelope.confidence.basis is KnowledgeBasis.RETRIEVED_EVIDENCE

        (conflict,) = events(run, EventKind.PARAMETRIC_RETRIEVAL_CONFLICT)
        assert conflict.payload["adapters"] == [f"{record.adapter_id}@{record.version}"]
        assert "45 minutes" in conflict.payload["claim"]
        assert not events(run, EventKind.PARAMETRIC_SERVED)

    async def test_repeated_conflicts_demote_the_adapter_and_queue_a_retrain(self) -> None:
        """The loop from serving back to training, closed without a human."""
        platform, record = await a_platform(adapter_text=self.STALE)
        monitor = ConflictMonitor(min_samples=3, critical_rate=0.5)
        platform.parametric.monitor = monitor
        platform.events.subscribe(monitor.observe)

        for _ in range(3):
            await ask(platform, ESCALATION)

        key = f"{record.adapter_id}@{record.version}"
        assert monitor.rate(key) == 1.0
        assert key in platform.parametric.retrain_queue
        assert platform.parametric.registry.servable(TENANT) == ()

        after = await ask(platform, ESCALATION)
        assert "parametric" not in after.path, "demoted: straight to the grounded path"


class TestIsolation:
    async def test_another_tenant_never_reaches_the_adapter(self) -> None:
        platform, _ = await a_platform()
        await platform.ingest_markdown(RUNBOOK, document_id="runbook", tenant_id="tenant-other")

        run = await ask(platform, ESCALATION, tenant="tenant-other")
        assert "parametric" not in run.path
        assert not events(run, EventKind.PARAMETRIC_SERVED)

    async def test_the_serving_layer_refuses_a_foreign_adapter(self) -> None:
        """The last line: even a request that names the adapter directly is refused at load."""
        from prag.core.models.context import RegionName, RenderedRegion
        from prag.core.models.generation import GenerationRequest, ModelSpec
        from prag.core.models.parametric import AdapterRef
        from prag.parametric import LocalParametricProvider

        platform, record = await a_platform()
        provider = LocalParametricProvider(platform.parametric.registry.store)
        spec = ModelSpec(
            model_id="base.lora",
            model_version="1",
            provider_id="local.parametric",
            adapters=(
                AdapterRef(
                    adapter_id=record.adapter_id,
                    version=record.version,
                    tier=record.tier,
                    coverage=1.0,
                ),
            ),
            profile="grounded_extraction",
            context_window=8_000,
            cost_per_1k_in=0.0,
            cost_per_1k_out=0.0,
        )
        request = GenerationRequest(
            request_id="req_x",
            tenant_id="tenant-other",
            spec=spec,
            regions=(RenderedRegion(name=RegionName.QUERY, content=ESCALATION),),
        )
        with pytest.raises(IsolationViolation):
            await provider.generate(request, Deadline.in_ms(1_000))
        with pytest.raises(IsolationViolation):
            await provider.generate(
                request.model_copy(update={"tenant_id": None}), Deadline.in_ms(1_000)
            )


class TestRevocation:
    async def test_erasing_the_source_falls_back_to_retrieval(self) -> None:
        """Revocation degrades to non-parametric, never to serving stale weights."""
        platform, record = await a_platform()
        affected = await platform.parametric.registry.revoke_documents(
            ["runbook.incident-response"]
        )
        assert [r.adapter_id for r in affected] == [record.adapter_id]

        run = await ask(platform, ESCALATION)
        assert "parametric" not in run.path
        assert run.state.result is not None
        assert run.state.result.confidence.basis is KnowledgeBasis.RETRIEVED_EVIDENCE
