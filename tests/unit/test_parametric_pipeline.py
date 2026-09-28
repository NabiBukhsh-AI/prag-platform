"""The offline parameterization pipeline, run end to end with the local stand-in trainer.

These test the pipeline's mechanics and gates — refusal at each step, the augmentation filter,
held-out recall, interference, promotion — not whether LoRA learns. See ``prag.parametric.local``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prag.core.models.common import Deadline, EmbeddingPurpose
from prag.core.models.parametric import GLOBAL_TENANT_SCOPE, AdapterStatus, ProbeResults
from prag.ingestion.embedding import HashingEmbeddingProvider
from prag.parametric import (
    EligibilityGate,
    MemorizingTrainer,
    Passage,
    QAPair,
    answer_from,
    cluster_passages,
    filter_pairs,
    parameterize_cluster,
    probe,
    promote_from_shadow,
    promotion_decision,
    template_augmenter,
)
from tests.unit.test_parametric import a_record, a_registry, economics

EMBEDDER = HashingEmbeddingProvider(dimensions=64)
SEED = Path(__file__).resolve().parents[2] / "eval" / "seed" / "corpus" / "tenant-local" / "public"
UNRELATED = ("what is the capital of france", "how do i bake sourdough bread at home")


async def passages(document_id: str, text: str) -> list[Passage]:
    paragraphs = [p for p in text.split("\n\n") if len(p.split()) > 12]
    vectors = await EMBEDDER.embed(paragraphs, EmbeddingPurpose.DOCUMENT, Deadline.in_ms(5_000))
    return [
        Passage(f"{document_id}#{i}", document_id, p, tuple(v))
        for i, (p, v) in enumerate(zip(paragraphs, vectors, strict=True))
    ]


async def runbook_cluster():
    text = (SEED / "runbook.incident-response.md").read_text(encoding="utf-8")
    (cluster,) = cluster_passages(await passages("runbook.incident-response", text), threshold=0.0)
    return cluster


class TestClustering:
    async def test_related_passages_group_and_unrelated_ones_do_not(self) -> None:
        runbook = (SEED / "runbook.incident-response.md").read_text(encoding="utf-8")
        policy = (SEED / "policy.access-control.md").read_text(encoding="utf-8")
        mixed = await passages("r", runbook) + await passages("p", policy)

        clusters = cluster_passages(mixed, threshold=0.35)
        assert len(clusters) > 1
        for cluster in clusters:
            assert 0.0 < cluster.coherence <= 1.0

    async def test_clustering_is_deterministic(self) -> None:
        text = (SEED / "policy.access-control.md").read_text(encoding="utf-8")
        items = await passages("p", text)
        first = cluster_passages(items, threshold=0.35)
        second = cluster_passages(items, threshold=0.35)
        assert [c.passages for c in first] == [c.passages for c in second]


class TestAugmentation:
    async def test_every_sentence_gets_three_surfaces_answered_by_itself(self) -> None:
        cluster = await runbook_cluster()
        pairs = template_augmenter(cluster.passages[0])

        assert pairs
        assert {p.surface for p in pairs} == {0, 1, 2}
        assert all(p.answer in cluster.passages[0].text.replace("\n", " ") for p in pairs)

    def test_the_filter_drops_what_the_source_does_not_entail(self) -> None:
        """A non-entailed synthetic pair teaches a falsehood with no way to revoke it."""
        source = {"d": "Incident records are retained for 30 days before archival."}
        pairs = [
            QAPair("how long retained", "Incident records are retained for 30 days.", "d"),
            QAPair("how long retained", "Records are kept forever in the vault.", "d"),
            QAPair("retained how long?", "Incident records are retained for 30 days.", "d"),
            QAPair("x", "Incident records are retained for 30 days.", "d"),
        ]
        kept, dropped = filter_pairs(pairs, source)

        assert [p.question for p in kept] == ["how long retained"]
        assert dropped == {"not_entailed": 1, "duplicate": 1, "low_diversity": 1}


class TestProbes:
    async def test_held_out_phrasings_recall_the_fact(self) -> None:
        cluster = await runbook_cluster()
        pairs = [p for passage in cluster.passages for p in template_augmenter(passage)]
        weights = MemorizingTrainer().train(p for p in pairs if p.surface != 2)

        results = probe(weights, [p for p in pairs if p.surface == 2], general_probes=UNRELATED)
        assert results.knowledge_recall >= 0.8
        assert results.general_regression == 0.0

    async def test_a_conflicting_sibling_shows_up_as_interference(self) -> None:
        cluster = await runbook_cluster()
        pairs = [p for passage in cluster.passages for p in template_augmenter(passage)]
        held_out = [p for p in pairs if p.surface == 2]
        weights = MemorizingTrainer().train(p for p in pairs if p.surface != 2)
        sibling = MemorizingTrainer().train(
            QAPair(p.question, "zzz a different answer", "other") for p in held_out
        )

        assert probe(weights, held_out, siblings=[sibling]).interference_delta > 0.5
        assert probe(weights, held_out).interference_delta == 0.0

    def test_merged_answers_carry_no_citation_only_lineage(self) -> None:
        weights = MemorizingTrainer().train([QAPair("retention window", "Kept 30 days.", "d9")])
        answer = answer_from([weights], "what is the retention window")
        assert answer.text == "Kept 30 days."
        assert "[E" not in answer.text
        assert answer.document_ids == ("d9",)

    @pytest.mark.parametrize(
        ("probes", "reason"),
        [
            (ProbeResults(knowledge_recall=0.5, general_regression=0, interference_delta=0),
             "knowledge_recall_below_floor"),
            (ProbeResults(knowledge_recall=0.9, general_regression=0.1, interference_delta=0),
             "general_regression_above_tolerance"),
            (ProbeResults(knowledge_recall=0.9, general_regression=0, interference_delta=0.2),
             "interference_above_tolerance"),
        ],
    )
    def test_each_promotion_criterion_blocks_alone(self, probes, reason) -> None:
        assert promotion_decision(probes) == (reason,)

    def test_all_three_passing_promotes(self) -> None:
        probes = ProbeResults(knowledge_recall=0.9, general_regression=0.0, interference_delta=0.0)
        assert promotion_decision(probes) == ()


async def run(cluster, **overrides):
    registry = overrides.pop("registry", None) or a_registry()
    fields = {
        "knowledge": a_record(tenant_id="tenant-a"),
        "economics": economics(),
        "gate": EligibilityGate(),
        "registry": registry,
        "tenant_scope": "tenant-a",
        "base_model_id": "base",
        "base_model_version": "base-v1",
        "embedding_model_version": EMBEDDER.model_version,
        "domain": "ops",
        "trainer": MemorizingTrainer(),
        "general_probes": UNRELATED,
    }
    return await parameterize_cluster(cluster, **{**fields, **overrides}), registry


class TestWorkflow:
    async def test_a_good_cluster_ends_in_shadow_with_its_audit_trail(self) -> None:
        report, registry = await run(await runbook_cluster())

        assert report.steps == [
            "scope", "eligibility", "augmentation", "training", "promotion_gate",
            "registered", "shadow",
        ]
        record = report.record
        assert record is not None
        assert record.status is AdapterStatus.SHADOW
        assert record.eligibility_snapshot is not None
        assert record.eligibility_snapshot.eligible
        assert record.probe_results == report.probes
        assert record.source_document_ids == ("runbook.incident-response",)
        assert record.blob_sha256
        assert registry.servable("tenant-a") == (), "shadow adapters never serve"

    async def test_tenant_knowledge_never_trains_a_global_adapter(self) -> None:
        """The cross-tenant canary at training time: a global adapter serves every tenant."""
        report, registry = await run(await runbook_cluster(), tenant_scope=GLOBAL_TENANT_SCOPE)

        assert report.refused == ("tenant_knowledge_in_global_adapter",)
        assert not report.registered
        assert registry.servable("tenant-b") == ()

    async def test_another_tenants_scope_is_refused(self) -> None:
        report, _ = await run(await runbook_cluster(), tenant_scope="tenant-b")
        assert report.refused == ("adapter_scope_differs_from_knowledge_tenant",)

    async def test_ineligible_knowledge_is_never_trained(self) -> None:
        calls = []

        class Spy:
            def train(self, pairs):
                calls.append(pairs)
                return b""

        report, _ = await run(
            await runbook_cluster(),
            knowledge=a_record(tenant_id="tenant-a", contains_pii=True),
            trainer=Spy(),
        )
        assert report.refused == ("contains_pii",)
        assert calls == [], "gate before spend"

    async def test_an_augmenter_that_invents_facts_stops_the_run(self) -> None:
        def hallucinating(passage):
            return [
                QAPair(f"question {i} about it", "The moon is cheese.", passage.document_id, i % 3)
                for i in range(9)
            ]

        cluster = await runbook_cluster()
        report, _ = await run(cluster, augmenter=hallucinating)
        assert report.refused == ("augmenter_not_entailed_rate_too_high",)
        assert report.dropped["not_entailed"] == 9 * len(cluster.passages)

    async def test_a_failing_probe_is_not_registered(self) -> None:
        report, _ = await run(await runbook_cluster(), min_recall=1.01)
        assert "knowledge_recall_below_floor" in report.refused
        assert not report.registered

    async def test_retraining_a_cluster_produces_a_new_version(self) -> None:
        cluster = await runbook_cluster()
        first, registry = await run(cluster)
        second, _ = await run(cluster, registry=registry)

        assert first.record is not None
        assert second.record is not None
        assert (first.record.version, second.record.version) == ("1", "2")


class TestShadowPromotion:
    async def test_a_faithfulness_regression_keeps_it_in_shadow(self) -> None:
        report, registry = await run(await runbook_cluster())
        assert report.record is not None

        reasons = await promote_from_shadow(
            registry, report.record, parametric_faithfulness=0.80, baseline_faithfulness=0.95
        )
        assert reasons == ("shadow_faithfulness_regression",)
        assert registry.servable("tenant-a") == ()

    async def test_parity_promotes_and_deprecates_the_old_version(self) -> None:
        cluster = await runbook_cluster()
        first, registry = await run(cluster)
        assert first.record is not None
        await promote_from_shadow(
            registry, first.record, parametric_faithfulness=0.95, baseline_faithfulness=0.95
        )
        second, _ = await run(cluster, registry=registry)
        assert second.record is not None
        await promote_from_shadow(
            registry, second.record, parametric_faithfulness=0.96, baseline_faithfulness=0.95
        )

        assert [r.version for r in registry.servable("tenant-a")] == ["2"]
        old = await registry.repository.get(first.record.adapter_id, "1")
        assert old is not None
        assert old.status is AdapterStatus.DEPRECATED
