"""Stance detection, provenance shadowing, the conflict monitor, and adapter composition."""

from __future__ import annotations

import pytest

from prag.core.models.common import Deadline
from prag.core.models.events import DomainEvent, EventKind
from prag.core.models.fusion import Stance
from prag.core.models.parametric import CompositionMode
from prag.core.protocols import ProvenanceShadower
from prag.fusion import ConflictMonitor, EntailmentProvenanceShadower, stance
from prag.generation import HeuristicGroundingVerifier
from prag.parametric import MemorizingTrainer, QAPair, answer_from
from tests.unit.test_context import a_group


class TestStance:
    @pytest.mark.parametrize(
        ("claim", "evidence", "expected"),
        [
            ("Records are retained for 60 days.", "Records are retained for 30 days.",
             Stance.CONTRADICTS),
            ("Records are retained for 30 days.", "Records are retained for 30 days.",
             Stance.SUPPORTS),
            ("The lead is not paged at night.", "The lead is paged at night.", Stance.CONTRADICTS),
            ("Records are retained for 30 days.", "Sev-2 pages during business hours.", None),
        ],
    )
    def test_stance(self, claim: str, evidence: str, expected: Stance | None) -> None:
        assert stance(claim, evidence) is expected

    def test_a_claim_without_numbers_is_not_contradicted_by_one_with_them(self) -> None:
        assert stance("Records are retained.", "Records are retained for 30 days.") is (
            Stance.SUPPORTS
        )


class TestShadower:
    SHADOWER = EntailmentProvenanceShadower(HeuristicGroundingVerifier(entailment_threshold=0.5))

    def test_it_satisfies_the_protocol(self) -> None:
        assert isinstance(self.SHADOWER, ProvenanceShadower)

    async def test_supported_claims_are_cited_and_the_rest_unsourced(self) -> None:
        report = await self.SHADOWER.shadow(
            "Incident records are retained for 30 days. The moon orbits the earth monthly.",
            [a_group("Incident records are retained for 30 days before archival.")],
            ["a1@1"],
            Deadline.in_ms(1_000),
        )
        assert report.grounding.claims_cited == 1
        assert report.grounding.claims_unsourced == 1
        assert report.conflicts == ()

    async def test_a_contradiction_is_reported_with_the_adapter_that_made_it(self) -> None:
        report = await self.SHADOWER.shadow(
            "Incident records are retained for 90 days.",
            [a_group("Incident records are retained for 30 days before archival.")],
            ["a1@1"],
            Deadline.in_ms(1_000),
        )
        (conflict,) = report.conflicts
        assert conflict.adapter_ids == ("a1@1",)
        assert conflict.positions[-1].stance is Stance.CONTRADICTS
        assert "30 days" in (conflict.positions[-1].excerpt or "")

    async def test_direct_support_elsewhere_outweighs_a_similar_sentence(self) -> None:
        """Two sentences on one topic with different numbers are often two different facts."""
        report = await self.SHADOWER.shadow(
            "A sev-1 pages the lead within 15 minutes.",
            [
                a_group("A sev-1 pages the lead within 15 minutes.", group_id="g1"),
                a_group("A sev-1 pages the lead's manager within 20 minutes.", group_id="g2"),
            ],
            ["a1@1"],
            Deadline.in_ms(1_000),
        )
        assert report.conflicts == ()


def an_event(kind: EventKind, adapter: str = "a1@1") -> DomainEvent:
    return DomainEvent(
        event_id="e", kind=kind, request_id="r", tenant_id="t", occurred_at_ms=0,
        payload={"adapters": [adapter]},
    )


class TestConflictMonitor:
    def test_nothing_is_decided_below_the_sample_floor(self) -> None:
        monitor = ConflictMonitor(min_samples=5)
        for _ in range(4):
            monitor.observe(an_event(EventKind.PARAMETRIC_RETRIEVAL_CONFLICT))
        assert monitor.take() == (frozenset(), frozenset())

    def test_a_stale_rate_queues_a_retrain_without_demoting(self) -> None:
        monitor = ConflictMonitor(min_samples=10, staleness_rate=0.05, critical_rate=0.5)
        for i in range(10):
            conflicted = i == 0
            monitor.observe(
                an_event(
                    EventKind.PARAMETRIC_RETRIEVAL_CONFLICT
                    if conflicted
                    else EventKind.PARAMETRIC_SERVED
                )
            )

        retrain, demote = monitor.take()
        assert retrain == {"a1@1"}
        assert demote == frozenset()

    def test_a_critical_rate_demotes_once(self) -> None:
        monitor = ConflictMonitor(min_samples=4, critical_rate=0.2)
        for _ in range(8):
            monitor.observe(an_event(EventKind.PARAMETRIC_RETRIEVAL_CONFLICT))
        assert monitor.take()[1] == {"a1@1"}
        assert monitor.take()[1] == frozenset(), "decided once, not on every later event"

    def test_other_events_are_ignored(self) -> None:
        monitor = ConflictMonitor(min_samples=1)
        monitor.observe(an_event(EventKind.SECURITY_EVENT))
        assert monitor.rate("a1@1") == 0.0


class TestComposition:
    def test_a_weighted_merge_prefers_the_stronger_adapter(self) -> None:
        strong = MemorizingTrainer().train([QAPair("paging target lead", "Fifteen minutes.", "d1")])
        weak = MemorizingTrainer().train([QAPair("paging target lead", "Forty minutes.", "d2")])
        answer = answer_from([weak, strong], "paging target lead", weights_scale=[0.4, 0.9])
        assert answer.text == "Fifteen minutes."

    async def test_sequential_probe_tries_the_second_adapter_only_when_unsure(self) -> None:
        from prag.core.models.context import RegionName, RenderedRegion
        from prag.core.models.generation import GenerationRequest, ModelSpec
        from prag.core.models.parametric import AdapterRef, AdapterTier
        from prag.parametric import LocalParametricProvider
        from tests.unit.test_parametric import a_registry, an_adapter

        registry = a_registry()
        first = await an_adapter("first")
        second = await an_adapter("second")
        await registry.register(first, MemorizingTrainer().train([QAPair("billing", "x", "d")]))
        await registry.promote("first", "1")
        await registry.register(
            second, MemorizingTrainer().train([QAPair("paging target", "Fifteen.", "d")])
        )
        await registry.promote("second", "1")

        provider = LocalParametricProvider(
            registry.store, composition_mode=CompositionMode.SEQUENTIAL_PROBE
        )
        refs = tuple(
            AdapterRef(adapter_id=n, version="1", tier=AdapterTier.CLUSTER_KNOWLEDGE, coverage=c)
            for n, c in (("first", 0.9), ("second", 0.8))
        )
        spec = ModelSpec(
            model_id="base.lora", model_version="1", provider_id="local.parametric",
            adapters=refs, profile="p", context_window=1_000,
            cost_per_1k_in=0.0, cost_per_1k_out=0.0,
        )
        result = await provider.generate(
            GenerationRequest(
                request_id="r", tenant_id="t", spec=spec,
                regions=(RenderedRegion(name=RegionName.QUERY, content="paging target"),),
            ),
            Deadline.in_ms(1_000),
        )
        assert result.text == "Fifteen."
