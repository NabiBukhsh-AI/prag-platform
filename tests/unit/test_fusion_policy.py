"""The fusion decision policy: one test per row of the §11.4 table, plus calibration.

The table is the specification. If a row changes, its test changes with it, deliberately.
"""

from __future__ import annotations

import random

import pytest

from prag.core.models.context import ContextBundle
from prag.core.models.fusion import (
    AbstentionCode,
    ConflictEvent,
    ConflictKind,
    ConflictPosition,
    ConflictResolution,
    KnowledgeBasis,
    ParametricSignal,
    RawConfidenceSignals,
    Stance,
)
from prag.core.models.query import Ambiguity, FieldPrediction, KnowledgeRequirement
from prag.core.protocols import ConfidenceCalibrator, FusionPolicy
from prag.fusion import (
    DEGRADED_RETRIEVAL_CONFIDENCE,
    IsotonicCalibrator,
    TablePolicy,
    expected_calibration_error,
    independent_agreement,
    raw_retrieval_score,
)
from tests.unit.test_context import a_group
from tests.unit.test_graph_engine import an_analysis
from tests.unit.test_intelligence import a_policy

NOW_MS = 1_800_000_000_000
DAY_MS = 86_400_000


def group(
    text: str,
    *,
    root: str = "doc-1",
    rerank: float | None = 0.9,
    authority: float = 0.8,
    updated_at_ms: int = 0,
    group_id: str | None = None,
):
    g = a_group(
        text,
        group_id=group_id or f"g-{root}-{abs(hash(text)) % 1000}",
        lineage_root=root,
        authority=authority,
        updated_at_ms=updated_at_ms,
    )
    rep = g.representative.model_copy(update={"rerank_score": rerank})
    return g.model_copy(update={"representative": rep, "members": (rep,)})


def bundle(*groups) -> ContextBundle:
    return ContextBundle(bundle_id="b", regions=(), evidence=groups, rendered_prompt_hash="h")


def analysis(*, private: bool = False, ambiguous: bool = False, half_life_days: float = 180.0):
    base = an_analysis()
    requirements = dict(base.knowledge_requirements)
    if private:
        requirements[KnowledgeRequirement.REQUIRES_PRIVATE_DATA] = FieldPrediction(
            value=True, confidence=0.9
        )
    return base.model_copy(
        update={
            "knowledge_requirements": requirements,
            "ambiguity": Ambiguity(
                is_ambiguous=ambiguous,
                clarification_candidates=("the sev-1 policy", "the sev-2 policy")
                if ambiguous
                else (),
            ),
            "temporality": base.temporality.model_copy(
                update={"estimated_half_life_days": half_life_days}
            ),
        }
    )


def parametric(confidence: float, answer: str = "") -> ParametricSignal:
    return ParametricSignal(
        raw_confidence=RawConfidenceSignals(),
        calibrated_confidence=confidence,
        calibrator_version="test",
        probe_answer=answer or None,
    )


POLICY = TablePolicy(clock=lambda: NOW_MS / 1000)
RETENTION = "Incident records are retained for 30 days before archival."


async def decide(context=None, signal=None, *, query=None, tenant=None, validation=None):
    return await POLICY.decide(
        query or analysis(), signal, context, validation, tenant or a_policy()
    )


class TestTheTable:
    def test_it_satisfies_the_protocol(self) -> None:
        assert isinstance(POLICY, FusionPolicy)

    async def test_retrieval_high_and_no_conflict_grounds_in_evidence(self) -> None:
        decision = await decide(bundle(group(RETENTION)))
        assert decision.basis is KnowledgeBasis.RETRIEVED_EVIDENCE
        assert decision.conflicts == ()
        assert decision.p_retrieval >= 0.7

    async def test_retrieval_high_and_parametric_conflicts_evidence_wins(self) -> None:
        stale = parametric(0.9, "Incident records are retained for 90 days before archival.")
        decision = await decide(bundle(group(RETENTION)), stale)

        assert decision.basis is KnowledgeBasis.RETRIEVED_EVIDENCE
        (conflict,) = decision.conflicts
        assert conflict.kind is ConflictKind.PARAMETRIC_VS_RETRIEVED
        assert conflict.resolution is ConflictResolution.EVIDENCE_WINS

    async def test_retrieval_moderate_and_parametric_agrees_boosts_confidence(self) -> None:
        moderate = bundle(group(RETENTION, rerank=0.5))
        alone = await decide(moderate)
        agreed = await decide(moderate, parametric(0.8, RETENTION))

        assert agreed.basis is KnowledgeBasis.RETRIEVED_EVIDENCE
        assert agreed.knowledge_score > alone.knowledge_score

    async def test_retrieval_low_parametric_high_generic_answers_parametrically(self) -> None:
        decision = await decide(None, parametric(0.9, "Water boils at 100 degrees."))
        assert decision.basis is KnowledgeBasis.PARAMETRIC
        assert "general knowledge" in (decision.epistemic_marking or "")

    async def test_retrieval_low_parametric_high_private_abstains(self) -> None:
        """The hard gate: no score overrides it."""
        decision = await decide(
            None, parametric(0.99, "Your contract renews in March."), query=analysis(private=True)
        )
        assert decision.basis is KnowledgeBasis.ABSTAIN
        assert decision.abstention is not None
        assert decision.abstention.reason_code is AbstentionCode.PRIVATE_QUERY_NO_EVIDENCE

    async def test_comparable_authority_conflicts_are_surfaced(self) -> None:
        decision = await decide(
            bundle(
                group(RETENTION, root="runbook", authority=0.8),
                group("Incident records are retained for 90 days before archival.",
                      root="wiki", authority=0.75),
            )
        )
        (conflict,) = decision.conflicts
        assert conflict.kind is ConflictKind.SOURCE_VS_SOURCE
        assert conflict.resolution is ConflictResolution.SURFACED
        assert decision.basis is KnowledgeBasis.RETRIEVED_EVIDENCE

    async def test_separated_authority_prefers_the_stronger_source(self) -> None:
        decision = await decide(
            bundle(
                group("Incident records are retained for 90 days before archival.",
                      root="wiki", authority=0.3),
                group(RETENTION, root="runbook", authority=0.95),
            )
        )
        (conflict,) = decision.conflicts
        assert conflict.resolution is ConflictResolution.AUTHORITY_WINS
        assert conflict.positions[0].origin_id == "runbook", "stronger first; both cited"
        assert len(conflict.positions) == 2

    async def test_both_low_abstains(self) -> None:
        decision = await decide(bundle(group(RETENTION, rerank=0.2)), parametric(0.3))
        assert decision.abstention is not None
        assert decision.abstention.reason_code is AbstentionCode.KNOWLEDGE_BELOW_FLOOR

    async def test_both_low_and_ambiguous_asks_to_clarify(self) -> None:
        decision = await decide(None, None, query=analysis(ambiguous=True))
        assert decision.abstention is not None
        assert decision.abstention.reason_code is AbstentionCode.AMBIGUOUS_NEEDS_CLARIFICATION
        assert "sev-1 policy" in (decision.abstention.suggested_action or "")

    async def test_stale_evidence_warns(self) -> None:
        old = NOW_MS - 400 * DAY_MS
        decision = await decide(
            bundle(group(RETENTION, updated_at_ms=old)), query=analysis(half_life_days=180)
        )
        assert decision.basis is KnowledgeBasis.RETRIEVED_EVIDENCE
        assert decision.staleness is not None
        assert decision.staleness.staleness_ratio == pytest.approx(400 / 180)

    async def test_stale_evidence_abstains_in_strict_mode(self) -> None:
        old = NOW_MS - 400 * DAY_MS
        decision = await decide(
            bundle(group(RETENTION, updated_at_ms=old)),
            query=analysis(half_life_days=180),
            tenant=a_policy().model_copy(update={"strict_mode": True}),
        )
        assert decision.abstention is not None
        assert decision.abstention.reason_code is AbstentionCode.EVIDENCE_TOO_STALE

    async def test_fresh_or_undated_evidence_carries_no_warning(self) -> None:
        recent = NOW_MS - 10 * DAY_MS
        assert (await decide(bundle(group(RETENTION, updated_at_ms=recent)))).staleness is None
        assert (await decide(bundle(group(RETENTION)))).staleness is None


class TestRetrievalConfidence:
    def test_weak_evidence_does_not_corroborate_itself(self) -> None:
        """An off-topic retrieval must not be lifted over the answering line by its own noise."""
        weak = [group("a b c", root=f"r{i}", rerank=0.35 - i * 0.05) for i in range(4)]
        assert raw_retrieval_score(weak) < 0.4

    def test_independent_corroboration_raises_confidence(self) -> None:
        one = [group(RETENTION, root="a", rerank=0.8)]
        two = [*one, group("Records are archived after thirty days.", root="b", rerank=0.7)]
        assert raw_retrieval_score(two) > raw_retrieval_score(one)

    def test_a_skipped_reranker_is_moderate_not_absent(self) -> None:
        """Degradation sheds quality, not the answer."""
        assert raw_retrieval_score([group(RETENTION, rerank=None)]) == (
            DEGRADED_RETRIEVAL_CONFIDENCE
        )

    def test_one_lineage_counts_once(self) -> None:
        """Three copies of one press release are one piece of evidence (§11.3)."""
        syndicated = [
            group(f"copy {i}", root="press-release", authority=0.9, group_id=f"g{i}")
            for i in range(3)
        ]
        sources = [*syndicated, group("other", root="audit", authority=0.3)]
        dispute = ConflictEvent(
            conflict_id="c",
            kind=ConflictKind.SOURCE_VS_SOURCE,
            claim="x",
            positions=(
                ConflictPosition(origin_id="press-release", stance=Stance.SUPPORTS, authority=0.9),
                ConflictPosition(origin_id="audit", stance=Stance.CONTRADICTS, authority=0.3),
            ),
            resolution=ConflictResolution.AUTHORITY_WINS,
        )
        # Counted per copy, agreement would be 2.7 / 3.0 = 0.9. Per lineage it is 0.9 / 1.2.
        assert independent_agreement(sources, [dispute]) == pytest.approx(0.75)


class TestCalibration:
    def test_it_satisfies_the_protocol(self) -> None:
        assert isinstance(IsotonicCalibrator.identity(), ConfidenceCalibrator)

    def test_an_uncalibrated_calibrator_reports_the_worst_error(self) -> None:
        assert IsotonicCalibrator.identity().expected_calibration_error() == 1.0

    def test_the_fit_is_monotone(self) -> None:
        rng = random.Random(7)
        scores = [rng.random() for _ in range(300)]
        outcomes = [rng.random() < s**2 for s in scores]
        fitted = IsotonicCalibrator.fit(scores, outcomes, calibrator_version="t")
        grid = [fitted.calibrate_score(x / 20) for x in range(21)]
        assert grid == sorted(grid)

    def test_fitting_reduces_calibration_error_on_held_out_data(self) -> None:
        """Overconfident raw scores: the model says 0.9 and is right 40% of the time."""
        rng = random.Random(11)

        def sample(n: int) -> tuple[list[float], list[bool]]:
            scores = [rng.random() for _ in range(n)]
            return scores, [rng.random() < 0.3 + 0.2 * s for s in scores]

        train, held = sample(4_000), sample(5_000)  # held-out large enough that bin noise < 0.05
        raw_error = expected_calibration_error(held[0], held[1])
        fitted = IsotonicCalibrator.fit(*train, calibrator_version="t", held_out=held)

        assert fitted.expected_calibration_error() < raw_error / 3
        assert fitted.expected_calibration_error() < 0.05

    def test_parametric_signals_are_combined_before_calibration(self) -> None:
        signals = RawConfidenceSignals(adapter_coverage=0.8, self_consistency=0.6)
        assert IsotonicCalibrator.identity().calibrate(signals) == pytest.approx(0.7)
