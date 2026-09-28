"""Metrics, scorecards, the regression gate, online sampling, and judge calibration."""

from __future__ import annotations

import math

import pytest

from prag.core.models.events import EvalInputs, EvalSample, MetricResult
from prag.core.protocols import Evaluator
from prag.evaluation import (
    NOT_APPLICABLE,
    FunctionMetric,
    GoldenCase,
    JudgeCalibration,
    MetricSummary,
    Scorecard,
    coverage_at_k,
    hit_rate_at_k,
    load_cases,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    regression_gate,
    sample_from_state,
    sampled,
    score_samples,
    standard_metrics,
)


def a_sample(sample_id: str = "s1", **fields: object) -> EvalSample:
    return EvalSample(sample_id=sample_id, query="q", **fields)


class TestRetrievalMetrics:
    RETRIEVED = ("d1", "d2", "d1", "d3", "d4")  # d1 twice: one document, two chunks

    def test_recall(self) -> None:
        assert recall_at_k(self.RETRIEVED, ("d2", "d9"), 5) == 0.5

    def test_precision_counts_documents_not_chunks(self) -> None:
        """Five chunks of one document are one retrieved document."""
        assert precision_at_k(self.RETRIEVED, ("d1",), 2) == 0.5

    def test_precision_is_over_what_was_returned(self) -> None:
        assert precision_at_k(("d1",), ("d1",), 5) == 1.0
        assert precision_at_k((), ("d1",), 5) == 0.0

    def test_hit_rate(self) -> None:
        assert hit_rate_at_k(self.RETRIEVED, ("d4",), 4) == 1.0
        assert hit_rate_at_k(self.RETRIEVED, ("d4",), 3) == 0.0

    def test_mrr(self) -> None:
        assert mrr(self.RETRIEVED, ("d3",)) == pytest.approx(1 / 3)
        assert mrr(self.RETRIEVED, ("d9",)) == 0.0

    def test_ndcg_is_one_for_a_perfect_ranking(self) -> None:
        assert ndcg_at_k(("d1", "d2", "d3"), ("d1", "d2"), 3) == 1.0

    def test_ndcg_rewards_rank(self) -> None:
        expected = (1 / math.log2(3)) / 1.0
        assert ndcg_at_k(("d9", "d1"), ("d1",), 2) == pytest.approx(expected)

    def test_coverage_counts_aspects_not_documents(self) -> None:
        """Three documents about one aspect cover less than one about each of two."""
        aspects = {"escalation": ("d1",), "retention": ("d7",)}
        assert coverage_at_k(("d1", "d2", "d3"), aspects, 3) == 0.5


class TestFunctionMetric:
    def test_it_satisfies_the_protocol(self) -> None:
        metric = FunctionMetric("m", lambda s: 1.0, requires=EvalInputs())
        assert isinstance(metric, Evaluator)

    async def test_a_threshold_sets_passed(self) -> None:
        metric = FunctionMetric("m", lambda s: 0.5, requires=EvalInputs(), threshold=0.6)
        assert (await metric.score(a_sample())).passed is False

    async def test_no_threshold_leaves_passed_unset(self) -> None:
        metric = FunctionMetric("m", lambda s: 0.5, requires=EvalInputs())
        assert (await metric.score(a_sample())).passed is None

    async def test_an_unscorable_sample_is_marked_not_applicable(self) -> None:
        metric = FunctionMetric("m", lambda s: None, requires=EvalInputs())
        result = await metric.score(a_sample())
        assert result.detail == NOT_APPLICABLE


class TestStandardMetrics:
    @staticmethod
    async def scores(sample: EvalSample) -> dict[str, MetricResult]:
        return {m.metric_id: await m.score(sample) for m in standard_metrics()}

    async def test_an_unlabelled_sample_skips_retrieval_metrics(self) -> None:
        """A missing label is excluded from the mean, never counted as a zero."""
        results = await self.scores(a_sample(retrieved_document_ids=("d1",)))
        assert results["recall@5"].detail == NOT_APPLICABLE

    async def test_an_abstention_still_counts_as_a_retrieval_miss(self) -> None:
        results = await self.scores(
            a_sample(relevant_document_ids=("d1",), metadata={"abstained": True})
        )
        assert results["recall@5"].score == 0.0
        assert results["recall@5"].detail != NOT_APPLICABLE

    async def test_citation_precision(self) -> None:
        results = await self.scores(
            a_sample(relevant_document_ids=("d1",), citations=("d1", "d2"))
        )
        assert results["citation_precision"].score == 0.5
        assert results["citation_precision"].passed is False

    async def test_an_uncited_answer_has_zero_citation_precision(self) -> None:
        results = await self.scores(a_sample(relevant_document_ids=("d1",), answer="x"))
        assert results["citation_precision"].score == 0.0

    async def test_groundedness_from_claim_counts(self) -> None:
        results = await self.scores(a_sample(metadata={"claims_total": 4, "claims_cited": 3}))
        assert results["groundedness"].score == 0.75

    async def test_completeness_is_case_insensitive(self) -> None:
        sample = a_sample(
            answer="Paged within 15 Minutes.", metadata={"expected_facts": ("15 minutes",)}
        )
        results = await self.scores(sample)
        assert results["completeness"].score == 1.0

    async def test_abstention_correctness(self) -> None:
        right = await self.scores(a_sample(metadata={"abstained": True, "expect_abstain": True}))
        wrong = await self.scores(a_sample(metadata={"abstained": False, "expect_abstain": True}))
        assert right["abstention_correct"].score == 1.0
        assert wrong["abstention_correct"].score == 0.0

    @pytest.mark.parametrize(
        ("metadata", "answer", "score"),
        [
            ({"expected_outcome": "blocked", "outcome": "blocked"}, None, 1.0),
            ({"expected_outcome": "blocked", "outcome": "answered"}, "ok", 0.0),
            ({"forbidden_strings": ("Halcyon",), "outcome": "answered"}, "Halcyon Labs", 0.0),
            ({"forbidden_strings": ("Halcyon",), "outcome": "answered"}, "nothing", 1.0),
        ],
    )
    async def test_adversarial(self, metadata: dict, answer: str | None, score: float) -> None:
        results = await self.scores(a_sample(answer=answer, metadata=metadata))
        assert results["adversarial"].score == score


class TestScorecard:
    async def test_not_applicable_results_are_excluded_from_the_mean(self) -> None:
        metric = FunctionMetric(
            "m", lambda s: None if s.sample_id == "skip" else 1.0, requires=EvalInputs()
        )
        card = await score_samples(
            [metric], [a_sample("a"), a_sample("skip")], dataset_id="d"
        )
        assert card.summaries["m"].mean == 1.0
        assert card.summaries["m"].n == 1

    async def test_the_baseline_round_trips(self) -> None:
        metric = FunctionMetric("m", lambda s: 0.8, requires=EvalInputs(), threshold=0.5)
        card = await score_samples([metric], [a_sample()], dataset_id="d")
        restored = Scorecard.from_json(card.to_json())

        assert restored.dataset_id == "d"
        assert restored.summaries == card.summaries


def a_card(**means: float) -> Scorecard:
    return Scorecard(
        dataset_id="d",
        summaries={
            k: MetricSummary(metric_id=k, mean=v, n=10, pass_rate=None) for k, v in means.items()
        },
    )


class TestRegressionGate:
    def test_passes_above_the_floor(self) -> None:
        assert regression_gate(a_card(faith=0.95), floors={"faith": 0.92}).passed

    def test_fails_below_the_floor(self) -> None:
        gate = regression_gate(a_card(faith=0.90), floors={"faith": 0.92})
        assert not gate.passed
        assert "below floor" in gate.failures[0]

    def test_fails_on_regression_past_tolerance(self) -> None:
        gate = regression_gate(
            a_card(faith=0.95), floors={"faith": 0.5}, baseline=a_card(faith=0.99), tolerance=0.02
        )
        assert not gate.passed
        assert "regressed" in gate.failures[0]

    def test_tolerates_noise_within_tolerance(self) -> None:
        gate = regression_gate(
            a_card(faith=0.98), floors={"faith": 0.5}, baseline=a_card(faith=0.99), tolerance=0.02
        )
        assert gate.passed

    def test_a_metric_that_stopped_being_scored_fails(self) -> None:
        """A gate over nothing passes nothing."""
        gate = regression_gate(a_card(), floors={"faith": 0.9})
        assert not gate.passed

    def test_an_uncalibrated_judge_gates_nothing(self) -> None:
        card = Scorecard(
            dataset_id="d",
            summaries={"judged": MetricSummary("judged", 0.1, 10, None, judged=True)},
        )
        gate = regression_gate(card, floors={"judged": 0.9})
        assert gate.passed
        assert gate.ungated

        assert not regression_gate(card, floors={"judged": 0.9}, judge_calibrated=True).passed


class TestOnlineSampling:
    def test_it_is_deterministic_per_request(self) -> None:
        """Every component that asks gets the same answer, and a replay samples the same set."""
        assert sampled("req_abc", 0.5) == sampled("req_abc", 0.5)

    def test_the_rate_is_respected(self) -> None:
        hits = sum(sampled(f"req_{i}", 0.1) for i in range(5_000))
        assert 350 < hits < 650

    def test_the_boundaries(self) -> None:
        assert not sampled("req_x", 0.0)
        assert sampled("req_x", 1.0)


class TestJudgeCalibration:
    def test_too_few_labels_is_uncalibrated(self) -> None:
        judge = JudgeCalibration("large.reasoning", "1")
        for _ in range(10):
            judge.record(0.9, True)
        assert not judge.calibrated()

    def test_agreement_above_the_bar_is_calibrated(self) -> None:
        judge = JudgeCalibration("large.reasoning", "1")
        for i in range(60):
            judge.record(0.9 if i % 10 else 0.1, True)  # agrees 90% of the time
        assert judge.agreement() == pytest.approx(0.9)
        assert judge.calibrated()

    def test_confident_nonsense_is_uncalibrated(self) -> None:
        judge = JudgeCalibration("large.reasoning", "1")
        for i in range(60):
            judge.record(0.95, i % 2 == 0)  # always confident, right half the time
        assert not judge.calibrated()


class TestDatasets:
    def test_cases_load_skipping_comments(self, tmp_path) -> None:
        path = tmp_path / "set.jsonl"
        path.write_text(
            '# a comment\n\n{"case_id": "c1", "query": "q", "expected_facts": ["x"]}\n',
            encoding="utf-8",
        )
        (case,) = load_cases(path)
        assert case.case_id == "c1"
        assert case.expected_facts == ("x",)

    def test_the_committed_seed_sets_parse(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parents[2] / "eval" / "seed"
        assert len(load_cases(root / "golden.jsonl")) >= 10
        assert len(load_cases(root / "adversarial.jsonl")) >= 10

    def test_a_raised_request_samples_without_state(self) -> None:
        case = GoldenCase(case_id="c", query="q", expected_outcome="blocked")
        sample = sample_from_state(case, outcome="blocked", state=None)

        assert sample.answer is None
        assert sample.metadata["abstained"] is True
        assert sample.metadata["outcome"] == "blocked"
