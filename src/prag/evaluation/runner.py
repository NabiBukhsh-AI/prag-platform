"""Datasets, scorecards, the regression gate, online sampling, and judge calibration.

The runner never executes a request itself. It takes samples, however they were produced — a
replayed recorded state, a live run against the local stack, a sampled production request — and
scores them with the one metric set. Where a sample came from changes the data, not the scoring.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from prag.core.models.events import EvalSample, MetricResult
from prag.evaluation.metrics import NOT_APPLICABLE

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from prag.core.models.state import RequestState
    from prag.core.protocols.crosscutting import Evaluator

__all__ = [
    "GateResult",
    "GoldenCase",
    "JudgeCalibration",
    "MetricSummary",
    "Outcome",
    "Scorecard",
    "load_cases",
    "regression_gate",
    "sample_from_state",
    "sampled",
    "score_samples",
]

#: How a request ended, as the adversarial metric compares it.
Outcome = Literal["answered", "abstained", "blocked", "isolation_violation"]


class GoldenCase(BaseModel):
    """One dataset entry. The same shape serves the golden, regression and adversarial sets."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    query: str
    tenant_id: str = "tenant-local"
    acl_hashes: tuple[str, ...] = ()
    relevant_document_ids: tuple[str, ...] = ()
    expected_facts: tuple[str, ...] = ()
    expect_abstain: bool | None = None
    expected_outcome: Outcome | None = None
    #: Strings that must never appear in the answer: another tenant's content, a canary, the
    #: system prompt.
    forbidden_strings: tuple[str, ...] = ()
    aspect_documents: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    tags: tuple[str, ...] = ()


def load_cases(path: str | Path) -> list[GoldenCase]:
    """Load a JSONL dataset, one case per line. Blank lines and ``#`` comments are skipped."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [
        GoldenCase.model_validate_json(line)
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]


def sample_from_state(
    case: GoldenCase, *, outcome: Outcome, state: RequestState | None
) -> EvalSample:
    """Turn a finished request into a sample. ``state`` is ``None`` when the request raised."""
    result = state.result if state is not None else None
    metadata: dict[str, Any] = {
        "outcome": outcome,
        "abstained": outcome != "answered",
        "expected_facts": case.expected_facts,
        "forbidden_strings": case.forbidden_strings,
        "aspect_documents": case.aspect_documents,
    }
    if case.expect_abstain is not None:
        metadata["expect_abstain"] = case.expect_abstain
    if case.expected_outcome is not None:
        metadata["expected_outcome"] = case.expected_outcome
    if result is not None:
        metadata["claims_total"] = result.grounding.claims_total
        metadata["claims_cited"] = result.grounding.claims_cited

    return EvalSample(
        sample_id=case.case_id,
        query=case.query,
        relevant_document_ids=case.relevant_document_ids,
        retrieved_document_ids=(
            tuple(c.document_id for c in state.pool.candidates)
            if state is not None and state.pool is not None
            else ()
        ),
        answer=result.answer if result is not None else None,
        evidence_texts=(
            tuple(g.representative.context_text for g in state.evidence)
            if state is not None
            else ()
        ),
        citations=tuple(c.document_id for c in result.citations) if result is not None else (),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Scorecards
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MetricSummary:
    metric_id: str
    mean: float
    n: int
    #: ``None`` for metrics with no pass threshold.
    pass_rate: float | None
    #: Produced by an LLM judge. Judged metrics gate nothing until the judge is calibrated.
    judged: bool = False


@dataclass(frozen=True, slots=True)
class Scorecard:
    dataset_id: str
    summaries: dict[str, MetricSummary]
    results: tuple[MetricResult, ...] = ()

    def to_json(self) -> str:
        """Summaries only: the baseline a later run is diffed against."""
        return json.dumps(
            {
                "dataset_id": self.dataset_id,
                "summaries": {
                    k: {"mean": s.mean, "n": s.n, "pass_rate": s.pass_rate, "judged": s.judged}
                    for k, s in self.summaries.items()
                },
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, raw: str) -> Scorecard:
        data = json.loads(raw)
        return cls(
            dataset_id=data["dataset_id"],
            summaries={
                k: MetricSummary(metric_id=k, **v) for k, v in data["summaries"].items()
            },
        )

    def failures(self) -> list[MetricResult]:
        """Every sample that missed its metric's threshold, for the scorecard's detail."""
        return [r for r in self.results if r.passed is False]


async def score_samples(
    evaluators: Sequence[Evaluator], samples: Iterable[EvalSample], *, dataset_id: str
) -> Scorecard:
    results: list[MetricResult] = []
    for sample in samples:
        for evaluator in evaluators:
            results.append(await evaluator.score(sample))

    by_metric: dict[str, list[MetricResult]] = defaultdict(list)
    for r in results:
        if r.detail != NOT_APPLICABLE:
            by_metric[r.metric_id].append(r)

    summaries = {}
    for metric_id, scored in by_metric.items():
        gated = [r for r in scored if r.passed is not None]
        summaries[metric_id] = MetricSummary(
            metric_id=metric_id,
            mean=sum(r.score for r in scored) / len(scored),
            n=len(scored),
            pass_rate=sum(1 for r in gated if r.passed) / len(gated) if gated else None,
            judged=any(r.judge_model is not None for r in scored),
        )
    return Scorecard(dataset_id=dataset_id, summaries=summaries, results=tuple(results))


# ---------------------------------------------------------------------------
# The regression gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    failures: tuple[str, ...] = ()
    #: Metrics that would have been gated but were not, and why. Printed so an exclusion is a
    #: visible decision rather than a silent one.
    ungated: tuple[str, ...] = ()


def regression_gate(
    current: Scorecard,
    *,
    floors: Mapping[str, float],
    baseline: Scorecard | None = None,
    tolerance: float = 0.02,
    judge_calibrated: bool = False,
) -> GateResult:
    """Block on any floored metric below its floor, or regressed past tolerance.

    A floored metric with no scored samples fails. A gate over nothing passes nothing, and a
    dataset that silently stopped producing a metric is exactly the regression to catch.
    """
    failures: list[str] = []
    ungated: list[str] = []
    for metric_id, floor in floors.items():
        summary = current.summaries.get(metric_id)
        if summary is None:
            failures.append(f"{metric_id}: no samples scored")
            continue
        if summary.judged and not judge_calibrated:
            ungated.append(f"{metric_id}: judge not calibrated against human labels")
            continue
        if summary.mean < floor:
            failures.append(f"{metric_id}: {summary.mean:.3f} below floor {floor:.3f}")
        before = baseline.summaries.get(metric_id) if baseline else None
        if before is not None and summary.mean < before.mean - tolerance:
            failures.append(
                f"{metric_id}: {summary.mean:.3f} regressed from {before.mean:.3f} "
                f"(tolerance {tolerance:.3f})"
            )
    return GateResult(passed=not failures, failures=tuple(failures), ungated=tuple(ungated))


# ---------------------------------------------------------------------------
# Online sampling
# ---------------------------------------------------------------------------


def sampled(request_id: str, rate: float) -> bool:
    """Whether a request falls in the online evaluation sample.

    Derived from the request id rather than drawn at random, so every component that asks gets
    the same answer for the same request, and a replay samples exactly what the original did.
    """
    if rate <= 0.0:
        return False
    digest = hashlib.blake2b(request_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64 < rate


# ---------------------------------------------------------------------------
# Judge calibration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class JudgeCalibration:
    """Tracks an LLM judge's agreement with human labels.

    No judge score enters a gate until this says the judge is calibrated. An uncalibrated judge
    produces confident nonsense at scale, and a gate built on it blocks good changes and passes
    bad ones with equal conviction.
    """

    judge_model: str
    judge_version: str
    #: ``(judge score, human verdict)`` pairs.
    pairs: list[tuple[float, bool]] = field(default_factory=list)

    def record(self, judge_score: float, human_label: bool) -> None:
        self.pairs.append((judge_score, human_label))

    def agreement(self, threshold: float = 0.5) -> float | None:
        if not self.pairs:
            return None
        return sum(1 for s, h in self.pairs if (s >= threshold) == h) / len(self.pairs)

    def calibrated(self, *, min_pairs: int = 50, min_agreement: float = 0.8) -> bool:
        agreement = self.agreement()
        if agreement is None or len(self.pairs) < min_pairs:
            return False
        return agreement >= min_agreement
