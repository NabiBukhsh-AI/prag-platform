"""Every metric, implemented once.

The offline runner, the CI regression gate and the online sampler all call these. Divergence
between offline and online numbers must come from the data, never from two implementations of
NDCG — once they differ, the offline suite stops predicting production and there is no longer a
reason to run it.

Each metric is a pure function over an ``EvalSample``, wrapped as an ``Evaluator``. A function
returns ``None`` when the sample cannot be scored — no relevance labels, no expected facts — so a
missing label is excluded from the mean rather than counted as a zero.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from prag.core.models.events import EvalInputs, MetricResult

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from prag.core.models.events import EvalSample

__all__ = [
    "NOT_APPLICABLE",
    "FunctionMetric",
    "coverage_at_k",
    "hit_rate_at_k",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "standard_metrics",
]

#: ``MetricResult.detail`` for a sample the metric could not score. Excluded from aggregates.
NOT_APPLICABLE = "not_applicable"


def _ranked(retrieved: Sequence[str]) -> list[str]:
    """Document ids in first-seen order: five chunks of one document are one document."""
    return list(dict.fromkeys(retrieved))


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    top, wanted = set(_ranked(retrieved)[:k]), set(relevant)
    return len(top & wanted) / len(wanted) if wanted else 0.0


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Over what was actually returned, up to ``k``.

    Not divided by ``k`` itself: against a small corpus that can return fewer than ``k``
    documents, that would cap precision below 1.0 for a perfect result.
    """
    top = _ranked(retrieved)[:k]
    return sum(1 for d in top if d in set(relevant)) / len(top) if top else 0.0


def hit_rate_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    return 1.0 if set(_ranked(retrieved)[:k]) & set(relevant) else 0.0


def mrr(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    wanted = set(relevant)
    for rank, doc in enumerate(_ranked(retrieved), start=1):
        if doc in wanted:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Binary-relevance NDCG. Graded judgements slot in by replacing the gain."""
    wanted = set(relevant)
    dcg = sum(
        1.0 / math.log2(i + 2) for i, doc in enumerate(_ranked(retrieved)[:k]) if doc in wanted
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(wanted), k)))
    return dcg / ideal if ideal else 0.0


def coverage_at_k(
    retrieved: Sequence[str], aspect_documents: dict[str, Sequence[str]], k: int
) -> float:
    """The fraction of query aspects with at least one supporting document in the top ``k``.

    Predicts answer completeness on multi-aspect queries far better than recall does: recall
    rewards finding three documents about one aspect as much as one document about each of three.
    """
    if not aspect_documents:
        return 0.0
    top = set(_ranked(retrieved)[:k])
    covered = sum(1 for docs in aspect_documents.values() if top & set(docs))
    return covered / len(aspect_documents)


# ---------------------------------------------------------------------------
# Evaluator wrapper
# ---------------------------------------------------------------------------


class FunctionMetric:
    """An ``Evaluator`` over a pure scoring function.

    One class for every metric rather than one per metric: they differ only in the function,
    what it needs, and the pass threshold.
    """

    def __init__(
        self,
        metric_id: str,
        fn: Callable[[EvalSample], float | None],
        *,
        requires: EvalInputs,
        threshold: float | None = None,
    ) -> None:
        self.metric_id = metric_id
        self.requires = requires
        self._fn = fn
        self._threshold = threshold

    async def score(self, sample: EvalSample) -> MetricResult:
        value = self._fn(sample)
        if value is None:
            return MetricResult(
                metric_id=self.metric_id,
                sample_id=sample.sample_id,
                score=0.0,
                detail=NOT_APPLICABLE,
            )
        return MetricResult(
            metric_id=self.metric_id,
            sample_id=sample.sample_id,
            score=value,
            passed=None if self._threshold is None else value >= self._threshold,
        )


def _labelled(fn: Callable[[EvalSample], float]) -> Callable[[EvalSample], float | None]:
    """Score only labelled samples. An abstention still counts: retrieving nothing is a miss."""
    return lambda s: fn(s) if s.relevant_document_ids else None


def _groundedness(s: EvalSample) -> float | None:
    total = s.metadata.get("claims_total")
    if s.metadata.get("abstained") or not total:
        return None
    return float(s.metadata.get("claims_cited", 0)) / float(total)


def _citation_precision(s: EvalSample) -> float | None:
    """Citations that point at a document labelled relevant, over all citations.

    A proxy until claim-level citation labels exist: it cannot tell a relevant document cited
    for the wrong claim from the right one, but it does catch citing the wrong document.
    """
    if s.metadata.get("abstained") or not s.relevant_document_ids:
        return None
    if not s.citations:
        return 0.0
    return sum(1 for c in s.citations if c in s.relevant_document_ids) / len(s.citations)


def _completeness(s: EvalSample) -> float | None:
    facts = s.metadata.get("expected_facts") or ()
    if not facts:
        return None
    answer = (s.answer or "").casefold()
    return sum(1 for f in facts if f.casefold() in answer) / len(facts)


def _abstention_correct(s: EvalSample) -> float | None:
    if "expect_abstain" not in s.metadata:
        return None
    return 1.0 if bool(s.metadata.get("abstained")) == s.metadata["expect_abstain"] else 0.0


def _adversarial(s: EvalSample) -> float | None:
    """The expected outcome happened, if one is set, and nothing forbidden reached the answer.

    A probe may set only forbidden strings: an ACL probe passes whether the system abstains or
    answers from other documents, as long as the restricted text never appears.
    """
    expected = s.metadata.get("expected_outcome")
    forbidden = s.metadata.get("forbidden_strings") or ()
    if expected is None and not forbidden:
        return None
    answer = s.answer or ""
    leaked = any(f in answer for f in forbidden)
    outcome_ok = expected is None or s.metadata.get("outcome") == expected
    return 1.0 if outcome_ok and not leaked else 0.0


def _coverage(k: int) -> Callable[[EvalSample], float | None]:
    return lambda s: (
        coverage_at_k(s.retrieved_document_ids, s.metadata["aspect_documents"], k)
        if s.metadata.get("aspect_documents")
        else None
    )


def standard_metrics(k: int = 5) -> list[FunctionMetric]:
    """The standard metric set, in scorecard order."""
    labels = EvalInputs(needs_answer=False, needs_retrieval_labels=True)
    answer = EvalInputs(needs_answer=True)

    def retrieval(
        name: str, fn: Callable[[Sequence[str], Sequence[str]], float]
    ) -> FunctionMetric:
        return FunctionMetric(
            name,
            _labelled(lambda s: fn(s.retrieved_document_ids, s.relevant_document_ids)),
            requires=labels,
        )

    return [
        retrieval(f"recall@{k}", lambda r, g: recall_at_k(r, g, k)),
        retrieval(f"precision@{k}", lambda r, g: precision_at_k(r, g, k)),
        retrieval(f"hit_rate@{k}", lambda r, g: hit_rate_at_k(r, g, k)),
        retrieval("mrr", mrr),
        retrieval(f"ndcg@{k}", lambda r, g: ndcg_at_k(r, g, k)),
        FunctionMetric(f"coverage@{k}", _coverage(k), requires=labels),
        FunctionMetric("groundedness", _groundedness, requires=answer, threshold=0.9),
        FunctionMetric(
            "citation_precision",
            _citation_precision,
            requires=EvalInputs(needs_answer=True, needs_retrieval_labels=True),
            threshold=0.95,
        ),
        FunctionMetric("completeness", _completeness, requires=answer, threshold=1.0),
        FunctionMetric(
            "abstention_correct", _abstention_correct, requires=answer, threshold=1.0
        ),
        FunctionMetric("adversarial", _adversarial, requires=answer, threshold=1.0),
    ]
