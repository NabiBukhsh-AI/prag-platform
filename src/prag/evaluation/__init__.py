"""Evaluation: one metric implementation, three triggers.

The offline runner, the CI regression gate and the online sampler all score with
``standard_metrics``. This package imports only ``core``; producing samples from a live platform
is the composition root's job (``prag.api.evaluation``).
"""

from prag.evaluation.metrics import (
    NOT_APPLICABLE,
    FunctionMetric,
    coverage_at_k,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    standard_metrics,
)
from prag.evaluation.runner import (
    GateResult,
    GoldenCase,
    JudgeCalibration,
    MetricSummary,
    Outcome,
    Scorecard,
    load_cases,
    regression_gate,
    sample_from_state,
    sampled,
    score_samples,
)

__all__ = [
    "NOT_APPLICABLE",
    "FunctionMetric",
    "GateResult",
    "GoldenCase",
    "JudgeCalibration",
    "MetricSummary",
    "Outcome",
    "Scorecard",
    "coverage_at_k",
    "hit_rate_at_k",
    "load_cases",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "regression_gate",
    "sample_from_state",
    "sampled",
    "score_samples",
    "standard_metrics",
]
