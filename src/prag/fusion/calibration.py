"""Confidence calibration, and the calibration error that says whether to trust it.

Raw confidence signals — token logprobs, reranker scores, coverage similarities — are rankings,
not probabilities. Every threshold in the fusion policy is written as if they were
probabilities, so without calibration the policy is built on noise. An isotonic fit maps a raw
score to the observed rate of correct answers at that score, monotonically, with no assumption
about the shape of the relationship.

The expected calibration error is a monitored number, not a training detail: calibration drifts
silently when a model or adapter changes, and a drifting calibrator moves the system's behaviour
without any configuration having changed.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from prag.core.models.fusion import RawConfidenceSignals

__all__ = ["IsotonicCalibrator", "expected_calibration_error", "raw_parametric_score"]


def expected_calibration_error(
    probabilities: Sequence[float], outcomes: Sequence[bool], *, bins: int = 10
) -> float:
    """Mean gap between stated confidence and observed accuracy, weighted by bin size."""
    if not probabilities:
        return 0.0
    totals = [[0.0, 0.0, 0] for _ in range(bins)]  # confidence sum, correct count, n
    for p, correct in zip(probabilities, outcomes, strict=True):
        slot = min(bins - 1, int(p * bins))
        totals[slot][0] += p
        totals[slot][1] += 1.0 if correct else 0.0
        totals[slot][2] += 1
    n = len(probabilities)
    return sum(
        abs(conf / count - hits / count) * count / n for conf, hits, count in totals if count
    )


def raw_parametric_score(raw: RawConfidenceSignals) -> float:
    """One scalar from whichever parametric signals are present, before calibration."""
    signals = [
        math.exp(raw.mean_logprob) if raw.mean_logprob is not None else None,
        raw.self_consistency,
        raw.verbalized_confidence,
        raw.adapter_coverage,
    ]
    present = [min(1.0, max(0.0, s)) for s in signals if s is not None]
    return sum(present) / len(present) if present else 0.0


class IsotonicCalibrator:
    """Implements ``ConfidenceCalibrator`` with a pool-adjacent-violators fit."""

    def __init__(
        self,
        points: Sequence[tuple[float, float]],
        *,
        calibrator_version: str,
        ece: float,
    ) -> None:
        #: ``(raw score, calibrated probability)``, sorted and non-decreasing in probability.
        self._points = list(points)
        self.calibrator_version = calibrator_version
        self._ece = ece

    @classmethod
    def identity(cls) -> IsotonicCalibrator:
        """Passes scores through. Its calibration error is reported as 1.0 — unknown is worst."""
        return cls([(0.0, 0.0), (1.0, 1.0)], calibrator_version="identity.uncalibrated", ece=1.0)

    @classmethod
    def fit(
        cls,
        scores: Sequence[float],
        outcomes: Sequence[bool],
        *,
        calibrator_version: str,
        held_out: tuple[Sequence[float], Sequence[bool]] | None = None,
    ) -> IsotonicCalibrator:
        """Fit on ``scores``/``outcomes``; report ECE on ``held_out`` when given.

        Without a held-out set the reported error is measured on the fitting data, which flatters
        the fit. Callers that gate on it should always pass one.
        """
        if not scores:
            return cls.identity()
        # Each block: [score sum, outcome sum, count]. Merge while a block's mean outcome falls
        # below its predecessor's, which is the whole of pool-adjacent-violators.
        blocks: list[list[float]] = []
        for score, outcome in sorted(zip(scores, outcomes, strict=True)):
            blocks.append([score, 1.0 if outcome else 0.0, 1.0])
            while len(blocks) > 1 and blocks[-2][1] / blocks[-2][2] > blocks[-1][1] / blocks[-1][2]:
                last = blocks.pop()
                for i in range(3):
                    blocks[-1][i] += last[i]
        points = [(s / n, o / n) for s, o, n in blocks]
        fitted = cls(points, calibrator_version=calibrator_version, ece=0.0)
        check_scores, check_outcomes = held_out or (scores, outcomes)
        fitted._ece = expected_calibration_error(
            [fitted.calibrate_score(s) for s in check_scores], list(check_outcomes)
        )
        return fitted

    def calibrate(self, raw: RawConfidenceSignals) -> float:
        return self.calibrate_score(raw_parametric_score(raw))

    def calibrate_score(self, score: float) -> float:
        """Piecewise-linear between fitted points, flat beyond the ends."""
        points = self._points
        if score <= points[0][0]:
            return points[0][1]
        for (x0, y0), (x1, y1) in pairwise(points):
            if score <= x1:
                return y0 if x1 == x0 else y0 + (y1 - y0) * (score - x0) / (x1 - x0)
        return points[-1][1]

    def expected_calibration_error(self) -> float:
        return self._ece
