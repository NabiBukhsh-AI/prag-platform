"""Fusion protocols: the knowledge decision policy and its calibrator.

These are two protocols rather than one because they are tuned on different cadences. The policy
table is adjusted by hand when the desired behaviour changes; the calibrator is refit whenever
the base model or adapter set changes. Merging them would couple a deliberate policy change to
an automated retraining job.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from prag.core.models.context import ContextBundle, ContextValidation
from prag.core.models.fusion import (
    KnowledgeDecision,
    ParametricSignal,
    RawConfidenceSignals,
)
from prag.core.models.identity import TenantPolicy
from prag.core.models.query import QueryAnalysis

__all__ = ["ConfidenceCalibrator", "FusionPolicy"]


@runtime_checkable
class FusionPolicy(Protocol):
    """Reconciles parametric belief with retrieved evidence."""

    async def decide(
        self,
        analysis: QueryAnalysis,
        parametric: ParametricSignal | None,
        context: ContextBundle | None,
        validation: ContextValidation | None,
        policy: TenantPolicy,
    ) -> KnowledgeDecision:
        """Decide what grounds the answer, and whether to answer at all.

        **Apply hard abstention gates before scoring.** They are gates, not score
        contributions, and no ``KnowledgeScore`` however high may override them. The gate that
        matters most: a query targeting private or tenant-specific knowledge, with no usable
        evidence, abstains — even when parametric confidence is high. A confident-sounding
        general answer to a question about the tenant's own data is the single most damaging
        output this system can produce, and it is damaging precisely because it looks correct.

        **Emit a conflict event for every detected parametric/retrieval contradiction.** Not
        only the ones that change the outcome. Evidence winning over a stale parametric prior is
        the normal, desired case — the corpus is the system of record — but each instance is
        also a data point saying an adapter has drifted, and the aggregate is what triggers
        retraining. Suppressing the ones that resolved cleanly would remove the signal exactly
        where it is most reliable.

        Both arguments are optional because both paths are optional. Parametric is ``None`` when
        the tier is disabled or nothing was selected; context is ``None`` on a parametric-only
        route. Both ``None`` is an abstention, not a crash.
        """
        ...


@runtime_checkable
class ConfidenceCalibrator(Protocol):
    """Maps raw confidence signals to a calibrated probability.

    Without this step the entire fusion policy is built on noise: raw token logprobs from an
    instruction-tuned model are badly calibrated, and thresholds tuned against them mean nothing.
    """

    calibrator_version: str

    def calibrate(self, raw: RawConfidenceSignals) -> float:
        """Return a calibrated probability in [0, 1].

        Synchronous and pure — an isotonic or small logistic fit, evaluated in microseconds.
        Anything requiring I/O here would put a network call inside the fusion decision.
        """
        ...

    def expected_calibration_error(self) -> float:
        """Current ECE against the held-out set this calibrator was fit on.

        Exposed on the protocol because it is a monitored metric with an alert, not a training
        detail. Calibration drifts silently, and every threshold in the fusion policy is
        expressed in terms of these numbers — so a drifting calibrator moves the system's
        behaviour without any configuration having changed.
        """
        ...
