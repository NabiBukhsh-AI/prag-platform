"""Fusion: reconciling what the weights believe with what the sources say.

Provenance shadowing and the conflict loop from serving back to training (Phase 4); the decision
policy, confidence calibration, the independence correction and source-versus-source conflicts
(Phase 5).

Imports only ``core``. Entailment comes in through the ``GroundingVerifier`` protocol, so a
shadow citation obeys exactly the rule an ordinary citation does.
"""

from prag.fusion.calibration import (
    IsotonicCalibrator,
    expected_calibration_error,
    raw_parametric_score,
)
from prag.fusion.conflicts import ConflictMonitor, contradicts, stance
from prag.fusion.policy import (
    DEGRADED_RETRIEVAL_CONFIDENCE,
    TablePolicy,
    independent_agreement,
    raw_retrieval_score,
    source_conflicts,
)
from prag.fusion.shadowing import PARAMETRIC_AUTHORITY, EntailmentProvenanceShadower

__all__ = [
    "DEGRADED_RETRIEVAL_CONFIDENCE",
    "PARAMETRIC_AUTHORITY",
    "ConflictMonitor",
    "EntailmentProvenanceShadower",
    "IsotonicCalibrator",
    "TablePolicy",
    "contradicts",
    "expected_calibration_error",
    "independent_agreement",
    "raw_parametric_score",
    "raw_retrieval_score",
    "source_conflicts",
    "stance",
]
