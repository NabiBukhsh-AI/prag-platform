"""Fusion: reconciling what the weights believe with what the sources say.

Phase 4 brings the two pieces the parametric route cannot ship without: provenance shadowing,
and the conflict loop from serving back to training. The full decision policy, calibration and
source-versus-source conflicts arrive with Phase 5.

Imports only ``core``. Entailment comes in through the ``GroundingVerifier`` protocol, so a
shadow citation obeys exactly the rule an ordinary citation does.
"""

from prag.fusion.conflicts import ConflictMonitor, contradicts, stance
from prag.fusion.shadowing import PARAMETRIC_AUTHORITY, EntailmentProvenanceShadower

__all__ = [
    "PARAMETRIC_AUTHORITY",
    "ConflictMonitor",
    "EntailmentProvenanceShadower",
    "contradicts",
    "stance",
]
