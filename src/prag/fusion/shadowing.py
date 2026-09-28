"""Provenance shadowing, architecture §11.5.

A parametric answer arrives with no citations, because weights cannot cite. Shadowing checks each
of its claims against retrieved evidence: a claim the evidence entails gets a citation, a claim
nothing entails is marked unsourced, and a claim the evidence contradicts means the parametric
answer is not served at all — the evidence wins.

Entailment is delegated to the platform's ``GroundingVerifier`` rather than reimplemented, so a
shadow citation and an ordinary citation are bound by the same rule and fail in the same ways.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from prag.core.ids import new_id, short_hash
from prag.core.models.context import ContextBundle
from prag.core.models.fusion import (
    ConflictEvent,
    ConflictKind,
    ConflictPosition,
    ConflictResolution,
    Stance,
)
from prag.core.models.generation import ShadowReport
from prag.fusion.conflicts import stance

if TYPE_CHECKING:
    from collections.abc import Sequence

    from prag.core.models.common import Deadline
    from prag.core.models.retrieval import EvidenceGroup
    from prag.core.protocols.generation import GroundingVerifier

__all__ = ["PARAMETRIC_AUTHORITY", "EntailmentProvenanceShadower"]

#: The fixed, deliberately low authority of parametric knowledge (§11.4, ``A_par``). Weights are
#: never the system of record; the corpus is.
PARAMETRIC_AUTHORITY = 0.55


class EntailmentProvenanceShadower:
    """Implements ``ProvenanceShadower``."""

    def __init__(self, verifier: GroundingVerifier) -> None:
        self._verifier = verifier

    async def shadow(
        self,
        answer: str,
        evidence: Sequence[EvidenceGroup],
        adapter_ids: Sequence[str],
        deadline: Deadline,
    ) -> ShadowReport:
        deadline.raise_if_expired()
        bundle = ContextBundle(
            bundle_id=new_id("shadow"),
            regions=(),
            evidence=tuple(evidence),
            rendered_prompt_hash=short_hash(answer),
        )
        grounding = await self._verifier.verify(answer, bundle)

        conflicts = []
        sentences = [
            (group, sentence)
            for group in evidence
            for sentence in _sentences(group.representative.context_text)
        ]
        for verdict in grounding.verdicts:
            stances = [(group, s, stance(verdict.claim, s)) for group, s in sentences]
            # Direct support anywhere clears the claim. A neighbouring sentence on the same topic
            # with a different number is often a different fact, not a contradiction.
            if any(found is Stance.SUPPORTS for _, _, found in stances):
                continue
            against = next(((g, s) for g, s, found in stances if found is Stance.CONTRADICTS), None)
            if against is None:
                continue
            group, sentence = against
            conflicts.append(
                ConflictEvent(
                    conflict_id=new_id("conflict"),
                    kind=ConflictKind.PARAMETRIC_VS_RETRIEVED,
                    claim=verdict.claim,
                    positions=(
                        ConflictPosition(
                            origin_id=",".join(adapter_ids) or "parametric",
                            stance=Stance.SUPPORTS,
                            authority=PARAMETRIC_AUTHORITY,
                        ),
                        ConflictPosition(
                            origin_id=group.representative.source_id,
                            stance=Stance.CONTRADICTS,
                            authority=group.authority,
                            as_of_ms=group.representative.metadata.updated_at_ms or None,
                            excerpt=sentence,
                        ),
                    ),
                    resolution=ConflictResolution.EVIDENCE_WINS,
                    adapter_ids=tuple(adapter_ids),
                )
            )
        return ShadowReport(grounding=grounding, conflicts=tuple(conflicts))


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", " ".join(text.split())) if s]
