"""The fusion decision policy, architecture §11.4.

Not a weighted sum. A sum lets a high value on one axis mask a disqualifying value on another, so
the knowledge score is computed and recorded, and then a table decides:

| Condition                                              | Action                                  |
|--------------------------------------------------------|-----------------------------------------|
| P_ret high, no conflict with parametric                | ground in evidence, cite                |
| P_ret high, conflicts with parametric                  | evidence wins, log the conflict         |
| P_ret moderate, parametric agrees                      | ground in evidence, confidence boosted  |
| P_ret low, P_par high, generic query                   | answer parametrically, marked as such   |
| P_ret low, P_par high, private query                   | abstain (hard gate)                     |
| sources conflict, comparable authority                 | surface the conflict                    |
| sources conflict, authority clearly separated          | prefer the higher authority, cite both  |
| both low                                               | abstain, or clarify if ambiguous        |
| evidence stale relative to the half-life               | warn, or abstain in strict mode         |

The two abstention rules are gates evaluated before anything else. No knowledge score overrides
them.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from prag.core.ids import new_id
from prag.core.models.fusion import (
    Abstention,
    AbstentionCode,
    ConflictEvent,
    ConflictKind,
    ConflictPosition,
    ConflictResolution,
    KnowledgeBasis,
    KnowledgeDecision,
    StalenessWarning,
    Stance,
)
from prag.core.models.query import KnowledgeRequirement
from prag.fusion.calibration import IsotonicCalibrator
from prag.fusion.conflicts import stance
from prag.fusion.shadowing import PARAMETRIC_AUTHORITY

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from prag.core.models.context import ContextBundle, ContextValidation
    from prag.core.models.fusion import ParametricSignal
    from prag.core.models.identity import TenantPolicy
    from prag.core.models.query import QueryAnalysis
    from prag.core.models.retrieval import EvidenceGroup

__all__ = [
    "DEGRADED_RETRIEVAL_CONFIDENCE",
    "TablePolicy",
    "independent_agreement",
    "raw_retrieval_score",
    "source_conflicts",
]

DEFAULT_WEIGHTS = {"w1": 0.40, "w2": 0.20, "w3": 0.20, "w4": 0.15, "w5": 0.05}
#: P_ret when the reranker was skipped by the degradation ladder. Rank-fusion scores are not
#: comparable to rerank scores, and reading a tiny fused score as "no confidence" would make every
#: degraded request abstain — degradation must shed quality, not the answer.
DEGRADED_RETRIEVAL_CONFIDENCE = 0.5
#: The least a source must score to count as corroborating another.
CORROBORATION_FLOOR = 0.4
_DAY_MS = 86_400_000

PARAMETRIC_MARKING = (
    "Answered from general knowledge rather than your documents; nothing retrieved confirms it."
)


def raw_retrieval_score(groups: Sequence[EvidenceGroup]) -> float:
    """P_ret before calibration: the best rerank score, plus corroboration from other lineages.

    ponytail: two of the four §11.2 signals (best score, independent corroboration); the score
    margin and aspect coverage join when context validation lands.
    """
    reranked = [g.representative.rerank_score for g in groups]
    scores = [s for s in reranked if s is not None]
    if not groups:
        return 0.0
    if not scores:
        return DEGRADED_RETRIEVAL_CONFIDENCE
    top = min(1.0, max(0.0, max(scores)))
    # Corroboration counts only sources that are themselves relevant. Relative to the top score
    # alone, weak evidence corroborates itself: when the best match is 0.35, everything above
    # 0.175 "agrees", and an off-topic retrieval is lifted over the answering threshold.
    floor = max(0.5 * top, CORROBORATION_FLOOR)
    corroborating = {
        g.lineage_root for g in groups if (g.representative.rerank_score or 0.0) >= floor
    }
    return 0.8 * top + 0.2 * min(1.0, len(corroborating) / 2)


def independent_agreement(
    groups: Sequence[EvidenceGroup], conflicts: Sequence[ConflictEvent] = ()
) -> float:
    """Authority-weighted share of independent sources not in conflict (§11.3).

    Groups sharing a lineage root count once, at the authority of their strongest member. Three
    documents quoting one press release are one piece of evidence, and counting them three times
    inflates confidence exactly when the system is most wrong.
    """
    by_root: dict[str, float] = {}
    for group in groups:
        by_root[group.lineage_root] = max(by_root.get(group.lineage_root, 0.0), group.authority)
    total = sum(by_root.values())
    if not total:
        return 0.0
    disputed = {
        p.origin_id
        for c in conflicts
        if c.kind is ConflictKind.SOURCE_VS_SOURCE
        for p in c.positions
        if p.stance is Stance.CONTRADICTS
    }
    supporting = sum(a for root, a in by_root.items() if root not in disputed)
    return supporting / total


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE.split(" ".join(text.split())) if len(s) > 12]


def source_conflicts(
    groups: Sequence[EvidenceGroup], *, surface_below_delta: float = 0.15
) -> list[ConflictEvent]:
    """Sentences from independent sources that state the same thing with different facts.

    Only across lineage roots: one document restating itself is not a disagreement. Resolution
    follows the table — comparable authority is surfaced, a clear gap prefers the stronger source
    and still cites both.
    """
    conflicts: list[ConflictEvent] = []
    seen: set[tuple[str, str]] = set()
    for i, left in enumerate(groups):
        for right in groups[i + 1 :]:
            if left.lineage_root == right.lineage_root:
                continue
            pair = tuple(sorted((left.lineage_root, right.lineage_root)))
            if pair in seen:
                continue
            for claim in _sentences(left.representative.context_text):
                against = next(
                    (
                        s
                        for s in _sentences(right.representative.context_text)
                        if stance(claim, s) is Stance.CONTRADICTS
                    ),
                    None,
                )
                if against is None:
                    continue
                seen.add(pair)
                delta = abs(left.authority - right.authority)
                resolution = (
                    ConflictResolution.SURFACED
                    if delta < surface_below_delta
                    else ConflictResolution.AUTHORITY_WINS
                )
                stronger, weaker = (
                    (left, right) if left.authority >= right.authority else (right, left)
                )
                conflicts.append(
                    ConflictEvent(
                        conflict_id=new_id("conflict"),
                        kind=ConflictKind.SOURCE_VS_SOURCE,
                        claim=claim,
                        positions=tuple(
                            ConflictPosition(
                                origin_id=g.lineage_root,
                                stance=Stance.SUPPORTS if g is stronger else Stance.CONTRADICTS,
                                authority=g.authority,
                                as_of_ms=g.representative.metadata.updated_at_ms or None,
                                excerpt=claim if g is left else against,
                            )
                            for g in (stronger, weaker)
                        ),
                        resolution=resolution,
                    )
                )
                break
    return conflicts


def _parametric_conflicts(
    signal: ParametricSignal, groups: Sequence[EvidenceGroup]
) -> list[ConflictEvent]:
    """Parametric claims the evidence contradicts and nothing supports. Evidence always wins."""
    if not signal.probe_answer:
        return []
    evidence = [(g, s) for g in groups for s in _sentences(g.representative.context_text)]
    adapters = tuple(f"{a.adapter_id}@{a.version}" for a in signal.adapters)
    conflicts = []
    for claim in _sentences(signal.probe_answer):
        stances = [(g, s, stance(claim, s)) for g, s in evidence]
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
                claim=claim,
                positions=(
                    ConflictPosition(
                        origin_id=",".join(adapters) or "parametric",
                        stance=Stance.SUPPORTS,
                        authority=PARAMETRIC_AUTHORITY,
                    ),
                    ConflictPosition(
                        origin_id=group.representative.source_id,
                        stance=Stance.CONTRADICTS,
                        authority=group.authority,
                        excerpt=sentence,
                    ),
                ),
                resolution=ConflictResolution.EVIDENCE_WINS,
                adapter_ids=adapters,
            )
        )
    return conflicts


class TablePolicy:
    """Implements ``FusionPolicy``."""

    def __init__(
        self,
        *,
        weights: Mapping[str, float] | None = None,
        parametric_authority: float = PARAMETRIC_AUTHORITY,
        high: float = 0.7,
        moderate: float = 0.4,
        agreement_boost: float = 0.1,
        surface_below_delta: float = 0.15,
        abstain_on_private_without_evidence: bool = True,
        retrieval_calibrator: IsotonicCalibrator | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._w = {**DEFAULT_WEIGHTS, **(weights or {})}
        self._a_par = parametric_authority
        self._high = high
        self._moderate = moderate
        self._boost = agreement_boost
        self._surface_below = surface_below_delta
        self._private_gate = abstain_on_private_without_evidence
        self._calibrator = retrieval_calibrator or IsotonicCalibrator.identity()
        self._clock = clock

    async def decide(
        self,
        analysis: QueryAnalysis,
        parametric: ParametricSignal | None,
        context: ContextBundle | None,
        validation: ContextValidation | None,
        policy: TenantPolicy,
    ) -> KnowledgeDecision:
        groups = tuple(context.evidence) if context is not None else ()
        p_ret = self._calibrator.calibrate_score(raw_retrieval_score(groups))
        p_par = parametric.calibrated_confidence if parametric is not None else 0.0

        conflicts = [
            *source_conflicts(groups, surface_below_delta=self._surface_below),
            *(_parametric_conflicts(parametric, groups) if parametric is not None else ()),
        ]
        agreement = independent_agreement(groups, conflicts)
        authority_max = max((g.authority for g in groups), default=0.0)
        f_ret = sum(g.freshness for g in groups) / len(groups) if groups else 0.0
        staleness = self._staleness(groups, analysis.temporality.estimated_half_life_days)
        unresolved = any(c.resolution is ConflictResolution.SURFACED for c in conflicts)
        penalty = 1.0 if unresolved else (0.5 if conflicts else 0.0)
        coverage_gap = 1.0 - validation.coverage if validation is not None else 0.0

        w = self._w
        score = (
            w["w1"] * p_ret * authority_max * f_ret
            + w["w2"] * p_par * self._a_par
            + w["w3"] * agreement
            - w["w4"] * penalty
            - w["w5"] * coverage_gap
        )

        def decision(basis: KnowledgeBasis, **extra: object) -> KnowledgeDecision:
            return KnowledgeDecision(
                basis=basis,
                knowledge_score=round(score, 4),
                p_parametric=p_par,
                p_retrieval=p_ret,
                agreement_independent=agreement,
                authority_max=authority_max,
                conflicts=tuple(conflicts),
                staleness=staleness,
                **extra,
            )

        def abstain(code: AbstentionCode, explanation: str, action: str) -> KnowledgeDecision:
            return decision(
                KnowledgeBasis.ABSTAIN,
                abstention=Abstention(
                    reason_code=code, explanation=explanation, suggested_action=action
                ),
            )

        private = analysis.requires(KnowledgeRequirement.REQUIRES_PRIVATE_DATA)
        retrieval_low = p_ret < self._moderate

        # Hard gate 1: private query, nothing usable retrieved. However confident the weights.
        if private and retrieval_low and self._private_gate:
            return abstain(
                AbstentionCode.PRIVATE_QUERY_NO_EVIDENCE,
                "The question is about your own data, and no document supports an answer.",
                "Check that the relevant source is indexed and that you have access to it.",
            )
        # Hard gate 2: both low.
        if retrieval_low and p_par < self._high:
            if analysis.ambiguity.is_ambiguous:
                return abstain(
                    AbstentionCode.AMBIGUOUS_NEEDS_CLARIFICATION,
                    "The question could mean several things and nothing answers it confidently.",
                    "Clarify: " + "; ".join(analysis.ambiguity.clarification_candidates[:3]),
                )
            return abstain(
                AbstentionCode.KNOWLEDGE_BELOW_FLOOR,
                "Nothing retrieved or learned answers this with enough confidence.",
                "Rephrase or broaden the question, or confirm the source has been indexed.",
            )
        if staleness is not None and staleness.staleness_ratio > 1.0 and policy.strict_mode:
            return abstain(
                AbstentionCode.EVIDENCE_TOO_STALE,
                "The only evidence is older than this kind of information stays accurate.",
                "Refresh the source, or ask with strict mode off to see the dated answer.",
            )

        # P_ret low, P_par high, generic: the weights answer, and say so.
        if retrieval_low:
            return decision(KnowledgeBasis.PARAMETRIC, epistemic_marking=PARAMETRIC_MARKING)

        # Evidence answers. A parametric conflict means evidence wins (already recorded); a
        # parametric agreement at moderate retrieval confidence boosts the score.
        parametric_agrees = (
            parametric is not None
            and p_par >= self._moderate
            and not any(c.kind is ConflictKind.PARAMETRIC_VS_RETRIEVED for c in conflicts)
        )
        if parametric_agrees and p_ret < self._high:
            score += self._boost
        return decision(KnowledgeBasis.RETRIEVED_EVIDENCE)

    def _staleness(
        self, groups: Sequence[EvidenceGroup], half_life_days: float
    ) -> StalenessWarning | None:
        """Age of the oldest dated evidence against the half-life. Undated evidence is skipped."""
        dated = [g.representative.metadata.updated_at_ms for g in groups]
        dated = [d for d in dated if d]
        if not dated or half_life_days <= 0:
            return None
        oldest = min(dated)
        ratio = (self._clock() * 1000 - oldest) / (half_life_days * _DAY_MS)
        if ratio <= 1.0:
            return None
        return StalenessWarning(
            oldest_evidence_ms=oldest, half_life_days=half_life_days, staleness_ratio=ratio
        )
