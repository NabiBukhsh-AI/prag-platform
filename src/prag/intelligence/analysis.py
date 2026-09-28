"""Producing the structured query representation.

A cascade: deterministic rules first, then a trained encoder, then an LLM as the low-confidence
fallback only. Phase 1 shipped a defaulted analysis; this is the first tier that actually
decides anything, and the encoder tier arrives once Phase 2 has collected labels to train it on.

The whole cascade exists because of one budget: **a 300 ms LLM call cannot sit on the critical
path of a 650 ms time-to-first-token target.** Routing has to be decided in tens of milliseconds
or it is not routing, it is a second generation.

``router_uncertainty`` is the field that matters most downstream. It is not decoration — above
the configured threshold the orchestrator hedges, running retrieval even on a parametric lean so
the more expensive path is available if fusion turns out to need it. An analyzer that reported
false confidence would suppress exactly the hedging that uncertainty is meant to trigger.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from prag.core.models.common import VolatilityClass
from prag.core.models.query import (
    Ambiguity,
    BudgetClass,
    Entity,
    FieldPrediction,
    KnowledgeRequirement,
    QueryAnalysis,
    QueryQuality,
    QueryStructure,
    SafetyPreflags,
    SourceHint,
    Temporality,
)
from prag.intelligence.classifiers.rules import classify_rules

if TYPE_CHECKING:
    from prag.core.models.common import Deadline
    from prag.core.models.identity import Principal
    from prag.core.models.query import SessionContext

__all__ = ["CascadeQueryAnalyzer"]

#: Half-life by volatility class, in days. Freshness scoring divides by this, so absolute
#: document age is meaningless without it: a two-year-old constitutional provision is fresh, a
#: two-day-old exchange rate is not.
_HALF_LIFE_DAYS: dict[VolatilityClass, float] = {
    VolatilityClass.STATIC: 3_650.0,
    VolatilityClass.SLOW: 180.0,
    VolatilityClass.FAST: 7.0,
    VolatilityClass.REALTIME: 0.04,
}

#: Confidence assigned to a value a rule settled. High but not certain: rules are precise about
#: what they match and blind to what they do not, and a pattern can be triggered by a sentence
#: that happens to contain the words.
_RULE_CONFIDENCE = 0.88
#: Confidence assigned to a defaulted value. Deliberately at the coin-flip line, because that is
#: what a default is, and rounding it up would launder a guess into a finding.
_DEFAULT_CONFIDENCE = 0.5

_ENTITY = re.compile(r"\b(?:[A-Z][\w-]+(?:\s+[A-Z][\w-]+)*|[a-z]+-\d+|[A-Z]{2,})\b")
_PRONOUN = re.compile(r"\b(it|its|that|this|they|them|those|there)\b", re.IGNORECASE)
_QUESTION_WORD = re.compile(r"^\s*(who|what|when|where|why|how|which|is|are|do|does|can)\b", re.I)


class CascadeQueryAnalyzer:
    """Rules first, model tiers after — and only when the rules left work undone."""

    def __init__(
        self,
        *,
        escalate_below_coverage: float = 0.6,
        default_domain: str = "general",
    ) -> None:
        self._escalate_below = escalate_below_coverage
        self._default_domain = default_domain
        self.escalations = 0

    async def analyze(
        self,
        query: str,
        session: SessionContext | None,
        principal: Principal,
        deadline: Deadline,
    ) -> QueryAnalysis:
        started = time.monotonic()
        normalized = " ".join(query.split())
        verdict = classify_rules(normalized)

        # T1 (encoder) and T2 (LLM fallback) land here once there are labels to train on and a
        # budget to spend. The tier is reported honestly in the meantime: a trace that claimed
        # T1 ran would attribute a default's decision to a model that does not exist.
        tier = "T0"
        if verdict.coverage < self._escalate_below:
            # The rules left real work undone. Counting it here means the escalation rate is
            # observable before the tiers that would serve it exist, which is what tells anyone
            # whether building them is worth it.
            self.escalations += 1

        requirements = self._requirements(verdict, normalized)
        volatility = verdict.volatility or VolatilityClass.SLOW
        multi_hop = verdict.multi_hop if verdict.multi_hop is not None else False

        return QueryAnalysis(
            request_id=principal.user_id + ":" + str(int(started * 1000)),
            raw_query=query,
            normalized_query=normalized,
            language="en",
            intent=self._prediction(verdict.intent, "lookup"),
            domain=self._prediction(None, self._default_domain),
            complexity=self._prediction(verdict.complexity, "simple_factual"),
            knowledge_requirements=requirements,
            temporality=Temporality(
                volatility_class=volatility,
                estimated_half_life_days=_HALF_LIFE_DAYS[volatility],
                confidence=_RULE_CONFIDENCE if verdict.volatility else _DEFAULT_CONFIDENCE,
            ),
            structure=QueryStructure(
                multi_hop=FieldPrediction(
                    value=multi_hop,
                    confidence=_RULE_CONFIDENCE
                    if verdict.multi_hop is not None
                    else _DEFAULT_CONFIDENCE,
                ),
                entities=self._entities(query),
            ),
            ambiguity=self._ambiguity(normalized, session),
            source_hints=(),
            query_quality=self._quality(normalized, session),
            budget_class=BudgetClass(latency_tier=principal.sla_tier, cost_tier=principal.sla_tier),
            safety_preflags=SafetyPreflags(),
            router_uncertainty=self._uncertainty(verdict),
            classifier_tier_used=tier,
            analysis_latency_ms=max(0, int((time.monotonic() - started) * 1000)),
        )

    def _requirements(
        self, verdict: object, normalized: str
    ) -> dict[KnowledgeRequirement, FieldPrediction]:
        """Turn rule verdicts into per-requirement predictions.

        Each carries its own confidence rather than inheriting one from the analysis, because
        the hard constraints in routing consult them individually. A single global confidence
        would let a well-determined requirement be discounted by an unrelated uncertainty.
        """
        pairs: list[tuple[KnowledgeRequirement, bool | None, bool]] = [
            (
                KnowledgeRequirement.REQUIRES_EXTERNAL_KNOWLEDGE,
                verdict.requires_external_knowledge,  # type: ignore[attr-defined]
                True,
            ),
            (KnowledgeRequirement.REQUIRES_LIVE_DATA, verdict.requires_live_data, False),  # type: ignore[attr-defined]
            (KnowledgeRequirement.REQUIRES_PRIVATE_DATA, verdict.requires_private_data, False),  # type: ignore[attr-defined]
            (
                KnowledgeRequirement.REQUIRES_EXACT_QUOTATION,
                verdict.requires_exact_quotation,  # type: ignore[attr-defined]
                False,
            ),
            (KnowledgeRequirement.REQUIRES_CITATION, verdict.requires_citation, True),  # type: ignore[attr-defined]
        ]

        return {
            requirement: FieldPrediction(
                value=default if settled is None else settled,
                confidence=_DEFAULT_CONFIDENCE if settled is None else _RULE_CONFIDENCE,
            )
            for requirement, settled, default in pairs
        }

    @staticmethod
    def _prediction(settled: str | None, default: str) -> FieldPrediction:
        return FieldPrediction(
            value=settled or default,
            confidence=_RULE_CONFIDENCE if settled else _DEFAULT_CONFIDENCE,
        )

    @staticmethod
    def _entities(query: str) -> tuple[Entity, ...]:
        """Capitalised runs and identifier-shaped tokens.

        Crude, and useful anyway: identifiers like ``sev-1`` are exactly where dense retrieval
        fails and a lexical leg earns its place, so spotting them is what makes hybrid routing
        worth doing. A trained NER model replaces this without changing a caller.
        """
        seen: dict[str, Entity] = {}
        for match in _ENTITY.finditer(query):
            text = match.group(0)
            if len(text) < 2 or text.lower() in {"the", "a", "an", "is", "are"}:
                continue
            seen.setdefault(
                text.lower(),
                Entity(
                    text=text,
                    type="identifier" if re.search(r"\d", text) else "proper_noun",
                    canonical_id=text.lower().replace(" ", "."),
                ),
            )
        return tuple(seen.values())

    @staticmethod
    def _ambiguity(normalized: str, session: SessionContext | None) -> Ambiguity:
        """Detect a query that cannot be answered well as written.

        Unresolved pronouns on a follow-up turn are the common case, and the one worth catching:
        "does it apply to them" retrieves nothing useful, and the clarification loop is one of
        the three cycles the graph exists to support.
        """
        has_pronoun = bool(_PRONOUN.search(normalized))
        is_follow_up = session is not None and session.turn_index > 0

        if has_pronoun and not is_follow_up:
            return Ambiguity(
                is_ambiguous=True,
                ambiguity_type="unresolved_reference",
                clarification_candidates=("What does this refer to?",),
            )
        if len(normalized.split()) <= 2 and not _QUESTION_WORD.match(normalized):
            return Ambiguity(
                is_ambiguous=True,
                ambiguity_type="underspecified",
                clarification_candidates=("Could you say more about what you need?",),
            )
        return Ambiguity()

    @staticmethod
    def _quality(normalized: str, session: SessionContext | None) -> QueryQuality:
        """Whether the query is good enough to retrieve against as written."""
        words = len(normalized.split())
        has_pronoun = bool(_PRONOUN.search(normalized))
        is_follow_up = session is not None and session.turn_index > 0

        needs_rewrite = words < 4 or (has_pronoun and is_follow_up)
        score = 0.9 if words >= 6 and not has_pronoun else 0.55 if words >= 4 else 0.3
        return QueryQuality(needs_rewrite=needs_rewrite, score=score)

    def _uncertainty(self, verdict: object) -> float:
        """Aggregate uncertainty across the heads that feed routing.

        Derived from how much the rules actually settled. Reporting low uncertainty on a
        defaulted analysis would suppress the hedged execution that uncertainty exists to
        trigger, which is the specific way a confident router does the most damage.
        """
        return round(1.0 - verdict.coverage, 3)  # type: ignore[attr-defined]


def source_hints_for(
    analysis: QueryAnalysis, available_source_ids: tuple[str, ...]
) -> tuple[SourceHint, ...]:
    """Prior belief that each source is worth querying.

    Priors, not decisions. The planner combines them with source health, budget and
    capabilities, so a high prior on an unavailable source produces no leg rather than a
    failure. Phase 2 spreads them evenly; the empirical table replaces this once there are
    outcomes to learn from.
    """
    if not available_source_ids:
        return ()
    even = 1.0 / len(available_source_ids)
    return tuple(SourceHint(source_id=source_id, prior=even) for source_id in available_source_ids)
