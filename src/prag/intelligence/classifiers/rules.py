"""The T0 rules classifier.

Deterministic pattern rules over the query, running in well under a millisecond. It is the first
tier of a cascade, and it exists because a large share of real traffic is decidable without a
model at all: "what is 17% of 340" needs no retrieval, "summarise the text I just pasted" needs
no external knowledge, and "what did we decide" is a follow-up whose answer is in the session.

Getting those out of the way cheaply is not an optimisation — it is what keeps the expensive
tiers available for the queries that need them, and it is why the fallback tier can be capped at
a small share of traffic without that cap ever binding on normal load.

**Rules only fire when they are confident.** A rule that half-matches returns nothing and lets
the next tier decide. The cascade's value depends on T0 being right when it speaks, not on it
speaking often: a wrong cheap answer costs more than an expensive correct one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from prag.core.models.common import VolatilityClass

__all__ = ["RuleVerdict", "classify_rules"]

_ARITHMETIC: Final = re.compile(
    # Two numbers joined by an operator, in any position. Anchoring at the start of the
    # string would miss the canonical phrasing: real arithmetic queries open with a question
    # word rather than a digit.
    r"\d[\d,.]*\s*(?:[-+*/^]|%\s+of|%\s*of|percent\s+of|divided\s+by|times)\s*\d|\b(?:calculate|compute|work\s+out)\b"
    # An explicit instruction to compute, recognisable without operands.
    r"|\b(?:calculate|compute|work\s+out)\b",
    re.IGNORECASE,
)

_TRANSFORM_VERB: Final = re.compile(
    r"\b(summari[sz]e|rewrite|rephrase|translate|reformat|convert|shorten|expand)\b",
    re.IGNORECASE,
)
#: Phrases that point at content already in the conversation rather than at a corpus.
_SELF_REFERENTIAL: Final = re.compile(
    r"\b(the (?:above|following|text|code|snippet) (?:i|I)? ?(?:just )?(?:pasted|sent|gave)?"
    r"|this (?:text|code|snippet|paragraph|message)"
    r"|what (?:did )?(?:we|you|i) (?:just )?(?:say|decide|discuss|agree))\b",
    re.IGNORECASE,
)
_LIVE_DATA: Final = re.compile(
    r"\b(right now|currently|at the moment|today'?s?|latest|live|real[- ]?time"
    r"|current (?:price|status|value|rate|balance)"
    r"|as of (?:today|now|this morning))\b",
    re.IGNORECASE,
)
_EXACT_QUOTATION: Final = re.compile(
    r"\b(verbatim|word[- ]for[- ]word|exact (?:wording|text|quote|language)"
    r"|quote (?:the|me)|exactly what (?:it|the \w+) says)\b",
    re.IGNORECASE,
)
#: Possessives and scoping words that mark a query as being about the tenant's own material.
_PRIVATE_SCOPE: Final = re.compile(
    r"\b(our|my|we|us|the company'?s?|internal|in[- ]house|this (?:team|org|tenant))\b",
    re.IGNORECASE,
)
_CITATION_DEMAND: Final = re.compile(
    r"\b(cite|citation|source|sources|according to|where does it say|which document)\b",
    re.IGNORECASE,
)
_MULTI_HOP: Final = re.compile(
    r"\b(compare|versus|vs\.?|difference between|and then|both .+ and |"
    r"how does .+ (?:affect|relate|compare))\b",
    re.IGNORECASE,
)
_GREETING: Final = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|ok|okay|got it|never ?mind)\b[\s!.?]*$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class RuleVerdict:
    """What the rules tier could determine, and how sure it is.

    Every field is optional. A verdict that determines two things out of ten is useful — those
    two skip a model call — and pretending to more than the rules established would put a
    confident wrong value where an honest gap belongs.
    """

    intent: str | None = None
    complexity: str | None = None
    requires_external_knowledge: bool | None = None
    requires_live_data: bool | None = None
    requires_private_data: bool | None = None
    requires_exact_quotation: bool | None = None
    requires_citation: bool | None = None
    multi_hop: bool | None = None
    volatility: VolatilityClass | None = None
    #: How much of the analysis the rules settled, in [0, 1]. Drives escalation: a low value
    #: means the next tier has real work to do rather than a formality to confirm.
    coverage: float = 0.0
    #: Which rules fired. Carried into the trace, because "T0 decided" is not an explanation and
    #: a routing decision nobody can account for is one nobody can fix.
    fired: tuple[str, ...] = ()

    @property
    def is_decisive(self) -> bool:
        """Whether the rules settled enough to skip the model tiers.

        The threshold is deliberately high. The cascade's value comes from T0 being right when
        it speaks, not from it speaking often — a wrong cheap answer costs more than an
        expensive correct one.
        """
        return self.coverage >= 0.6


def classify_rules(query: str) -> RuleVerdict:
    """Apply the deterministic rules. No model call, no I/O.

    Order matters only where two rules could disagree, and the ones that could are checked
    against the more specific pattern first.
    """
    fired: list[str] = []
    settled: dict[str, object] = {}

    if _GREETING.match(query):
        # A greeting or acknowledgement. Retrieving for it is pure waste, and answering it from
        # a corpus produces something absurd.
        return RuleVerdict(
            intent="conversational",
            complexity="trivial",
            requires_external_knowledge=False,
            requires_live_data=False,
            requires_private_data=False,
            requires_citation=False,
            multi_hop=False,
            coverage=1.0,
            fired=("greeting",),
        )

    if _ARITHMETIC.search(query):
        fired.append("arithmetic")
        settled |= {
            "intent": "computation",
            "complexity": "simple_factual",
            "requires_external_knowledge": False,
            "requires_citation": False,
        }

    if _SELF_REFERENTIAL.search(query):
        # The content is already in the conversation. Retrieval would return documents about a
        # topic the user is not asking about, and the model would then have to ignore them.
        fired.append("self_referential")
        settled |= {"requires_external_knowledge": False, "requires_citation": False}

    if _TRANSFORM_VERB.search(query) and _SELF_REFERENTIAL.search(query):
        fired.append("transformation")
        settled |= {"intent": "transformation", "complexity": "simple_factual"}

    if _LIVE_DATA.search(query):
        # Live data cannot be parametric at any confidence: weights are a snapshot, and a
        # snapshot answering "what is it right now" is wrong in a way that reads as right.
        fired.append("live_data")
        settled |= {
            "requires_live_data": True,
            "requires_external_knowledge": True,
            "volatility": VolatilityClass.REALTIME,
        }

    if _EXACT_QUOTATION.search(query):
        # Weights reproduce meaning, not verbatim spans, reliably. A request for exact wording
        # is one the parametric route cannot serve honestly.
        fired.append("exact_quotation")
        settled |= {"requires_exact_quotation": True, "requires_citation": True}

    if _PRIVATE_SCOPE.search(query):
        fired.append("private_scope")
        settled |= {"requires_private_data": True, "requires_external_knowledge": True}

    if _CITATION_DEMAND.search(query):
        fired.append("citation_demand")
        settled |= {"requires_citation": True}

    if _MULTI_HOP.search(query):
        fired.append("multi_hop")
        settled |= {"multi_hop": True, "complexity": "multi_step"}

    # Coverage counts the fields the rules actually settled, out of the eight that feed routing.
    # It is a measure of how much work remains, not a confidence score about what was decided.
    coverage = len(settled) / 8.0

    return RuleVerdict(
        intent=settled.get("intent"),  # type: ignore[arg-type]
        complexity=settled.get("complexity"),  # type: ignore[arg-type]
        requires_external_knowledge=settled.get("requires_external_knowledge"),  # type: ignore[arg-type]
        requires_live_data=settled.get("requires_live_data"),  # type: ignore[arg-type]
        requires_private_data=settled.get("requires_private_data"),  # type: ignore[arg-type]
        requires_exact_quotation=settled.get("requires_exact_quotation"),  # type: ignore[arg-type]
        requires_citation=settled.get("requires_citation"),  # type: ignore[arg-type]
        multi_hop=settled.get("multi_hop"),  # type: ignore[arg-type]
        volatility=settled.get("volatility"),  # type: ignore[arg-type]
        coverage=min(1.0, coverage),
        fired=tuple(fired),
    )
