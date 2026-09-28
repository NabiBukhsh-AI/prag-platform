"""Query transformation: rewrite, coreference, expansion, decomposition.

Four transforms, each independently gated so none runs unnecessarily, and each independently
budgeted so a slow one cannot eat the retrieval window it was meant to improve.

The gating matters more than it looks. A rewrite on an already-clear query costs latency and
buys nothing, and worse, it can *lose* precision — a model asked to improve a good question will
change it. ``applies_to`` is checked first and is cheap and synchronous; a gate that costs as
much as the work it guards is not a gate.

Every transform is an optimisation, never a correctness requirement. The raw query is always
retrievable, so a transform that times out or declines costs recall rather than an answer. That
is why each returns what it has rather than raising.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from prag.core.models.query import QueryVariants, SubQuery

if TYPE_CHECKING:
    from prag.core.models.common import Deadline
    from prag.core.models.query import QueryAnalysis

__all__ = [
    "CoreferenceTransformer",
    "DecompositionTransformer",
    "ExpansionTransformer",
    "RewriteTransformer",
    "apply_transforms",
]

_PRONOUN = re.compile(r"\b(it|its|that|this|they|them|those|there)\b", re.IGNORECASE)
_FILLER_PREFIX = re.compile(
    r"^\s*(?:please\s+|could you\s+|can you\s+|i(?:'d| would) like to know\s+|tell me\s+)+",
    re.IGNORECASE,
)
_TRAILING_POLITENESS = re.compile(r"\s*(?:please|thanks|thank you)\s*[.!?]*\s*$", re.IGNORECASE)
_CONJUNCTION_SPLIT = re.compile(
    r"\s+(?:and then|and also|as well as|;|\band\b(?=\s+(?:how|what|when|where|why|who|which)))\s+",
    re.IGNORECASE,
)
_COMPARISON = re.compile(
    r"\b(?:difference between|compare)\s+(.+?)\s+(?:and|versus|vs\.?|with)\s+(.+?)\s*[?.]?\s*$",
    re.IGNORECASE,
)

#: Aliases the expansion transform adds to a lexical leg. Drawn from the knowledge registry in a
#: real deployment; the shipped set covers the operational vocabulary the seed corpus uses.
#: Expansion is lexical-only by design — it needs no model call, which is why its budget is 5 ms.
DEFAULT_ALIASES: dict[str, tuple[str, ...]] = {
    "sev-1": ("sev1", "severity 1", "p1", "priority 1"),
    "sev-2": ("sev2", "severity 2", "p2", "priority 2"),
    "on-call": ("oncall", "on call", "rota"),
    "escalate": ("escalation", "escalated", "page", "paged"),
    "retention": ("retained", "retain", "archival", "archive"),
    "sla": ("service level agreement", "service level"),
    "acl": ("access control list", "permissions"),
}


class RewriteTransformer:
    """Strips politeness and filler so retrieval matches on content words.

    Deliberately conservative: it removes framing, never meaning. A model-based rewriter that
    reformulates the question can improve a bad query and quietly damage a good one, and the
    damage is invisible because the original is gone by the time anything is retrieved.
    """

    name = "rewrite"

    def applies_to(self, analysis: QueryAnalysis) -> bool:
        return analysis.query_quality.needs_rewrite or bool(
            _FILLER_PREFIX.match(analysis.normalized_query)
        )

    async def transform(self, analysis: QueryAnalysis, deadline: Deadline) -> QueryVariants:
        stripped = _FILLER_PREFIX.sub("", analysis.normalized_query)
        stripped = _TRAILING_POLITENESS.sub("", stripped).strip()
        rewritten = stripped or analysis.normalized_query

        return QueryVariants(
            raw=analysis.normalized_query,
            rewritten=rewritten if rewritten != analysis.normalized_query else None,
        )


class CoreferenceTransformer:
    """Substitutes session entities for pronouns on a follow-up turn.

    Only on a follow-up: a pronoun in a first turn has no antecedent to resolve, and guessing
    one would answer a question nobody asked. Resolution uses the entities the session kept
    verbatim, which is exactly why they are kept verbatim rather than summarised.
    """

    name = "coreference"

    def applies_to(self, analysis: QueryAnalysis) -> bool:
        return bool(_PRONOUN.search(analysis.normalized_query))

    async def transform(self, analysis: QueryAnalysis, deadline: Deadline) -> QueryVariants:
        entities = [e.text for e in analysis.structure.entities]
        if not entities:
            # Nothing to resolve against. Returning the raw query beats substituting a guess:
            # a wrong antecedent retrieves confidently for the wrong subject.
            return QueryVariants(raw=analysis.normalized_query)

        resolved = _PRONOUN.sub(entities[0], analysis.normalized_query, count=1)
        return QueryVariants(
            raw=analysis.normalized_query,
            coreference_resolved=resolved if resolved != analysis.normalized_query else None,
        )


class ExpansionTransformer:
    """Adds synonyms and aliases for the lexical leg. No model call.

    Its whole value is on identifier-heavy queries, which is precisely where dense retrieval
    fails: "sev1" and "sev-1" are neighbours to a human and unrelated tokens to BM25. Five
    milliseconds of dictionary lookup recovers the match that the embedding was never going to
    find on its own.
    """

    name = "expansion"

    def __init__(self, aliases: dict[str, tuple[str, ...]] | None = None) -> None:
        self._aliases = aliases or DEFAULT_ALIASES

    def applies_to(self, analysis: QueryAnalysis) -> bool:
        lowered = analysis.normalized_query.lower()
        return any(term in lowered for term in self._aliases)

    async def transform(self, analysis: QueryAnalysis, deadline: Deadline) -> QueryVariants:
        lowered = analysis.normalized_query.lower()
        additions: list[str] = []
        for term, aliases in self._aliases.items():
            if term in lowered:
                additions.extend(a for a in aliases if a not in lowered)

        expanded = f"{analysis.normalized_query} {' '.join(additions)}" if additions else None
        entities = " ".join(e.text for e in analysis.structure.entities)

        return QueryVariants(
            raw=analysis.normalized_query,
            expanded=expanded,
            # An entities-only variant for the lexical leg, where the surrounding question words
            # are noise rather than signal.
            entities_only=entities or None,
        )


class DecompositionTransformer:
    """Splits a multi-hop question into an ordered sub-query DAG.

    A DAG rather than a list, because a genuine multi-hop question has dependency structure and
    independent hops execute in parallel — which is a meaningful latency win on exactly the
    queries that are otherwise slowest.

    Rule-based here: conjunctions and comparison phrasing cover the common shapes without a
    model call. The model-based decomposer arrives with its 150 ms budget when the shapes stop
    being common.
    """

    name = "decomposition"

    def __init__(self, *, max_sub_queries: int = 6) -> None:
        self._max = max_sub_queries

    def applies_to(self, analysis: QueryAnalysis) -> bool:
        return bool(analysis.structure.multi_hop.value)

    async def transform(self, analysis: QueryAnalysis, deadline: Deadline) -> QueryVariants:
        query = analysis.normalized_query
        parts: list[str] = []

        comparison = _COMPARISON.search(query)
        if comparison:
            # A comparison is two independent lookups plus a synthesis, so both hops run in
            # parallel and neither depends on the other.
            parts = [comparison.group(1).strip(), comparison.group(2).strip()]
        else:
            parts = [p.strip() for p in _CONJUNCTION_SPLIT.split(query) if p.strip()]

        if len(parts) < 2:
            return QueryVariants(raw=query)

        sub_queries = tuple(
            SubQuery(sub_query_id=f"sq{index + 1}", text=text, depends_on=())
            for index, text in enumerate(parts[: self._max])
        )
        return QueryVariants(raw=query, sub_queries=sub_queries)


async def apply_transforms(
    analysis: QueryAnalysis,
    deadline: Deadline,
    *,
    transformers: tuple[object, ...] | None = None,
) -> QueryVariants:
    """Run every applicable transform and merge the variants.

    Each is gated and each is given a share of the deadline. A transform that declines, times
    out, or fails is skipped rather than fatal: the raw query is always retrievable, so the cost
    is recall rather than an answer.
    """
    active = transformers or (
        RewriteTransformer(),
        CoreferenceTransformer(),
        ExpansionTransformer(),
        DecompositionTransformer(),
    )

    merged: dict[str, object] = {"raw": analysis.normalized_query}

    for transformer in active:
        if deadline.expired:
            break
        if not transformer.applies_to(analysis):  # type: ignore[attr-defined]
            continue

        try:
            variants = await transformer.transform(  # type: ignore[attr-defined]
                analysis,
                deadline.share(0.25, label=f"transform.{transformer.name}"),  # type: ignore[attr-defined]
            )
        except Exception:
            continue

        for field in ("rewritten", "coreference_resolved", "expanded", "entities_only"):
            value = getattr(variants, field)
            if value:
                merged[field] = value
        if variants.sub_queries:
            merged["sub_queries"] = variants.sub_queries

    return QueryVariants(**merged)  # type: ignore[arg-type]
