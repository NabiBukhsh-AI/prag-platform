"""Evidence selection: which groups go in the window, and in what order.

Packing is a constrained selection problem — maximise expected information gain per token,
subject to the evidence budget — solved greedily by value density.

Two terms in the value function do the work that a naive relevance sort cannot:

**Redundancy**, computed as marginal relevance against what is already selected. Without it the
greedy pass fills the entire budget with eight near-identical chunks about the most salient
aspect of the query, because those are exactly the chunks that score highest.

**Coverage bonus**, which rewards evidence addressing a query aspect nothing selected has
touched yet. It is what makes the difference between answering the loudest part of a question
very well and answering the whole question.

Ordering is separate from selection and matters on its own. Attention over a long context is not
uniform, so the same evidence in a different order produces measurably different answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from prag.core.models.context import OrderingMode
from prag.ingestion.chunking.tokens import estimate_tokens

if TYPE_CHECKING:
    from collections.abc import Sequence

    from prag.core.models.retrieval import EvidenceGroup

__all__ = ["PackedEvidence", "order_groups", "pack_evidence", "query_aspects"]

_WORD = re.compile(r"[a-z0-9][a-z0-9\-_.]*")

#: Words that appear in nearly every query and therefore distinguish nothing. Kept small on
#: purpose: an aggressive stop list removes terms that are genuinely load-bearing in a technical
#: corpus, where "not", "before" and "all" often are the question.
_STOPWORD_TEXT = (
    "a an the of to in for on at by with from is are was were be been being do does did "
    "what which who whom whose when where why how and or if then than that this these those "
    "i you we they it its as can could should would may might must our your their"
)
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def query_aspects(query: str) -> frozenset[str]:
    """The distinct things a query is asking about.

    A deliberately crude proxy for entities, sub-questions and constraints: the content words.
    It is enough to stop the packer from spending its whole budget on one aspect, which is the
    failure the coverage bonus exists to prevent. A real aspect extractor arrives with the query
    intelligence layer, and this function is where it will land.
    """
    return frozenset(
        word for word in _WORD.findall(query.lower()) if word not in _STOPWORDS and len(word) > 2
    )


def _covered_aspects(text: str, aspects: frozenset[str]) -> frozenset[str]:
    lowered = text.lower()
    return frozenset(aspect for aspect in aspects if aspect in lowered)


def _redundancy(group: EvidenceGroup, selected: Sequence[EvidenceGroup]) -> float:
    """Maximal marginal relevance against the already-selected set.

    Takes the *maximum* similarity to any selected group rather than the mean. A chunk that
    duplicates one already-selected chunk is redundant regardless of how different it is from
    the rest, and averaging would hide that behind the others.
    """
    if not selected:
        return 0.0

    from prag.evidence.dedup import token_overlap

    return max(
        token_overlap(group.representative.text, other.representative.text) for other in selected
    )


@dataclass(frozen=True, slots=True)
class PackedEvidence:
    """The outcome of packing."""

    selected: tuple[EvidenceGroup, ...]
    dropped: tuple[EvidenceGroup, ...]
    used_tokens: int
    #: Fraction of query aspects that some selected group mentions. Feeds the coverage warning,
    #: and it is reported even when packing succeeded, because "it fit" and "it covered the
    #: question" are different claims.
    coverage: float
    uncovered_aspects: tuple[str, ...]

    @property
    def has_coverage_gap(self) -> bool:
        return bool(self.uncovered_aspects)


def pack_evidence(
    groups: Sequence[EvidenceGroup],
    *,
    query: str,
    budget_tokens: int,
    redundancy_weight: float = 1.0,
    coverage_weight: float = 0.5,
) -> PackedEvidence:
    """Greedily select groups by value density until the budget is spent.

    Value density rather than raw value: a group worth twice as much but costing three times the
    tokens is a worse buy, and selecting by value alone fills the window with long chunks that
    crowd out several better short ones.
    """
    aspects = query_aspects(query)
    remaining = list(groups)
    selected: list[EvidenceGroup] = []
    dropped: list[EvidenceGroup] = []
    covered: set[str] = set()
    used = 0

    while remaining:
        scored: list[tuple[float, int, EvidenceGroup]] = []
        for index, group in enumerate(remaining):
            cost = estimate_tokens(group.representative.context_text)
            if cost == 0:
                continue

            base = group.representative.effective_score * group.authority * group.freshness
            redundancy = _redundancy(group, selected)
            new_aspects = _covered_aspects(group.representative.context_text, aspects) - covered
            bonus = 1.0 + coverage_weight * (len(new_aspects) / len(aspects) if aspects else 0.0)

            value = base * max(0.0, 1.0 - redundancy_weight * redundancy) * bonus
            # Index is the tie-break, so equal-value groups always resolve the same way and a
            # recorded request replays to the same selection.
            scored.append((value / cost, index, group))

        if not scored:
            break

        scored.sort(key=lambda item: (-item[0], item[1]))
        _, best_index, best = scored[0]
        cost = estimate_tokens(best.representative.context_text)

        if used + cost > budget_tokens:
            # Does not fit. Drop it and keep going rather than stopping: a smaller group further
            # down the list may still fit, and stopping at the first overflow would waste the
            # remaining budget on nothing.
            dropped.append(best)
            remaining.pop(best_index)
            continue

        selected.append(best)
        covered |= _covered_aspects(best.representative.context_text, aspects)
        used += cost
        remaining.pop(best_index)

    uncovered = tuple(sorted(aspects - covered))
    coverage = (len(covered) / len(aspects)) if aspects else 1.0

    return PackedEvidence(
        selected=tuple(selected),
        dropped=tuple(dropped),
        used_tokens=used,
        coverage=coverage,
        uncovered_aspects=uncovered,
    )


def order_groups(groups: Sequence[EvidenceGroup], mode: OrderingMode) -> tuple[EvidenceGroup, ...]:
    """Arrange selected evidence within its region.

    Not cosmetic. Attention over a long context is not uniform, and the same evidence in a
    different order produces measurably different answers.
    """
    if not groups:
        return ()

    ranked = sorted(groups, key=lambda g: (-g.representative.effective_score, g.group_id))

    match mode:
        case OrderingMode.DESCENDING:
            return tuple(ranked)

        case OrderingMode.CHRONOLOGICAL:
            # Correct for narrative and timeline queries, where the order of events is the
            # answer and re-sorting by relevance destroys it.
            return tuple(
                sorted(groups, key=lambda g: (g.representative.metadata.updated_at_ms, g.group_id))
            )

        case OrderingMode.SOURCE_GROUPED:
            # Correct when the answer must compare sources: interleaving them forces the model
            # to reassemble each position from fragments scattered through the region.
            return tuple(
                sorted(
                    groups,
                    key=lambda g: (
                        g.representative.source_id,
                        -g.representative.effective_score,
                        g.group_id,
                    ),
                )
            )

        case OrderingMode.EDGE_WEIGHTED:
            # Strongest evidence at the start and the end, weakest in the middle. A direct
            # mitigation for the lost-in-the-middle effect, and measurably better than
            # descending order once the evidence region exceeds roughly 3k tokens.
            head: list[EvidenceGroup] = []
            tail: list[EvidenceGroup] = []
            for position, group in enumerate(ranked):
                (head if position % 2 == 0 else tail).append(group)
            return tuple(head + list(reversed(tail)))

    raise ValueError(f"unknown ordering mode {mode!r}")
