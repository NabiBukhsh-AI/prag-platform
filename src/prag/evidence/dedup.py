"""Collapsing candidates into evidence groups.

Near-duplicates are **linked, not dropped**. That distinction is the whole point. Dropping them
loses the fact that several sources carried the same content; linking them records it, and that
record is what stops the agreement signal downstream from counting one fact three times.

Three documents quoting the same press release are one piece of evidence. A system that treats
them as three corroborating sources becomes most confident exactly where it is most wrong, and
it does so silently — which is why this is one of the more common quiet failures in production
RAG and why independence is computed here rather than assumed later.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from prag.core.ids import derive_id, short_hash
from prag.core.models.retrieval import EvidenceGroup

if TYPE_CHECKING:
    from collections.abc import Sequence

    from prag.core.models.retrieval import Candidate

__all__ = ["group_candidates", "normalized_fingerprint", "token_overlap"]

_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION = re.compile(r"[^\w\s]")


def normalized_fingerprint(text: str) -> str:
    """A hash that ignores formatting differences.

    The same paragraph reformatted, re-punctuated, or re-cased is the same evidence. Hashing the
    raw text would treat a Markdown copy and an HTML copy of one policy as independent sources,
    which is precisely the inflation this module exists to prevent.
    """
    lowered = _PUNCTUATION.sub(" ", text.lower())
    return short_hash(_WHITESPACE.sub(" ", lowered).strip())


def token_overlap(left: str, right: str) -> float:
    """Jaccard overlap of token sets.

    A cheap stand-in for embedding-cosine near-duplicate detection, and deliberately so at this
    stage: it needs no model, it is deterministic, and it catches the case that actually matters
    — the same passage reproduced with light edits. Embedding-based near-duplicate detection is
    a Phase 2 refinement over the candidate pool, not a prerequisite for grouping.
    """
    left_tokens = set(_WHITESPACE.split(_PUNCTUATION.sub(" ", left.lower()).strip()))
    right_tokens = set(_WHITESPACE.split(_PUNCTUATION.sub(" ", right.lower()).strip()))
    left_tokens.discard("")
    right_tokens.discard("")
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def group_candidates(
    candidates: Sequence[Candidate],
    *,
    near_duplicate_threshold: float = 0.85,
) -> tuple[EvidenceGroup, ...]:
    """Collapse candidates into evidence groups, marking which are independent.

    Grouping happens on two signals, in order:

    **Exact and near-duplicate text.** The same passage retrieved twice, or reproduced with
    light edits, is one group. Left ungrouped it fills the context window with copies and reads
    to the model as corroboration.

    **Shared lineage.** Two groups deriving from the same original document are *not*
    independent even when their text differs entirely — a summary and its source disagree
    textually while carrying exactly one source's authority. They stay separate groups, because
    they may say different things, but only the first is marked independent.

    Group order follows the best candidate's score, so packing sees the strongest evidence
    first without a second sort.
    """
    if not candidates:
        return ()

    ordered = sorted(candidates, key=lambda c: (-c.effective_score, c.candidate_id))

    members: list[list[Candidate]] = []
    fingerprints: list[str] = []

    for candidate in ordered:
        fingerprint = normalized_fingerprint(candidate.text)
        merged_into: int | None = None

        for index, existing in enumerate(fingerprints):
            if existing == fingerprint:
                merged_into = index
                break
            if token_overlap(members[index][0].text, candidate.text) >= near_duplicate_threshold:
                merged_into = index
                break

        if merged_into is None:
            members.append([candidate])
            fingerprints.append(fingerprint)
        else:
            members[merged_into].append(candidate)

    # Independence is decided across groups, after grouping: the first group to claim a lineage
    # root is independent, and every later group sharing it is one more view of the same source.
    seen_roots: set[str] = set()
    groups: list[EvidenceGroup] = []

    for index, group_members in enumerate(members):
        # The highest-authority member represents the group. Two copies of one passage can carry
        # different authority when they came from different sources, and the answer should be
        # attributed to the most authoritative one that carried it.
        representative = max(group_members, key=lambda c: (c.metadata.authority, c.effective_score))
        root = representative.metadata.lineage_root
        independent = root not in seen_roots
        seen_roots.add(root)

        groups.append(
            EvidenceGroup(
                group_id=derive_id("group", representative.chunk_id, str(index)),
                members=tuple(group_members),
                representative=representative,
                lineage_root=root,
                authority=max(c.metadata.authority for c in group_members),
                # Freshness is filled in by the freshness scorer, which needs the query's
                # half-life estimate. Defaulting to 1.0 here would silently assert that every
                # piece of evidence is current, so it stays neutral until something knows.
                freshness=0.5,
                independent=independent,
                citation_marker=f"E{index + 1}",
            )
        )

    return tuple(groups)
