"""Fusing ranked lists from several sources.

**Reciprocal rank fusion is the default because it needs no score calibration.** A BM25 score
and a cosine similarity are not comparable — one is unbounded and corpus-dependent, the other
sits in [-1, 1] — so any scheme that adds or averages them is comparing units that do not share
a scale. RRF compares *positions*, which every ranker produces on the same scale by construction.

That property is worth more than the precision a tuned weighted sum might buy, because the
weights would need retuning whenever a source's corpus changed size, and nobody would notice
they had gone stale until retrieval quality had already drifted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prag.core.models.retrieval import FusionMethod

if TYPE_CHECKING:
    from collections.abc import Sequence

    from prag.core.models.retrieval import Candidate, FusionConfig, LegResult

__all__ = ["fuse_legs", "reciprocal_rank_fusion"]


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Candidate]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[Candidate]:
    """Fuse ranked lists by summed reciprocal rank.

    Each candidate scores ``weight / (k + rank)`` in every list it appears in, summed across
    lists. ``k`` damps the influence of the very top ranks: with a small ``k`` a first place is
    worth many times a third, which makes the fusion hostage to whichever source happened to be
    most confident. The conventional 60 is deliberately flat.

    A candidate found by two sources outranks one found by either alone, which is the point —
    independent agreement is evidence, and rank fusion is the cheapest way to spend it.
    """
    if not ranked_lists:
        return []

    scores: dict[str, float] = {}
    best: dict[str, Candidate] = {}
    ranks: dict[str, dict[str, int]] = {}

    for list_index, ranked in enumerate(ranked_lists):
        weight = weights[list_index] if weights and list_index < len(weights) else 1.0
        for rank, candidate in enumerate(ranked):
            # Keyed on the chunk rather than the candidate id, because the same chunk retrieved
            # by two sources arrives with two candidate ids and is one piece of evidence. Keying
            # on the candidate id would let a source's naming defeat the agreement signal.
            key = candidate.chunk_id or candidate.candidate_id
            scores[key] = scores.get(key, 0.0) + weight / (k + rank + 1)
            ranks.setdefault(key, {}).update(candidate.rank_by_leg or {})

            existing = best.get(key)
            if existing is None or candidate.effective_score > existing.effective_score:
                best[key] = candidate

    fused: list[Candidate] = []
    for key, score in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])):
        candidate = best[key]
        fused.append(
            candidate.model_copy(
                update={
                    "fused_score": round(score, 8),
                    # Per-leg ranks are merged rather than overwritten, so the trace can still
                    # say which source found a candidate and where — which is what makes a
                    # source's contribution measurable rather than assumed.
                    "rank_by_leg": {**ranks.get(key, {}), **(candidate.rank_by_leg or {})},
                }
            )
        )

    return fused


def _weighted_fusion(
    ranked_lists: Sequence[Sequence[Candidate]],
    *,
    weights: Sequence[float] | None = None,
) -> list[Candidate]:
    """Weighted sum of raw scores.

    Correct only where every list came from the same scorer, which in practice means a single
    source queried with several variants. Across sources it compares incommensurable units, and
    the failure is silent: the fused order looks plausible and is dominated by whichever source
    happens to produce the largest numbers.
    """
    scores: dict[str, float] = {}
    best: dict[str, Candidate] = {}

    for list_index, ranked in enumerate(ranked_lists):
        weight = weights[list_index] if weights and list_index < len(weights) else 1.0
        for candidate in ranked:
            key = candidate.chunk_id or candidate.candidate_id
            scores[key] = scores.get(key, 0.0) + weight * candidate.effective_score
            existing = best.get(key)
            if existing is None or candidate.effective_score > existing.effective_score:
                best[key] = candidate

    return [
        best[key].model_copy(update={"fused_score": round(score, 8)})
        for key, score in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def fuse_legs(leg_results: Sequence[LegResult], config: FusionConfig) -> tuple[Candidate, ...]:
    """Fuse the usable legs of one plan.

    Only usable legs contribute. A failed leg is absent rather than empty: an empty ranked list
    would still shift RRF's denominators for nothing, and a leg that returned nothing because it
    was unreachable should not look like a leg that searched and found nothing.
    """
    usable = [result for result in leg_results if result.usable and result.candidates]
    if not usable:
        return ()

    ranked = [list(result.candidates) for result in usable]
    weights = [config.weights.get(result.source_id, 1.0) for result in usable]

    if config.method is FusionMethod.WEIGHTED:
        fused = _weighted_fusion(ranked, weights=weights)
    else:
        fused = reciprocal_rank_fusion(ranked, k=config.k, weights=weights)

    return tuple(fused)
