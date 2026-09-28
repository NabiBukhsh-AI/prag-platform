"""Reranking tiers.

Reranking is the most expensive optional stage on the retrieval path — a top-50 cross-encoder
rerank is around 120 ms on GPU and 400 ms or worse on CPU — and it is the first thing the
degradation ladder drops. Everything here follows from that:

**Every reranker must be genuinely skippable.** Nothing downstream may require a rerank score to
exist. The fused ordering is always a valid ordering, and a reranker that made itself mandatory
would remove the ladder's cheapest rung.

**A deadline breach truncates rather than raises.** A partially reranked list is strictly better
than none, and the caller already decided it would rather have fusion ordering than nothing.

The tier is chosen per SLA rather than globally, because the same 120 ms is negligible against a
30 second batch budget and unaffordable against a 650 ms interactive one.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from prag.core.errors import DeadlineExceeded

if TYPE_CHECKING:
    from collections.abc import Sequence

    from prag.core.models.common import Deadline
    from prag.core.models.retrieval import Candidate

__all__ = ["LexicalOverlapReranker", "NoOpReranker", "RerankOutcome", "rerank_with_budget"]

_WORD = re.compile(r"[a-z0-9][a-z0-9\-_.]*")
_STOPWORD_TEXT = (
    "a an the of to in for on at by with from is are was were be been being and or as "
    "that this it its what which who when where why how do does did can could should would must"
)
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def _content(text: str) -> frozenset[str]:
    return frozenset(t for t in _WORD.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2)


class NoOpReranker:
    """Returns the input order, truncated.

    Not a placeholder — it is the ``none`` tier, and it exists so that "no reranking" is a
    configured choice rather than a missing dependency. A deployment that has not installed a
    cross-encoder still has a valid, declared rerank tier.
    """

    model_id = "none"

    async def rerank(
        self,
        query: str,
        candidates: Sequence[Candidate],
        top_k: int,
        deadline: Deadline,
    ) -> Sequence[Candidate]:
        return tuple(candidates[:top_k])


class LexicalOverlapReranker:
    """Scores query-candidate overlap, with a position prior.

    A stand-in for a cross-encoder, and honest about being one: it captures term overlap and
    nothing about meaning, so it will not rescue a candidate whose wording differs from the
    query's. What it does capture is the case that matters most in practice — a candidate that
    ranked well on fused position *and* actually contains the query's terms.

    The position prior is deliberate. A reranker that ignored the fused order would discard the
    evidence that several independent sources agreed, which is usually a stronger signal than
    surface overlap. It shrinks toward the incoming ranking rather than replacing it.
    """

    model_id = "lexical-overlap"

    def __init__(self, *, position_weight: float = 0.35) -> None:
        if not 0.0 <= position_weight <= 1.0:
            raise ValueError(f"position_weight must be in [0, 1], got {position_weight}")
        self._position_weight = position_weight

    async def rerank(
        self,
        query: str,
        candidates: Sequence[Candidate],
        top_k: int,
        deadline: Deadline,
    ) -> Sequence[Candidate]:
        if not candidates:
            return ()

        query_tokens = _content(query)
        if not query_tokens:
            return tuple(candidates[:top_k])

        scored: list[tuple[float, int, Candidate]] = []
        for position, candidate in enumerate(candidates):
            if deadline.expired:
                # Out of time mid-pass. Keeping what was scored and appending the rest in their
                # incoming order beats discarding the work or raising: both would throw away a
                # partial improvement the caller can use.
                remaining = candidates[position:]
                scored.sort(key=lambda item: (-item[0], item[1]))
                partial = [c.model_copy(update={"rerank_score": s}) for s, _, c in scored]
                return tuple((partial + list(remaining))[:top_k])

            overlap = len(query_tokens & _content(candidate.context_text)) / len(query_tokens)
            prior = 1.0 / (1.0 + position)
            score = (1.0 - self._position_weight) * overlap + self._position_weight * prior
            scored.append((round(score, 8), position, candidate))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return tuple(
            candidate.model_copy(update={"rerank_score": score})
            for score, _, candidate in scored[:top_k]
        )


class RerankOutcome:
    """What reranking did, or why it did not run.

    ``skipped_reason`` is the field worth having. A request that skipped the reranker and one
    that ran it and found nothing better produce similar orderings and very different
    explanations, and the difference is what tells anyone whether the ladder is firing on normal
    traffic.
    """

    __slots__ = ("candidates", "elapsed_ms", "ran", "skipped_reason")

    def __init__(
        self,
        candidates: tuple[Candidate, ...],
        *,
        ran: bool,
        skipped_reason: str | None = None,
        elapsed_ms: int = 0,
    ) -> None:
        self.candidates = candidates
        self.ran = ran
        self.skipped_reason = skipped_reason
        self.elapsed_ms = elapsed_ms


async def rerank_with_budget(
    reranker: object,
    query: str,
    candidates: Sequence[Candidate],
    *,
    input_k: int,
    output_k: int,
    deadline: Deadline,
    allowed: bool = True,
) -> RerankOutcome:
    """Rerank if the budget allows, and report honestly when it does not.

    ``allowed`` comes from the degradation ladder. At level 1 the reranker is skipped and the
    fused ordering stands — the cheapest rung, chosen first because it sheds the most cost for
    the least quality.
    """
    if not allowed:
        return RerankOutcome(
            tuple(candidates[:output_k]), ran=False, skipped_reason="degraded_budget"
        )
    if not candidates:
        return RerankOutcome((), ran=False, skipped_reason="no_candidates")
    if deadline.expired:
        return RerankOutcome(
            tuple(candidates[:output_k]), ran=False, skipped_reason="deadline_exceeded"
        )

    started = time.monotonic()
    try:
        reranked = await reranker.rerank(  # type: ignore[attr-defined]
            query, list(candidates[:input_k]), output_k, deadline
        )
    except DeadlineExceeded:
        return RerankOutcome(
            tuple(candidates[:output_k]), ran=False, skipped_reason="deadline_exceeded"
        )
    except Exception:
        # A broken reranker degrades the ordering; it must not fail the request. The fused order
        # was already valid, which is exactly why this stage is allowed to be skippable.
        return RerankOutcome(
            tuple(candidates[:output_k]), ran=False, skipped_reason="reranker_error"
        )

    return RerankOutcome(
        tuple(reranked),
        ran=True,
        elapsed_ms=int((time.monotonic() - started) * 1000),
    )
