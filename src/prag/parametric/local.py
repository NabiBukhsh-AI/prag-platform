"""The local stand-in for LoRA training and multi-LoRA inference.

**This is not a language model and does not learn.** It "trains" by storing the filtered QA
pairs in the weight blob, and "infers" by matching a question against the stored questions by
token overlap. It exists so that everything around training — augmentation filtering, held-out
probes, interference checks, promotion gates, residency, revocation, tenant scoping, provenance
shadowing — runs end to end on a laptop, with no GPU.

What it faithfully reproduces is the property that makes parametric knowledge dangerous: it
answers with no evidence in context, cites nothing, and cannot be filtered per request. What it
does not reproduce is anything about how well a real adapter learns. Numbers measured with it
test the pipeline's mechanics and gates, never the value of LoRA. The real trainer (PEFT on a
GPU worker) and server (vLLM multi-LoRA) replace ``MemorizingTrainer`` and ``answer_from``
behind the same shapes.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = ["MemorizingTrainer", "ParametricAnswer", "QAPair", "answer_from", "content_tokens"]

_WORD = re.compile(r"[a-z0-9][a-z0-9\-]*")
_FILLER = frozenset(
    {
        "a", "an", "the", "of", "to", "in", "for", "on", "at", "by", "with", "from",
        "is", "are", "was", "were", "be", "been", "being", "and", "or", "as", "that",
        "this", "it", "its", "what", "which", "who", "when", "where", "why", "how",
        "do", "does", "did", "can", "could", "should", "would", "must", "will",
        "about", "tell", "me", "please", "our", "your",
    }
)  # fmt: skip
_STEM = 5


def content_tokens(text: str) -> frozenset[str]:
    """Content words, each with a short stem so "escalated" meets "escalation"."""
    tokens: set[str] = set()
    for word in _WORD.findall(text.lower()):
        if word in _FILLER or len(word) <= 2:
            continue
        tokens.add(word)
        if len(word) > _STEM:
            tokens.add(word[:_STEM])
    return frozenset(tokens)


@dataclass(frozen=True, slots=True)
class QAPair:
    question: str
    answer: str
    document_id: str
    #: Which surface of the fact this is. Probes hold one surface out, so recall measures
    #: whether a fact is reachable under a phrasing the adapter never saw.
    surface: int = 0


@dataclass(frozen=True, slots=True)
class ParametricAnswer:
    text: str
    #: Overlap between the question and the best stored question, in [0, 1]. The stand-in's
    #: equivalent of a token logprob, and just as uncalibrated.
    confidence: float
    document_ids: tuple[str, ...] = ()


class MemorizingTrainer:
    """Produces a weight blob from QA pairs. See the module docstring for what this is not."""

    def train(self, pairs: Iterable[QAPair]) -> bytes:
        rows = sorted((asdict(p) for p in pairs), key=lambda r: (r["question"], r["answer"]))
        return json.dumps({"format": "memorizing.v1", "pairs": rows}).encode("utf-8")


def _pairs(weights: bytes) -> list[QAPair]:
    data = json.loads(weights.decode("utf-8"))
    return [QAPair(**row) for row in data["pairs"]]


def answer_from(
    weights: Sequence[bytes], question: str, *, weights_scale: Sequence[float] | None = None
) -> ParametricAnswer:
    """Answer from one or more merged adapters, with no evidence in context.

    Merging concatenates the adapters' tables, scaled by ``weights_scale``. That is where
    interference shows up in the stand-in: a sibling with a better-matching but different
    answer displaces the right one, which is the stand-in's version of summed deltas degrading
    each other.
    """
    asked = content_tokens(question)
    if not asked:
        return ParametricAnswer(text="", confidence=0.0)

    scale = list(weights_scale or [1.0] * len(weights))
    best: tuple[float, QAPair] | None = None
    for blob, factor in zip(weights, scale, strict=True):
        for pair in _pairs(blob):
            stored = content_tokens(pair.question) | content_tokens(pair.answer)
            score = factor * len(asked & stored) / len(asked)
            if best is None or (score, pair.answer) > (best[0], best[1].answer):
                best = (score, pair)

    if best is None:
        return ParametricAnswer(text="", confidence=0.0)
    score, pair = best
    return ParametricAnswer(
        text=pair.answer, confidence=min(1.0, score), document_ids=(pair.document_id,)
    )
