"""Claim extraction, entailment, and citation binding.

One rule governs this module: **a citation is attached only after the evidence has been checked
against the claim.** Never before, and never because a marker happened to be nearby.

Attaching a plausible-looking citation to a claim its source does not support is worse than
attaching none. An uncited claim reads as the model's assertion and a reader discounts it
accordingly; a cited one reads as verified. Post-hoc citation attachment therefore converts an
unsupported statement into an apparently checked one, and it does so most convincingly exactly
where the model was least reliable.

Two failure modes are distinguished, because they mean different things:

**An unsupported claim** — nothing in the context entails it. The claim is marked unsourced, or
stripped under a strict tenant policy.

**An invalid citation** — a marker that resolves to no evidence group present in this request.
That is a hallucination *even when the claim happens to be true*, because the model has cited
something it was never given, and a system that tolerates it cannot tell a lucky guess from a
grounded answer.

The entailment scorer here is lexical and deliberately conservative. A calibrated NLI model
arrives with the evaluation phase; until then this errs toward marking claims unsourced rather
than toward attaching citations it cannot justify, because the direction of that error is the
whole point.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from prag.core.models.generation import Citation, ClaimVerdict, GroundingReport

if TYPE_CHECKING:
    from collections.abc import Sequence

    from prag.core.models.context import ContextBundle
    from prag.core.models.retrieval import EvidenceGroup

__all__ = [
    "Claim",
    "HeuristicGroundingVerifier",
    "cited_markers",
    "entailment_score",
    "extract_claims",
]

#: A citation marker as rendered into the evidence region: ``[E3]``.
_MARKER = re.compile(r"\[(E\d+)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
#: One or more citation markers at the head of a fragment, left there by sentence splitting.
_LEADING_MARKERS = re.compile(r"^\s*(?:\[E\d+\]\s*)+")
_WORD = re.compile(r"[a-z0-9][a-z0-9\-_.%/]*")

#: Tokens that carry no evidential weight. Kept minimal: in a technical corpus, "not", "before"
#: and "all" are frequently the entire claim, so an aggressive list would discard the words that
#: decide whether evidence supports a statement or contradicts it.
_FILLER_TEXT = (
    "a an the of to in for on at by with from is are was were be been being and or as "
    "that this it its there here also then thus so we you they"
)
_FILLER = frozenset(_FILLER_TEXT.split())

#: A sentence that asserts nothing factual — a question, or pure meta-commentary — has nothing
#: to be grounded against, and scoring it would drag the faithfulness metric down for a
#: sentence that was never a claim.
_NON_CLAIM_PREFIX = re.compile(
    r"^\s*("
    r"i (do not|don't|cannot|can't)\b"
    r"|there (is|are) no\b"
    r"|based on the (provided|available)\b"
    r")",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Claim:
    """One factual assertion extracted from an answer."""

    text: str
    #: Markers the model attached to this claim. Present here as *stated intent*, not as
    #: verified fact: whether they survive depends entirely on the entailment check.
    claimed_markers: tuple[str, ...]
    #: Character span within the answer, so a correction event can name what it retracted.
    span: tuple[int, int]

    @property
    def is_substantive(self) -> bool:
        """Whether this sentence asserts something that could be right or wrong.

        Questions and refusals are excluded. So are very short fragments, which are usually
        connectives rather than claims and which lexical entailment cannot score meaningfully.
        """
        stripped = self.text.strip()
        if len(stripped) < 12 or stripped.endswith("?"):
            return False
        return not _NON_CLAIM_PREFIX.match(stripped)


def cited_markers(text: str) -> tuple[str, ...]:
    """Citation markers appearing in a span of text, in order and deduplicated."""
    seen: list[str] = []
    for match in _MARKER.finditer(text):
        marker = match.group(1)
        if marker not in seen:
            seen.append(marker)
    return tuple(seen)


def extract_claims(answer: str) -> tuple[Claim, ...]:
    """Split an answer into claims, at sentence granularity.

    Sentences rather than clauses. Clause-level extraction catches more, but it needs a parser
    and it produces fragments whose truth depends on a neighbouring clause — which makes an
    entailment check on the fragment alone meaningless.
    """
    claims: list[Claim] = []
    offset = 0

    for raw in _SENTENCE.split(answer):
        if not raw.strip():
            offset += len(raw) + 1
            continue

        start = answer.find(raw, offset)
        start = offset if start < 0 else start
        offset = start + len(raw)

        # A citation is written after the sentence it supports — "…within 15 minutes. [E1]" —
        # so sentence splitting strands the marker at the head of the *next* sentence. Left
        # uncorrected, every claim is checked against the evidence for the claim before it, and
        # the verifier then strips citations that were correct all along.
        leading = _LEADING_MARKERS.match(raw)
        if leading and claims:
            trailing = cited_markers(leading.group(0))
            previous = claims[-1]
            claims[-1] = Claim(
                text=f"{previous.text} {leading.group(0).strip()}".strip(),
                claimed_markers=tuple(dict.fromkeys(previous.claimed_markers + trailing)),
                span=(previous.span[0], start + leading.end()),
            )
            raw = raw[leading.end() :]
            start += leading.end()
            if not raw.strip():
                continue

        claims.append(
            Claim(
                text=raw.strip(),
                claimed_markers=cited_markers(raw),
                span=(start, start + len(raw)),
            )
        )

    return tuple(claims)


def _content_tokens(text: str) -> frozenset[str]:
    stripped = _MARKER.sub(" ", text.lower())
    return frozenset(
        token for token in _WORD.findall(stripped) if token not in _FILLER and len(token) > 1
    )


def entailment_score(claim: str, evidence: str) -> float:
    """How much of the claim the evidence actually accounts for, in [0, 1].

    Directional on purpose: it measures the fraction of the *claim's* content that appears in
    the evidence, not their mutual similarity. A long passage that happens to contain a few of
    the claim's words should not score highly just because it is long, and a symmetric measure
    like Jaccard would penalise a short precise quote for being short.

    A lexical stand-in for NLI, and it cannot detect negation — "records are retained" and
    "records are not retained" score alike. That is a known limitation rather than a hidden one:
    the calibrated NLI model replaces this function without changing any caller, and until then
    the conservative threshold is what keeps the error on the safe side.
    """
    claim_tokens = _content_tokens(claim)
    if not claim_tokens:
        return 0.0

    evidence_tokens = _content_tokens(evidence)
    if not evidence_tokens:
        return 0.0

    return len(claim_tokens & evidence_tokens) / len(claim_tokens)


class HeuristicGroundingVerifier:
    """Verifies claims against evidence and binds only the citations that survive."""

    def __init__(
        self,
        *,
        entailment_threshold: float = 0.6,
        #: Whether a claim may be cited against evidence the model did not itself cite. Enabled,
        #: because a correct claim supported by evidence that was in the context deserves its
        #: citation — the model failing to name the marker is a formatting miss, not a
        #: grounding failure. It never invents support that is absent.
        allow_unclaimed_support: bool = True,
    ) -> None:
        self._threshold = entailment_threshold
        self._allow_unclaimed_support = allow_unclaimed_support

    async def verify(self, answer: str, bundle: ContextBundle) -> GroundingReport:
        """Check every claim, and return citations only for those the evidence supports."""
        by_marker = {group.citation_marker: group for group in bundle.evidence}
        claims = [c for c in extract_claims(answer) if c.is_substantive]

        verdicts: list[ClaimVerdict] = []
        citations: list[Citation] = []
        cited = 0

        for claim in claims:
            best_group, best_score = self._best_support(claim, bundle.evidence, by_marker)
            entailed = best_group is not None and best_score >= self._threshold

            if entailed and best_group is not None:
                cited += 1
                citations.append(self._bind(claim, best_group, best_score))

            verdicts.append(
                ClaimVerdict(
                    claim=claim.text,
                    entailed=entailed,
                    entailment_score=best_score,
                    cited_group_ids=(best_group.group_id,) if entailed and best_group else (),
                )
            )

        return GroundingReport(
            claims_total=len(claims),
            claims_cited=cited,
            claims_unsourced=len(claims) - cited,
            verdicts=tuple(verdicts),
        )

    def _best_support(
        self,
        claim: Claim,
        evidence: Sequence[EvidenceGroup],
        by_marker: dict[str, EvidenceGroup],
    ) -> tuple[EvidenceGroup | None, float]:
        """Find the evidence that best supports a claim, preferring what it cited.

        Markers the model stated are checked first. A marker naming a group that is not in this
        request's context is discarded outright rather than scored — it is a hallucinated
        citation, and scoring it would give a fabricated reference a chance to survive.
        """
        candidates: list[EvidenceGroup] = [
            by_marker[marker] for marker in claim.claimed_markers if marker in by_marker
        ]

        if not candidates and self._allow_unclaimed_support:
            candidates = list(evidence)

        best: EvidenceGroup | None = None
        best_score = 0.0
        for group in candidates:
            score = entailment_score(claim.text, group.representative.context_text)
            if score > best_score:
                best, best_score = group, score

        return best, best_score

    @staticmethod
    def _bind(claim: Claim, group: EvidenceGroup, score: float) -> Citation:
        representative = group.representative
        return Citation(
            marker=group.citation_marker,
            group_id=group.group_id,
            document_id=representative.document_id,
            document_version=representative.document_version,
            source_id=representative.source_id,
            span=claim.span,
            entailment_score=score,
        )


def strip_invalid_markers(answer: str, valid_markers: frozenset[str]) -> str:
    """Remove citation markers that resolve to nothing in this request.

    Runs after grounding, not before, so that stripping a claim does not leave an orphan marker
    behind and so that a marker removed here is only ever one the verifier already refused.
    """
    return _MARKER.sub(lambda m: m.group(0) if m.group(1) in valid_markers else "", answer).replace(
        "  ", " "
    )
