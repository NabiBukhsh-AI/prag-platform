"""Contradiction detection, and the conflict-rate monitor that closes serving back to training.

Every parametric-versus-retrieved conflict is a case where the weights were wrong. Aggregated
per adapter over a rolling window, the rate is the primary automated defence against parametric
drift: above a staleness rate the adapter's cluster is queued for retraining, above a critical
rate the adapter is demoted and its traffic falls back to the non-parametric path.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque

from prag.core.models.events import DomainEvent, EventKind
from prag.core.models.fusion import Stance

__all__ = ["ConflictMonitor", "contradicts", "stance"]

_WORD = re.compile(r"[a-z0-9][a-z0-9\-.]*")
_NUMBER = re.compile(r"^\d+(?:\.\d+)?$")
_NEGATIONS = frozenset({"not", "never", "no", "cannot", "without"})
_FILLER = frozenset(
    {
        "a", "an", "the", "of", "to", "in", "for", "on", "at", "by", "with", "from", "is",
        "are", "was", "were", "be", "been", "being", "and", "or", "as", "that", "this", "it",
        "its", "there",
    }
)  # fmt: skip


def _tokens(text: str) -> tuple[frozenset[str], frozenset[str], bool]:
    """Content words, numbers, and whether the text is negated."""
    words = [w.rstrip(".") for w in _WORD.findall(text.lower())]
    numbers = frozenset(w for w in words if _NUMBER.match(w))
    content = frozenset(
        w for w in words if w not in _FILLER and w not in _NEGATIONS and w not in numbers
    )
    return content, numbers, any(w in _NEGATIONS for w in words)


def stance(claim: str, evidence: str, *, topical_overlap: float = 0.6) -> Stance | None:
    """Whether ``evidence`` speaks to ``claim``, and if so whether it agrees.

    ``None`` when the evidence is about something else: most of the claim's non-numeric content
    must appear in it. On topic, different quantities or opposite polarity contradict; anything
    else supports. "Records are retained for 60 days" against "records are retained for 30 days"
    is the case that matters most — a lexical entailment scorer rates the two as nearly
    identical, which is exactly how a stale adapter's wrong number would otherwise earn a
    citation.

    ponytail: a lexical stand-in for NLI; the calibrated NLI model replaces this function and no
    caller changes.
    """
    claim_words, claim_numbers, claim_negated = _tokens(claim)
    evidence_words, evidence_numbers, evidence_negated = _tokens(evidence)
    if not claim_words or len(claim_words & evidence_words) / len(claim_words) < topical_overlap:
        return None
    if claim_numbers and evidence_numbers and not claim_numbers <= evidence_numbers:
        return Stance.CONTRADICTS
    if claim_negated != evidence_negated:
        return Stance.CONTRADICTS
    return Stance.SUPPORTS


def contradicts(claim: str, evidence: str) -> bool:
    return stance(claim, evidence) is Stance.CONTRADICTS


class ConflictMonitor:
    """Tracks each adapter's conflict rate over its most recent parametric attempts.

    Subscribes to the event bus. Every attempt is either served (``PARAMETRIC_SERVED``) or
    contradicted (``PARAMETRIC_RETRIEVAL_CONFLICT``), so the rate has an honest denominator.
    Decisions are queued rather than acted on, because acting means registry writes and the bus
    handler must not block on I/O; the platform applies them after publishing.
    """

    def __init__(
        self,
        *,
        window: int = 200,
        min_samples: int = 20,
        staleness_rate: float = 0.05,
        critical_rate: float = 0.20,
    ) -> None:
        self._outcomes: dict[str, deque[bool]] = defaultdict(lambda: deque(maxlen=window))
        self._min_samples = min_samples
        self._staleness = staleness_rate
        self._critical = critical_rate
        self._retrain: set[str] = set()
        self._demote: set[str] = set()
        self._acted: set[str] = set()

    def observe(self, event: DomainEvent) -> None:
        if event.kind is EventKind.PARAMETRIC_SERVED:
            conflicted = False
        elif event.kind is EventKind.PARAMETRIC_RETRIEVAL_CONFLICT:
            conflicted = True
        else:
            return
        for adapter in event.payload.get("adapters", ()):
            self._outcomes[adapter].append(conflicted)
            self._assess(adapter)

    def rate(self, adapter: str) -> float:
        outcomes = self._outcomes.get(adapter)
        return sum(outcomes) / len(outcomes) if outcomes else 0.0

    def _assess(self, adapter: str) -> None:
        if len(self._outcomes[adapter]) < self._min_samples or adapter in self._acted:
            return
        rate = self.rate(adapter)
        if rate > self._critical:
            self._demote.add(adapter)
            self._retrain.add(adapter)
            self._acted.add(adapter)
        elif rate > self._staleness:
            self._retrain.add(adapter)

    def take(self) -> tuple[frozenset[str], frozenset[str]]:
        """Pending ``(retrain, demote)`` decisions, as ``adapter_id@version``, then cleared."""
        retrain, demote = frozenset(self._retrain), frozenset(self._demote)
        self._retrain.clear()
        self._demote.clear()
        return retrain, demote
