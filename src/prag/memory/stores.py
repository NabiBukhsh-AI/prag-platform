"""Session and long-term memory, architecture §14.

One implementation, configured per namespace:

- **Session** memory is one conversation. Items expire after a TTL, and only the most recent
  turns enter context; older turns are compressed into a rolling summary with entities and
  decisions kept verbatim, and the raw turns stay in storage for audit.
- **Long-term** memory is facts about a user. It accepts only user-asserted or confirmed
  structured facts, ranks by relevance against the query and by salience that decays with age
  and disuse, and evicts the least valuable item when full rather than the oldest.

Both are keyed by tenant *and* user, and both enforce write gating themselves. Model-generated
content never persists as a long-term fact: that single rule stops a hallucination from becoming
a stored "fact" about the user, which the system could never recover from on its own.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from prag.core.errors import MemoryWriteRefused
from prag.core.ids import short_hash
from prag.core.models.common import MemoryNamespace, Provenance
from prag.core.models.memory import MemoryItem, MemorySelector, SessionSummary

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from prag.core.models.identity import Principal

__all__ = ["PrincipalMemoryStore", "decisions_in", "entities_in"]

_DAY_MS = 86_400_000
_HOUR_MS = 3_600_000
_WORD = re.compile(r"[a-z0-9][a-z0-9\-]*")
_FILLER = frozenset(
    {
        "a", "an", "the", "of", "to", "in", "for", "on", "at", "by", "with", "from", "is", "are",
        "was", "were", "be", "and", "or", "as", "that", "this", "it", "what", "which", "who",
        "how", "do", "does", "did", "i", "we", "you", "my", "our", "me", "us",
    }
)  # fmt: skip
_DECISION = re.compile(
    r"\b(chose|choose|decided|decide|agreed|agree|approved|selected|picked|settled on|"
    r"going with|will use|we'll use|prefer)\b",
    re.IGNORECASE,
)
_ENTITY = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][A-Za-z0-9]*(?:[-\s][A-Z][A-Za-z0-9]*)*")


def _tokens(text: str) -> frozenset[str]:
    return frozenset(w for w in _WORD.findall(text.lower()) if w not in _FILLER and len(w) > 1)


def decisions_in(items: Sequence[MemoryItem]) -> tuple[str, ...]:
    """User-stated decisions, verbatim. A summary is most likely to blur exactly these."""
    return tuple(
        i.text
        for i in items
        if i.provenance is Provenance.USER_ASSERTED and _DECISION.search(i.text)
    )


def entities_in(items: Sequence[MemoryItem]) -> tuple[str, ...]:
    """Capitalised names the user used, verbatim, in first-seen order.

    Sentence-initial words are skipped: "We" is not an entity. Coreference resolution needs these
    exactly as written, so they are never paraphrased.
    """
    seen: dict[str, None] = {}
    for item in items:
        if item.provenance is not Provenance.USER_ASSERTED:
            continue
        for match in _ENTITY.finditer(item.text):
            seen.setdefault(match.group().strip(), None)
    return tuple(seen)


class PrincipalMemoryStore:
    """Implements ``MemoryStore`` for one namespace."""

    def __init__(
        self,
        namespace: MemoryNamespace,
        *,
        ttl_hours: float | None = None,
        window: int | None = None,
        max_items: int | None = None,
        decay_half_life_days: float | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.namespace = namespace
        self._ttl_ms = ttl_hours * _HOUR_MS if ttl_hours else None
        #: Session only: how many recent turns enter context. Older ones are summarised.
        self._window = window
        self._max_items = max_items
        self._half_life = decay_half_life_days
        self._clock = clock
        self._items: dict[tuple[str, str], list[MemoryItem]] = {}
        self._summaries: dict[tuple[str, str, str], SessionSummary] = {}

    @classmethod
    def session(
        cls, *, ttl_hours: float = 24, window: int = 8, **kw: object
    ) -> PrincipalMemoryStore:
        return cls(
            MemoryNamespace.SESSION,
            ttl_hours=ttl_hours,
            window=window,
            **kw,  # type: ignore[arg-type]
        )

    @classmethod
    def long_term(
        cls, *, max_items: int = 500, decay_half_life_days: float = 60.0, **kw: object
    ) -> PrincipalMemoryStore:
        return cls(
            MemoryNamespace.LONG_TERM,
            max_items=max_items,
            decay_half_life_days=decay_half_life_days,
            **kw,  # type: ignore[arg-type]
        )

    @staticmethod
    def _key(principal: Principal) -> tuple[str, str]:
        return (principal.tenant_id, principal.user_id)

    def _now(self) -> int:
        return int(self._clock() * 1000)

    def _live(self, principal: Principal) -> list[MemoryItem]:
        """The principal's items, with expired ones dropped for good."""
        key = self._key(principal)
        items = self._items.get(key, [])
        if self._ttl_ms is not None:
            cutoff = self._now() - self._ttl_ms
            items = [i for i in items if i.created_at_ms >= cutoff]
            self._items[key] = items
        return items

    def _decayed(self, item: MemoryItem, now: int) -> float:
        """Salience after decay. Use refreshes it: the clock runs from the last access."""
        if not self._half_life:
            return item.salience
        reference = max(item.created_at_ms, item.last_accessed_ms or 0)
        age_days = max(0.0, (now - reference) / _DAY_MS)
        return item.salience * 0.5 ** (age_days / self._half_life)

    def _recent(self, items: list[MemoryItem]) -> list[MemoryItem]:
        """The last ``window`` items of each session; the rest live on only in the summary."""
        if not self._window:
            return items
        by_session: dict[str | None, list[MemoryItem]] = {}
        for item in items:
            by_session.setdefault(item.session_id, []).append(item)
        return [i for group in by_session.values() for i in group[-self._window :]]

    async def read(
        self,
        principal: Principal,
        query: str,
        limit: int,
        *,
        session_id: str | None = None,
    ) -> Sequence[MemoryItem]:
        now = self._now()
        items = self._live(principal)
        if session_id is not None:
            items = [i for i in items if i.session_id == session_id]
        items = self._recent(items)

        asked = _tokens(query)
        scored = []
        for item in items:
            overlap = len(asked & _tokens(item.text)) / len(asked) if asked else 0.0
            # Long-term facts enter context only when they bear on the question; recent session
            # turns always may, because a follow-up rarely repeats the words it refers back to.
            if self.namespace is MemoryNamespace.LONG_TERM and asked and not overlap:
                continue
            scored.append((overlap, self._decayed(item, now), item.created_at_ms, item))
        scored.sort(key=lambda s: s[:3], reverse=True)
        chosen = [s[3] for s in scored[: max(0, limit)]]

        if self._half_life and chosen:
            touched = {i.item_id for i in chosen}
            key = self._key(principal)
            self._items[key] = [
                i.model_copy(update={"last_accessed_ms": now}) if i.item_id in touched else i
                for i in self._items.get(key, [])
            ]
        return chosen

    async def write(self, principal: Principal, item: MemoryItem) -> None:
        if item.namespace is not self.namespace:
            raise MemoryWriteRefused(
                "item belongs to another namespace",
                store=str(self.namespace),
                item=str(item.namespace),
            )
        if not item.persistable:
            raise MemoryWriteRefused(
                "provenance not permitted in this namespace",
                namespace=str(item.namespace),
                provenance=str(item.provenance),
            )

        items = self._live(principal)
        if self.namespace is MemoryNamespace.LONG_TERM:
            # A fact stated twice is one fact, restated: keep it once, at the higher salience,
            # with its clock refreshed.
            normalized = " ".join(item.text.lower().split())
            for index, existing in enumerate(items):
                if " ".join(existing.text.lower().split()) == normalized:
                    items[index] = existing.model_copy(
                        update={
                            "salience": max(existing.salience, item.salience),
                            "last_accessed_ms": self._now(),
                        }
                    )
                    return
        items.append(item)

        if self._max_items and len(items) > self._max_items:
            # Evict the least valuable, not the oldest: a standing preference stated a year ago
            # can matter more than yesterday's small talk.
            now = self._now()
            items.remove(min(items, key=lambda i: (self._decayed(i, now), i.created_at_ms)))
        self._items[self._key(principal)] = items

    async def summarize(self, principal: Principal, session_id: str) -> SessionSummary:
        """Roll up one conversation. Entities and decisions verbatim; older turns compressed.

        ponytail: extractive stand-in for the small summarisation model; it keeps the first
        sentence of each compressed turn. The verbatim slots are the contract and do not change
        when the model does.
        """
        items = [i for i in self._live(principal) if i.session_id == session_id]
        older = items[: -self._window] if self._window else items
        prose = " ".join(re.split(r"(?<=[.!?])\s+", i.text.strip())[0] for i in older)[:1_000]
        entities, decisions = entities_in(items), decisions_in(items)
        summary = SessionSummary(
            session_id=session_id,
            summary=prose,
            turns_summarized=len(older),
            verbatim_entities=entities,
            verbatim_decisions=decisions,
            summary_hash=short_hash(f"{prose}|{entities}|{decisions}"),
            updated_at_ms=self._now(),
        )
        self._summaries[(*self._key(principal), session_id)] = summary
        return summary

    def latest_summary(self, principal: Principal, session_id: str) -> SessionSummary | None:
        return self._summaries.get((*self._key(principal), session_id))

    def turn_count(self, principal: Principal, session_id: str) -> int:
        return sum(1 for i in self._live(principal) if i.session_id == session_id)

    def history(self, principal: Principal, session_id: str) -> tuple[MemoryItem, ...]:
        """Every retained turn, for audit. Not what enters context; ``read`` decides that."""
        return tuple(i for i in self._live(principal) if i.session_id == session_id)

    async def forget(self, principal: Principal, selector: MemorySelector) -> int:
        if selector.is_empty:
            return 0
        key = self._key(principal)
        before = self._items.get(key, [])
        kept = [item for item in before if not _matches(item, selector)]
        self._items[key] = kept
        if selector.session_id is not None:
            self._summaries.pop((*key, selector.session_id), None)
        return len(before) - len(kept)


def _matches(item: MemoryItem, selector: MemorySelector) -> bool:
    """Every set field must match. Unset fields do not widen the match."""
    return all(
        (
            not selector.item_ids or item.item_id in selector.item_ids,
            selector.namespace is None or item.namespace is selector.namespace,
            selector.provenance is None or item.provenance is selector.provenance,
            selector.session_id is None or item.session_id == selector.session_id,
            selector.older_than_ms is None or item.created_at_ms < selector.older_than_ms,
        )
    )
