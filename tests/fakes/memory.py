"""An in-memory ``MemoryStore``."""

from __future__ import annotations

import time
from collections.abc import Sequence

from prag.core.errors import PragError
from prag.core.ids import derive_id, short_hash
from prag.core.models.common import MemoryNamespace, Provenance
from prag.core.models.identity import Principal
from prag.core.models.memory import MemoryItem, MemorySelector, SessionSummary

__all__ = ["InMemoryMemoryStore"]


class InMemoryMemoryStore:
    """A memory tier backed by a dict, keyed by principal.

    Enforces the write-gating rule rather than trusting callers, which is the behaviour the
    contract suite exists to verify: a long-term store that accepts model-generated content will
    pass every test that only checks reads.
    """

    def __init__(self, namespace: MemoryNamespace = MemoryNamespace.SESSION) -> None:
        self.namespace = namespace
        self._items: dict[tuple[str, str], list[MemoryItem]] = {}

    @staticmethod
    def _key(principal: Principal) -> tuple[str, str]:
        # Keyed by tenant *and* user. Tenant alone would let two users in one tenant read each
        # other's memory, which is a leak that a single-tenant test would never surface.
        return (principal.tenant_id, principal.user_id)

    async def read(self, principal: Principal, query: str, limit: int) -> Sequence[MemoryItem]:
        items = self._items.get(self._key(principal), [])
        terms = {t for t in query.lower().split() if t}
        scored = [item for item in items if not terms or terms & set(item.text.lower().split())]
        scored.sort(key=lambda i: (i.salience, i.created_at_ms), reverse=True)
        return scored[:limit]

    async def write(self, principal: Principal, item: MemoryItem) -> None:
        if item.namespace is not self.namespace:
            raise PragError(
                "item namespace does not match store namespace",
                store=str(self.namespace),
                item=str(item.namespace),
            )

        # The rule that matters: long-term memory refuses model-generated content. Letting the
        # model's own output persist as a user fact is how a system starts believing things
        # nobody told it, and that belief outlives the session that created it.
        if not item.persistable:
            raise PragError(
                "provenance not permitted in this namespace",
                namespace=str(item.namespace),
                provenance=str(item.provenance),
            )

        self._items.setdefault(self._key(principal), []).append(item)

    async def summarize(self, principal: Principal, session_id: str) -> SessionSummary:
        items = self._items.get(self._key(principal), [])
        text = " ".join(i.text for i in items)
        return SessionSummary(
            session_id=session_id,
            summary=text[:500],
            turns_summarized=len(items),
            verbatim_entities=(),
            verbatim_decisions=tuple(
                i.text for i in items if i.provenance is Provenance.USER_ASSERTED
            ),
            summary_hash=short_hash(text),
            updated_at_ms=int(time.time() * 1000),
        )

    async def forget(self, principal: Principal, selector: MemorySelector) -> int:
        # An empty selector matches nothing. A forget call that wipes a principal's entire
        # memory because a field was left unset is not a failure mode worth leaving open.
        if selector.is_empty:
            return 0

        key = self._key(principal)
        before = self._items.get(key, [])
        kept = [item for item in before if not self._matches(item, selector)]
        self._items[key] = kept
        return len(before) - len(kept)

    @staticmethod
    def _matches(item: MemoryItem, selector: MemorySelector) -> bool:
        if selector.item_ids and item.item_id not in selector.item_ids:
            return False
        if selector.namespace is not None and item.namespace is not selector.namespace:
            return False
        if selector.provenance is not None and item.provenance is not selector.provenance:
            return False
        return not (
            selector.older_than_ms is not None and item.created_at_ms >= selector.older_than_ms
        )


def make_memory_item(
    text: str,
    *,
    namespace: MemoryNamespace = MemoryNamespace.SESSION,
    provenance: Provenance = Provenance.USER_ASSERTED,
    salience: float = 0.5,
) -> MemoryItem:
    return MemoryItem(
        item_id=derive_id("mem", text, str(namespace)),
        namespace=namespace,
        text=text,
        provenance=provenance,
        created_at_ms=int(time.time() * 1000),
        salience=salience,
    )
