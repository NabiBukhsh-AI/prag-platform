"""Conformance suite for ``MemoryStore``.

The write-gating test is the one that matters. A long-term store which accepts model-generated
content passes every read test and still corrupts itself over weeks.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from prag.core.errors import PragError
from prag.core.models.common import MemoryNamespace, Provenance
from prag.core.models.identity import Principal
from prag.core.models.memory import MemorySelector
from prag.core.protocols import MemoryStore
from tests.fakes.memory import make_memory_item

pytestmark = pytest.mark.contract


def test_satisfies_protocol(session_memory: Any) -> None:
    assert isinstance(session_memory, MemoryStore)
    assert session_memory.namespace in tuple(MemoryNamespace)


async def test_write_then_read(session_memory: Any, principal: Principal) -> None:
    item = make_memory_item("the deploy window is Thursday")
    await session_memory.write(principal, item)

    found = await session_memory.read(principal, "deploy", limit=10)
    assert [i.item_id for i in found] == [item.item_id]


async def test_memory_does_not_cross_principals(
    session_memory: Any, principal: Principal, other_principal: Principal
) -> None:
    """One caller's memory is invisible to another.

    Keyed by tenant *and* user: tenant alone would let two users in one tenant read each
    other's memory, which a single-tenant test would never surface.
    """
    await session_memory.write(principal, make_memory_item("my private note"))
    assert await session_memory.read(other_principal, "private", limit=10) == []


async def test_read_respects_limit(session_memory: Any, principal: Principal) -> None:
    for i in range(10):
        await session_memory.write(principal, make_memory_item(f"note {i} shared"))
    assert len(await session_memory.read(principal, "shared", limit=3)) <= 3


async def test_long_term_rejects_model_generated(
    long_term_memory: Any, principal: Principal
) -> None:
    """The rule that keeps the system from believing its own output.

    Enforced at the store, not trusted from the caller. A model's answer persisted as a user
    fact outlives the session that produced it and is indistinguishable, later, from something
    the user actually said.
    """
    item = make_memory_item(
        "the user probably prefers dark mode",
        namespace=MemoryNamespace.LONG_TERM,
        provenance=Provenance.MODEL_GENERATED,
    )
    assert item.persistable is False

    with pytest.raises(PragError):
        await long_term_memory.write(principal, item)


@pytest.mark.parametrize(
    "provenance",
    [Provenance.USER_ASSERTED, Provenance.CONFIRMED_STRUCTURED],
)
async def test_long_term_accepts_permitted_provenance(
    long_term_memory: Any, principal: Principal, provenance: Provenance
) -> None:
    item = make_memory_item(
        "works in the Karachi office",
        namespace=MemoryNamespace.LONG_TERM,
        provenance=provenance,
    )
    await long_term_memory.write(principal, item)
    assert await long_term_memory.read(principal, "Karachi", limit=5)


async def test_namespace_mismatch_is_refused(session_memory: Any, principal: Principal) -> None:
    """A store accepts only its own namespace.

    Namespaces are a citation boundary: an answer must not cite a session assertion as though
    it were a retrieved document. A store that quietly accepts foreign items erases the
    boundary before the citation validator ever sees it.
    """
    wrong = make_memory_item("x", namespace=MemoryNamespace.LONG_TERM)
    with pytest.raises(PragError):
        await session_memory.write(principal, wrong)


async def test_summarize_keeps_decisions_verbatim(
    session_memory: Any, principal: Principal
) -> None:
    await session_memory.write(principal, make_memory_item("we chose Postgres"))
    summary = await session_memory.summarize(principal, "session-1")

    assert summary.session_id == "session-1"
    assert summary.summary_hash
    assert "we chose Postgres" in summary.verbatim_decisions


async def test_forget_returns_a_count(session_memory: Any, principal: Principal) -> None:
    """Right-to-erasure needs an auditable record of what was actually erased."""
    item = make_memory_item("forget me")
    await session_memory.write(principal, item)

    removed = await session_memory.forget(principal, MemorySelector(item_ids=(item.item_id,)))
    assert removed == 1
    assert await session_memory.read(principal, "forget", limit=5) == []


async def test_empty_selector_forgets_nothing(session_memory: Any, principal: Principal) -> None:
    """An unset selector must not be read as "everything".

    A forget call that wipes a principal's entire memory because a field was left unset is not
    a failure mode worth leaving open.
    """
    await session_memory.write(principal, make_memory_item("keep me"))
    assert await session_memory.forget(principal, MemorySelector()) == 0
    assert await session_memory.read(principal, "keep", limit=5)


async def test_forget_by_age(session_memory: Any, principal: Principal) -> None:
    await session_memory.write(principal, make_memory_item("recent note"))
    future = int(time.time() * 1000) + 60_000
    assert await session_memory.forget(principal, MemorySelector(older_than_ms=future)) == 1
