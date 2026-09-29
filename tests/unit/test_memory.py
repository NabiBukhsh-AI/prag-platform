"""The memory stores' own behaviour, beyond the shared contract suite."""

from __future__ import annotations

import pytest

from prag.core.errors import MemoryWriteRefused
from prag.core.models.common import MemoryNamespace, Provenance
from prag.core.models.identity import Principal
from prag.core.models.memory import MemorySelector
from prag.memory import PrincipalMemoryStore, decisions_in, entities_in
from tests.fakes.memory import make_memory_item

DAY_MS = 86_400_000
USER = Principal(tenant_id="t", user_id="u1")


class Clock:
    def __init__(self, ms: int = 1_800_000_000_000) -> None:
        self.ms = ms

    def __call__(self) -> float:
        return self.ms / 1000

    def advance(self, *, days: float = 0, hours: float = 0) -> None:
        self.ms += int(days * DAY_MS + hours * 3_600_000)


def turn(text: str, clock: Clock, *, session: str = "s1", provenance=Provenance.USER_ASSERTED):
    return make_memory_item(text, session_id=session, provenance=provenance, created_at_ms=clock.ms)


def fact(text: str, clock: Clock, *, salience: float = 0.5):
    return make_memory_item(
        text, namespace=MemoryNamespace.LONG_TERM, salience=salience, created_at_ms=clock.ms
    )


class TestSession:
    async def test_turns_expire_after_the_ttl(self) -> None:
        clock = Clock()
        store = PrincipalMemoryStore.session(ttl_hours=24, clock=clock)
        await store.write(USER, turn("the deploy window is Thursday", clock))

        clock.advance(hours=23)
        assert await store.read(USER, "deploy", 5, session_id="s1")
        clock.advance(hours=2)
        assert await store.read(USER, "deploy", 5, session_id="s1") == []

    async def test_only_the_recent_window_enters_context(self) -> None:
        """Older turns live on in the summary and in storage for audit, not in the prompt."""
        clock = Clock()
        store = PrincipalMemoryStore.session(window=3, clock=clock)
        for i in range(6):
            await store.write(USER, turn(f"turn {i} about the rollout", clock))
            clock.advance(hours=0.01)

        visible = await store.read(USER, "rollout", 10, session_id="s1")
        assert sorted(i.text for i in visible) == [f"turn {i} about the rollout" for i in (3, 4, 5)]
        assert len(store.history(USER, "s1")) == 6

    async def test_recent_turns_are_read_even_without_shared_words(self) -> None:
        """A follow-up rarely repeats the words it refers back to."""
        clock = Clock()
        store = PrincipalMemoryStore.session(clock=clock)
        await store.write(USER, turn("we are migrating the billing service", clock))
        found = await store.read(USER, "what about the other one?", 5, session_id="s1")
        assert [i.text for i in found] == ["we are migrating the billing service"]

    async def test_the_summary_keeps_decisions_and_entities_verbatim(self) -> None:
        clock = Clock()
        store = PrincipalMemoryStore.session(window=2, clock=clock)
        for text in (
            "We run the Payments API on Kubernetes.",
            "The team decided to freeze deploys on Fridays.",
            "What does the runbook say?",
            "Thanks.",
        ):
            await store.write(USER, turn(text, clock))
        answer = turn("Paging happens in 15 minutes.", clock, provenance=Provenance.MODEL_GENERATED)
        await store.write(USER, answer)

        summary = await store.summarize(USER, "s1")
        assert summary.turns_summarized == 3
        assert "The team decided to freeze deploys on Fridays." in summary.verbatim_decisions
        assert {"Payments API", "Kubernetes"} <= set(summary.verbatim_entities)
        assert store.latest_summary(USER, "s1") == summary

    async def test_model_turns_never_become_decisions(self) -> None:
        clock = Clock()
        items = [turn("We chose Postgres.", clock, provenance=Provenance.MODEL_GENERATED)]
        assert decisions_in(items) == ()
        assert entities_in(items) == ()

    async def test_forgetting_a_session_drops_its_turns_and_summary(self) -> None:
        clock = Clock()
        store = PrincipalMemoryStore.session(clock=clock)
        await store.write(USER, turn("one", clock, session="s1"))
        await store.write(USER, turn("two", clock, session="s2"))
        await store.summarize(USER, "s1")

        assert await store.forget(USER, MemorySelector(session_id="s1")) == 1
        assert store.latest_summary(USER, "s1") is None
        assert store.turn_count(USER, "s2") == 1


class TestLongTerm:
    async def test_model_generated_facts_are_refused(self) -> None:
        clock = Clock()
        store = PrincipalMemoryStore.long_term(clock=clock)
        item = make_memory_item(
            "the user seems to prefer dark mode",
            namespace=MemoryNamespace.LONG_TERM,
            provenance=Provenance.MODEL_GENERATED,
        )
        with pytest.raises(MemoryWriteRefused):
            await store.write(USER, item)

    async def test_only_relevant_facts_are_read(self) -> None:
        clock = Clock()
        store = PrincipalMemoryStore.long_term(clock=clock)
        await store.write(USER, fact("prefers answers in Urdu", clock))
        await store.write(USER, fact("works in the Karachi office", clock))

        found = await store.read(USER, "which office am I in", 5)
        assert [i.text for i in found] == ["works in the Karachi office"]

    async def test_salience_decays_with_age(self) -> None:
        clock = Clock()
        store = PrincipalMemoryStore.long_term(decay_half_life_days=30, clock=clock)
        await store.write(USER, fact("uses the staging cluster", clock, salience=0.9))
        clock.advance(days=120)  # four half-lives: 0.9 -> 0.056
        await store.write(USER, fact("uses the production cluster", clock, salience=0.3))

        found = await store.read(USER, "which cluster", 5)
        assert [i.text for i in found] == [
            "uses the production cluster",
            "uses the staging cluster",
        ]

    async def test_reading_a_fact_refreshes_it(self) -> None:
        """Use keeps a fact alive: decay runs from the last access, not only from creation."""
        clock = Clock()
        store = PrincipalMemoryStore.long_term(decay_half_life_days=30, max_items=2, clock=clock)
        await store.write(USER, fact("uses the staging cluster", clock))
        await store.write(USER, fact("prefers short answers", clock))
        clock.advance(days=60)
        await store.read(USER, "cluster", 5)  # touches only the cluster fact
        await store.write(USER, fact("works nights", clock))  # over capacity: evict one

        remaining = {i.text for i in await store.read(USER, "", 10)}
        assert "uses the staging cluster" in remaining
        assert "prefers short answers" not in remaining

    async def test_capacity_evicts_the_least_valuable_not_the_oldest(self) -> None:
        clock = Clock()
        store = PrincipalMemoryStore.long_term(max_items=2, clock=clock)
        await store.write(USER, fact("is the on-call lead", clock, salience=0.9))
        clock.advance(days=1)
        await store.write(USER, fact("likes tea", clock, salience=0.1))
        clock.advance(days=1)
        await store.write(USER, fact("works in Karachi", clock, salience=0.5))

        remaining = {i.text for i in await store.read(USER, "", 10)}
        assert remaining == {"is the on-call lead", "works in Karachi"}

    async def test_a_restated_fact_is_stored_once(self) -> None:
        clock = Clock()
        store = PrincipalMemoryStore.long_term(clock=clock)
        await store.write(USER, fact("Works in Karachi", clock, salience=0.4))
        await store.write(USER, fact("works in karachi", clock, salience=0.8))

        (only,) = await store.read(USER, "karachi", 5)
        assert only.salience == 0.8

    async def test_a_session_item_is_refused(self) -> None:
        store = PrincipalMemoryStore.long_term()
        with pytest.raises(MemoryWriteRefused):
            await store.write(USER, make_memory_item("x"))
