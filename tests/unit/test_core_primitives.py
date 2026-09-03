"""Deadlines, identifiers, errors, and the DI container."""

from __future__ import annotations

import time

import pytest

from prag.core.di import Container, Scope
from prag.core.errors import (
    AbstentionRequired,
    AclRecheckMismatch,
    ConfigurationError,
    DeadlineExceeded,
    EligibilityBlocked,
    IsolationViolation,
    PragError,
    RetrievalError,
    Severity,
    SourceUnavailable,
)
from prag.core.ids import (
    derive_id,
    new_id,
    new_request_id,
    new_span_id,
    new_trace_id,
    short_hash,
)
from prag.core.models.common import Deadline


class TestDeadline:
    def test_tightening_only(self) -> None:
        """A node handed 200 ms cannot grant its callee 500 ms."""
        parent = Deadline.in_ms(200, label="parent")
        assert parent.narrowed_to(5_000).expires_at == parent.expires_at
        assert parent.narrowed_to(50).expires_at < parent.expires_at

    def test_share_of_remaining(self) -> None:
        parent = Deadline.in_ms(400, label="plan")
        leg = parent.share(0.5, label="leg")
        assert leg.expires_at < parent.expires_at
        assert leg.label == "leg"

    def test_sub_deadline_inherits_the_label_by_default(self) -> None:
        assert Deadline.in_ms(100, label="root").narrowed_to(50).label == "root"

    def test_expiry_raises_with_context(self) -> None:
        """ "Deadline exceeded" is not actionable; which deadline, and by how much, is."""
        with pytest.raises(DeadlineExceeded) as excinfo:
            Deadline.in_ms(0, label="rerank").raise_if_expired()

        assert excinfo.value.context["label"] == "rerank"
        assert "overrun_ms" in excinfo.value.context

    def test_live_deadline_does_not_raise(self) -> None:
        Deadline.in_ms(5_000, label="fine").raise_if_expired()

    def test_remaining_floors_at_zero(self) -> None:
        assert Deadline.in_ms(0).remaining_ms == 0.0

    @pytest.mark.parametrize("bad", [-1, -0.001])
    def test_negative_is_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="negative"):
            Deadline.in_ms(bad)

    @pytest.mark.parametrize("bad", [0.0, -0.5, 1.5])
    def test_invalid_share_is_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="fraction"):
            Deadline.in_ms(100).share(bad)


class TestIdentifiers:
    def test_request_ids_are_prefixed_and_unique(self) -> None:
        ids = {new_request_id() for _ in range(1000)}
        assert len(ids) == 1000
        assert all(i.startswith("req_") for i in ids)

    def test_ids_sort_chronologically(self) -> None:
        """Lexicographic order equals creation order, so a sorted log reads causally.

        At millisecond granularity: the timestamp prefix orders ids across milliseconds, and
        the random suffix tie-breaks arbitrarily within one. Ids minted in the same millisecond
        are concurrent for every purpose the platform has, so nothing depends on their relative
        order — but anything separated in time must sort correctly.
        """
        ordered = []
        for _ in range(20):
            ordered.append(new_id("evt"))
            time.sleep(0.002)
        assert ordered == sorted(ordered)

    def test_ids_minted_together_remain_unique(self) -> None:
        """Ordering within a millisecond is arbitrary; uniqueness is not."""
        burst = [new_id("evt") for _ in range(5000)]
        assert len(set(burst)) == 5000

    def test_trace_and_span_ids_are_w3c_shaped(self) -> None:
        trace, span = new_trace_id(), new_span_id()
        assert len(trace) == 32
        assert len(span) == 16
        assert int(trace, 16) >= 0
        assert int(span, 16) >= 0

    def test_invalid_prefix_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="identifier"):
            new_id("not a prefix")

    def test_derived_ids_are_stable(self) -> None:
        """Replay asserts identical decisions; it cannot if identity is freshly random."""
        assert derive_id("cand", "doc-1", 3) == derive_id("cand", "doc-1", 3)

    def test_namespaces_do_not_collide(self) -> None:
        assert derive_id("cand", "x") != derive_id("group", "x")

    def test_parts_are_length_prefixed(self) -> None:
        """("ab", "c") must not hash the same as ("a", "bc")."""
        assert derive_id("n", "ab", "c") != derive_id("n", "a", "bc")

    def test_none_is_distinct_from_absence(self) -> None:
        """An unset optional must not be equivalent to omitting it.

        That equivalence is exactly the kind of collision that shows up later as a cache
        serving the wrong entry.
        """
        assert derive_id("n", "a", None) != derive_id("n", "a")

    def test_short_hash_is_stable_and_sized(self) -> None:
        assert short_hash("abc") == short_hash("abc")
        assert len(short_hash("abc", length=8)) == 8

    @pytest.mark.parametrize("bad", [0, 3, 65])
    def test_short_hash_rejects_bad_length(self, bad: int) -> None:
        with pytest.raises(ValueError, match="length"):
            short_hash("abc", length=bad)


class TestErrors:
    def test_reason_codes_are_stable(self) -> None:
        """These appear in dashboards and client responses; they are part of the contract."""
        assert SourceUnavailable("x").reason_code == "source_unavailable"
        assert DeadlineExceeded("x").reason_code == "deadline_exceeded"

    def test_hierarchy_allows_catching_by_category(self) -> None:
        assert isinstance(SourceUnavailable("x"), RetrievalError)
        assert isinstance(SourceUnavailable("x"), PragError)

    def test_acl_mismatch_is_an_isolation_violation(self) -> None:
        """It must be catchable as isolation, not merely as retrieval.

        A recheck mismatch means one of the five isolation layers failed, and it has to be
        handled by the code that pages a human rather than by a retry.
        """
        error = AclRecheckMismatch("canary sighted")
        assert isinstance(error, IsolationViolation)
        assert error.severity is Severity.CRITICAL
        assert not error.retryable

    def test_context_renders_into_the_message(self) -> None:
        error = SourceUnavailable("unreachable", source_id="vector.primary")
        assert "vector.primary" in str(error)
        assert error.as_dict()["context"]["source_id"] == "vector.primary"

    def test_eligibility_carries_every_reason(self) -> None:
        """Fixing one blocker only to hit the next teaches nothing about eligibility."""
        error = EligibilityBlocked("blocked", reasons=["acl_narrow", "half_life_short"])
        assert len(error.reasons) == 2

    def test_abstention_is_informational(self) -> None:
        """Declining to answer is a success state, not a failure."""
        error = AbstentionRequired(
            "no evidence for a private query",
            abstention_code="private_query_no_evidence",
            suggested_action="request access to the source",
        )
        assert error.severity is Severity.INFO
        assert error.suggested_action


class TestContainer:
    def test_singleton_is_shared_and_transient_is_not(self) -> None:
        container = Container()
        container.register(list, lambda _: [], scope=Scope.SINGLETON)
        container.register(dict, lambda _: {}, scope=Scope.TRANSIENT)

        assert container.resolve(list) is container.resolve(list)
        assert container.resolve(dict) is not container.resolve(dict)

    def test_factories_resolve_their_own_dependencies(self) -> None:
        """So registration order does not matter: nothing is built until first resolve."""

        class Repo:
            def __init__(self, dsn: str) -> None:
                self.dsn = dsn

        container = Container()
        container.register(Repo, lambda c: Repo(c.resolve(str)))
        container.register_instance(str, "postgres://local")

        assert container.resolve(Repo).dsn == "postgres://local"

    def test_missing_registration_fails_loudly(self) -> None:
        """A missing dependency is a wiring bug, not an optional feature."""
        with pytest.raises(ConfigurationError) as excinfo:
            Container().resolve(list)
        assert "no implementation registered" in excinfo.value.detail

    def test_duplicate_registration_is_refused(self) -> None:
        container = Container()
        container.register(list, lambda _: [])
        with pytest.raises(ConfigurationError, match="duplicate"):
            container.register(list, lambda _: [])

    def test_cycles_are_named(self) -> None:
        """Otherwise a cycle is a recursion error thousands of frames deep, naming nothing."""

        class A: ...

        class B: ...

        container = Container()
        container.register(A, lambda c: c.resolve(B))
        container.register(B, lambda c: c.resolve(A))

        with pytest.raises(ConfigurationError) as excinfo:
            container.resolve(A)
        assert excinfo.value.context["cycle"] == "A -> B -> A"

    def test_override_restores_the_previous_binding(self) -> None:
        """An override that leaks produces a failure reproducible only in suite order."""
        container = Container()
        container.register_instance(str, "real")

        with container.override(str, "fake"):
            assert container.resolve(str) == "fake"
        assert container.resolve(str) == "real"

    def test_override_of_an_unregistered_protocol_is_removed_after(self) -> None:
        container = Container()
        with container.override(str, "temporary"):
            assert container.resolve(str) == "temporary"
        assert not container.has(str)

    def test_validate_reports_every_failure_at_once(self) -> None:
        """Startup should say what is broken, not the first thing it noticed."""

        def explode(_: Container) -> str:
            raise RuntimeError("no credentials")

        container = Container()
        container.register(str, explode)
        container.register(list, explode)

        with pytest.raises(ConfigurationError) as excinfo:
            container.validate()
        assert len(excinfo.value.context["failures"]) == 2

    def test_validate_skips_transients(self) -> None:
        """They may legitimately need per-request arguments that do not exist at startup."""
        container = Container()
        container.register(dict, lambda _: {}, scope=Scope.TRANSIENT)
        container.validate()
