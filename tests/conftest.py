"""Fixtures shared across every suite."""

from __future__ import annotations

import pytest

from prag.core.ids import new_request_id, new_trace_id
from prag.core.models.common import Deadline, SlaTier
from prag.core.models.identity import Budget, Principal, TenantPolicy, UtilityWeights

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


@pytest.fixture
def principal() -> Principal:
    """A standard-tier caller with one non-public ACL entry.

    Deliberately not public-only. A principal whose every permission is public passes ACL tests
    that a real caller would fail, because the filter is never actually exercised.
    """
    return Principal(
        tenant_id="tenant-a",
        user_id="user-1",
        groups=("engineering",),
        scopes=("answer:read",),
        acl_hashes=("acl-engineering",),
        sla_tier=SlaTier.STANDARD,
    )


@pytest.fixture
def other_principal() -> Principal:
    """A caller in a different tenant, for isolation assertions."""
    return Principal(
        tenant_id="tenant-b",
        user_id="user-2",
        acl_hashes=("acl-finance",),
    )


@pytest.fixture
def policy() -> TenantPolicy:
    return TenantPolicy(
        tenant_id="tenant-a",
        config_version="test-1",
        utility_weights=UtilityWeights(quality=0.60, latency=0.20, cost=0.20),
    )


@pytest.fixture
def strict_policy() -> TenantPolicy:
    """Strict mode: irreconcilable conflicts and stale evidence become abstentions."""
    return TenantPolicy(
        tenant_id="tenant-a",
        config_version="test-1",
        utility_weights=UtilityWeights(quality=0.85, latency=0.05, cost=0.10),
        strict_mode=True,
    )


# ---------------------------------------------------------------------------
# Budget and time
# ---------------------------------------------------------------------------


@pytest.fixture
def budget() -> Budget:
    return Budget(
        wall_ms_total=2000,
        wall_ms_remaining=2000,
        usd_total=0.05,
        max_tokens_in=8000,
        max_tokens_out=1024,
    )


@pytest.fixture
def deadline() -> Deadline:
    """A deadline generous enough that tests do not fail on a slow CI runner.

    Tests that care about expiry construct their own rather than racing this one.
    """
    return Deadline.in_ms(5_000, label="test")


@pytest.fixture
def expired_deadline() -> Deadline:
    return Deadline.in_ms(0, label="test-expired")


@pytest.fixture
def request_id() -> str:
    return new_request_id()


@pytest.fixture
def trace_id() -> str:
    return new_trace_id()
