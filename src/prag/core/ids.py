"""Identifier generation.

Two kinds of identifier, and the distinction matters for replay:

**Random, time-sortable identifiers** for things that happen once: requests, traces, plans,
bundles. These use a millisecond timestamp prefix followed by randomness, so lexicographic sort
equals chronological sort. That property is worth more than it sounds — it makes index locality
good in Postgres, and it makes a sorted log listing read in causal order without a join.

**Derived identifiers** for things that must be stable across runs: a candidate's identity, an
evidence group's identity, a cache key's identity. These are content hashes. Replaying a
recorded request against pinned versions has to produce byte-identical decisions, and it cannot
do that if the identity of a retrieved chunk is freshly random on every run.

Trace and span identifiers follow W3C Trace Context so they interoperate with OpenTelemetry
without translation.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import time
from typing import Final

__all__ = [
    "derive_id",
    "new_bundle_id",
    "new_id",
    "new_plan_id",
    "new_request_id",
    "new_span_id",
    "new_trace_id",
    "short_hash",
]

# Crockford base32: no I, L, O, or U, so identifiers survive being read aloud and transcribed.
_ALPHABET: Final = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_RANDOM_CHARS: Final = 16
_TIME_CHARS: Final = 10


def _encode(value: int, length: int) -> str:
    out = ["0"] * length
    for i in range(length - 1, -1, -1):
        out[i] = _ALPHABET[value & 0x1F]
        value >>= 5
    return "".join(out)


def _sortable_suffix() -> str:
    """A time-ordered, collision-resistant token.

    The layout is ULID-shaped: 48 bits of millisecond timestamp, then 80 bits of randomness.
    Within a single millisecond ordering is arbitrary, which is fine — nothing in the platform
    depends on intra-millisecond ordering, and everything that reads these depends on the
    coarse-grained sort.
    """
    millis = time.time_ns() // 1_000_000
    return _encode(millis, _TIME_CHARS) + _encode(
        int.from_bytes(secrets.token_bytes(10), "big"), _RANDOM_CHARS
    )


def new_id(prefix: str) -> str:
    """A prefixed, time-sortable identifier, for example ``req_01JZ...``.

    The prefix is not decoration. An identifier that says what it identifies turns a
    copy-pasted string in an incident channel into something you can act on without asking
    which table to look in.
    """
    if not prefix or not prefix.isidentifier():
        raise ValueError(f"prefix must be a valid identifier, got {prefix!r}")
    return f"{prefix}_{_sortable_suffix()}"


def new_request_id() -> str:
    """Identity of one inbound request. Appears in every span, log line, and envelope."""
    return new_id("req")


def new_plan_id() -> str:
    """Identity of one retrieval plan, so a plan can be correlated with what it returned."""
    return new_id("plan")


def new_bundle_id() -> str:
    """Identity of one assembled context bundle."""
    return new_id("ctx")


def new_trace_id() -> str:
    """A W3C Trace Context trace-id: 32 lowercase hex characters, never all zero."""
    return os.urandom(16).hex()


def new_span_id() -> str:
    """A W3C Trace Context span-id: 16 lowercase hex characters, never all zero."""
    return os.urandom(8).hex()


def derive_id(namespace: str, *parts: str | int | None) -> str:
    """A deterministic identifier derived from content.

    Same inputs, same output, forever — which is what makes recorded-state replay meaningful.
    ``namespace`` keeps derivations from different call sites in separate spaces, so a chunk id
    and a cache key built from the same document never collide.

    ``None`` parts are preserved as a distinct value rather than skipped, so
    ``derive_id("x", "a", None)`` and ``derive_id("x", "a")`` differ. Dropping them would make
    an absent optional field silently equivalent to its omission, and that is exactly the kind
    of collision that shows up as a cache serving the wrong entry.
    """
    digest = hashlib.blake2b(namespace.encode("utf-8"), digest_size=16)
    for part in parts:
        # Length-prefixed so that ("ab", "c") cannot hash the same as ("a", "bc").
        encoded = b"\x00" if part is None else str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return f"{namespace}_{digest.hexdigest()}"


def short_hash(text: str, *, length: int = 16) -> str:
    """A short content hash, for prompt hashes and dedup keys.

    Not a security boundary: it is a collision-resistant label, sized for readability in a trace
    viewer. Anything that needs cryptographic strength should say so and use full-width digests.
    """
    if not 4 <= length <= 64:
        raise ValueError(f"length must be between 4 and 64, got {length}")
    return hashlib.blake2b(text.encode("utf-8"), digest_size=32).hexdigest()[:length]
