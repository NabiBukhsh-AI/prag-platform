"""Shared value objects and enumerations.

Anything here is depended on by more than one of the other model modules. Keeping it separate
is what stops ``identity`` and ``retrieval`` from importing each other for the sake of one enum.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, Field

from prag.core.errors import DeadlineExceeded

__all__ = [
    "CollectionInfo",
    "Deadline",
    "EmbeddingPurpose",
    "GuardrailPhase",
    "HealthState",
    "HealthStatus",
    "MemoryNamespace",
    "Provenance",
    "SlaTier",
    "SourceCapabilities",
    "VolatilityClass",
]


class SlaTier(StrEnum):
    """Per-principal service tier. Selects the routing utility weights and the rerank tier."""

    INTERACTIVE = "interactive"
    STANDARD = "standard"
    HIGH_STAKES = "high_stakes"
    BATCH = "batch"


class VolatilityClass(StrEnum):
    """How fast a piece of knowledge goes stale.

    Drives cache TTL selection and parametric eligibility: nothing faster than ``SLOW`` has a
    half-life long enough to survive a retrain cadence, so nothing faster can be parameterized.
    """

    STATIC = "static"
    SLOW = "slow"
    FAST = "fast"
    REALTIME = "realtime"


class EmbeddingPurpose(StrEnum):
    """Query or document.

    Asymmetric embedding models produce different vectors for the same text depending on which
    side of the retrieval it sits on. Making the caller state its purpose means a provider that
    ignores the distinction and one that honours it are interchangeable at the call site.
    """

    QUERY = "query"
    DOCUMENT = "document"


class GuardrailPhase(StrEnum):
    """When in the request a guardrail runs."""

    INPUT = "input"
    RETRIEVAL = "retrieval"
    PRE_GENERATION = "pre_generation"
    OUTPUT = "output"


class MemoryNamespace(StrEnum):
    """Which memory tier an item belongs to.

    Namespaces are a correctness boundary, not a label. A citation may never resolve across
    them: an answer cannot cite a session assertion as though it were a retrieved document.
    """

    WORKING = "working"
    SESSION = "session"
    LONG_TERM = "long_term"


class Provenance(StrEnum):
    """Where a memory item or claim came from.

    ``MODEL_GENERATED`` is the one that matters: long-term memory must refuse it. Letting a
    model's own output persist as a user fact is how a system starts confidently believing
    things nobody ever told it.
    """

    USER_ASSERTED = "user_asserted"
    CONFIRMED_STRUCTURED = "confirmed_structured"
    RETRIEVED_DOCUMENT = "retrieved_document"
    PARAMETRIC = "parametric"
    MODEL_GENERATED = "model_generated"


class HealthState(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class HealthStatus(BaseModel):
    """A dependency's self-report, as returned by every source and provider."""

    state: HealthState
    checked_at_ms: int
    latency_ms: int | None = None
    detail: str | None = None

    @property
    def usable(self) -> bool:
        """Degraded dependencies are still worth calling; unavailable ones are not."""
        return self.state is not HealthState.UNAVAILABLE


class SourceCapabilities(BaseModel):
    """What a knowledge source can actually do.

    The retrieval planner reads this instead of hardcoding assumptions per source id, which is
    what allows a new source to be registered in config without touching the planner.
    """

    supports_filters: bool
    supports_vectors: bool
    supports_text: bool
    supports_traversal: bool = False
    max_top_k: int = Field(gt=0)


class CollectionInfo(BaseModel):
    """Vector collection metadata.

    ``embedding_model`` and ``embedding_version`` are here because a collection whose vectors
    were written by a different model than the one embedding the query returns plausible
    nonsense rather than an error. Migration compares these rather than trusting a naming
    convention.
    """

    name: str
    vector_count: int
    dimensions: int
    embedding_model: str
    embedding_version: str


_MONOTONIC_TO_MS: Final = 1_000.0


@dataclass(frozen=True, slots=True)
class Deadline:
    """An absolute point in time by which an operation must have finished.

    Every external call takes one. Passing a *duration* instead would be a bug waiting to
    happen: durations do not compose, so five calls each given "200 ms" can spend a second
    between them, and no individual call did anything wrong.

    Timing is monotonic. Wall-clock time can step backwards on an NTP correction, and a deadline
    that goes backwards silently grants a request more time than it was budgeted.

    Not a Pydantic model on purpose. A deadline is meaningful only within the process that
    created it — serialising one and honouring it elsewhere would compare a monotonic reading
    against a different machine's clock.
    """

    #: ``time.monotonic()`` reading at which this deadline expires.
    expires_at: float
    #: What the deadline was created for. Appears in the error when it is breached, which turns
    #: "deadline exceeded" into "which deadline, set by whom".
    label: str = "unlabelled"
    _created_at: float = field(default_factory=time.monotonic, compare=False)

    @classmethod
    def in_ms(cls, ms: float, *, label: str = "unlabelled") -> Deadline:
        """A deadline ``ms`` milliseconds from now."""
        if ms < 0:
            raise ValueError(f"deadline cannot be negative, got {ms}")
        return cls(expires_at=time.monotonic() + ms / _MONOTONIC_TO_MS, label=label)

    @property
    def remaining_ms(self) -> float:
        """Milliseconds left, floored at zero rather than going negative."""
        return max(0.0, (self.expires_at - time.monotonic()) * _MONOTONIC_TO_MS)

    @property
    def elapsed_ms(self) -> float:
        """Milliseconds since this deadline was created."""
        return (time.monotonic() - self._created_at) * _MONOTONIC_TO_MS

    @property
    def expired(self) -> bool:
        return time.monotonic() >= self.expires_at

    def raise_if_expired(self) -> None:
        """Fail fast at a checkpoint rather than starting work that cannot finish.

        Worth calling before any expensive step, not only after. Discovering at the end of a
        120 ms rerank that the budget ran out 100 ms ago has already spent the money.
        """
        if self.expired:
            raise DeadlineExceeded(
                "deadline expired",
                label=self.label,
                overrun_ms=round((time.monotonic() - self.expires_at) * _MONOTONIC_TO_MS, 3),
            )

    def narrowed_to(self, ms: float, *, label: str | None = None) -> Deadline:
        """A sub-deadline no later than this one.

        Sub-deadlines can only tighten. A node handed 200 ms cannot grant its own callee 500 ms,
        because the request budget does not care how optimistic the node was.
        """
        candidate = time.monotonic() + max(0.0, ms) / _MONOTONIC_TO_MS
        return Deadline(
            expires_at=min(candidate, self.expires_at),
            label=label or self.label,
        )

    def share(self, fraction: float, *, label: str | None = None) -> Deadline:
        """A sub-deadline covering ``fraction`` of the remaining time.

        Used where a stage's cost is proportional to what is left rather than fixed, such as
        splitting the remaining budget across a plan's parallel legs.
        """
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        return self.narrowed_to(self.remaining_ms * fraction, label=label)
