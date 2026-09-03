"""Domain events, plus the cache and evaluation shapes the cross-cutting protocols need.

Events exist so that side effects never sit on the request path. Trace export, evaluation
sampling, cache population, index feedback, and conflict events are all published and drained
after the response has been sent. A request that waits for its own telemetry has made
observability a latency cost, and the first thing anyone does about that is turn the
observability off.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "CacheEntry",
    "CacheKey",
    "DomainEvent",
    "EvalInputs",
    "EvalSample",
    "EventKind",
    "MetricResult",
]


class EventKind(StrEnum):
    """Side effects the request path publishes rather than performs."""

    TRACE_EXPORTED = "trace_exported"
    EVAL_SAMPLED = "eval_sampled"
    CACHE_POPULATE = "cache_populate"
    CACHE_INVALIDATE = "cache_invalidate"
    INDEX_FEEDBACK = "index_feedback"
    #: Drives per-adapter staleness aggregation and the retraining trigger.
    PARAMETRIC_RETRIEVAL_CONFLICT = "parametric_retrieval_conflict"
    SECURITY_EVENT = "security_event"
    #: An ACL recheck mismatch or a canary sighting. Pages a human.
    ISOLATION_ALERT = "isolation_alert"
    ABSTAINED = "abstained"
    DEGRADED = "degraded"


class DomainEvent(BaseModel):
    """One published side effect.

    Accumulated on the request state and drained to the bus after the response. Carrying them
    on the state rather than publishing inline means a request that fails midway does not emit
    a half-story: the interpreter decides what to publish once it knows how the request ended.
    """

    model_config = ConfigDict(frozen=True)

    event_id: str
    kind: EventKind
    request_id: str
    tenant_id: str
    occurred_at_ms: int
    #: Event-specific body. Deliberately untyped here: adding an event kind must not require a
    #: change to the bus, and the consumers that care about a kind know its shape.
    payload: dict[str, Any] = Field(default_factory=dict)


class CacheKey(BaseModel):
    """A cache key, constructed rather than concatenated.

    Every field is part of the key for a reason that has bitten someone: the tenant and ACL set
    because an entry must never cross a permission boundary, the config version because a config
    change alters behaviour, and the embedding version because a key that outlives a reindex
    returns results for vectors that no longer exist.
    """

    model_config = ConfigDict(frozen=True)

    tier: str
    tenant_id: str
    #: Sorted ACL hashes. Sorted so that two principals with the same permissions in a different
    #: order share an entry; unsorted, the cache would be correct and nearly useless.
    acl_discriminator: tuple[str, ...] = ()
    config_version: str
    embedding_version: str | None = None
    #: The hashed subject of the entry: a normalised query, a prompt, a plan.
    subject_hash: str = ""
    #: Anything else that changes the answer, such as an SLA tier or a strict-mode flag.
    qualifiers: tuple[tuple[str, str], ...] = ()

    def render(self) -> str:
        """The flat string form used by the backing store.

        Field order is fixed and part of the on-disk format. Reordering it invalidates every
        existing entry, which is survivable but should be a deliberate act rather than a
        refactoring side effect.
        """
        parts = [
            self.tier,
            self.tenant_id,
            ",".join(self.acl_discriminator),
            self.config_version,
            self.embedding_version or "-",
            self.subject_hash,
            ";".join(f"{k}={v}" for k, v in self.qualifiers),
        ]
        return "|".join(parts)


class CacheEntry(BaseModel):
    """A cached value plus what it took to produce.

    ``document_ids`` is what makes targeted invalidation possible. Without it, a document update
    can only be handled by flushing a tier, and a system that flushes its cache on every ingest
    has no cache.
    """

    model_config = ConfigDict(frozen=True)

    value: Any
    stored_at_ms: int
    #: Documents this entry depends on, for the reverse index used by invalidation.
    document_ids: tuple[str, ...] = ()
    adapter_ids: tuple[str, ...] = ()
    #: What it cost to compute. Turns cache hit rate into money saved rather than a percentage.
    origin_cost_usd: float = Field(default=0.0, ge=0.0)
    origin_latency_ms: int = Field(default=0, ge=0)


class EvalInputs(BaseModel):
    """What a metric needs in order to score.

    Declared rather than discovered, so a runner can batch the fetches for a whole dataset
    instead of pulling per sample. A metric that needs evidence and one that needs only the
    answer should not cost the same to run over ten thousand cases.
    """

    model_config = ConfigDict(frozen=True)

    needs_answer: bool = True
    needs_evidence: bool = False
    needs_reference: bool = False
    needs_retrieval_labels: bool = False
    needs_judge: bool = False


class EvalSample(BaseModel):
    """One case to score.

    ``recorded_state`` is what makes replay-based evaluation possible: the same graph, the same
    pinned versions, and an assertion that the decisions come out identical. Without it, an
    evaluation run measures the model and the world together and cannot separate them.
    """

    model_config = ConfigDict(frozen=True)

    sample_id: str
    query: str
    reference_answer: str | None = None
    relevant_document_ids: tuple[str, ...] = ()
    answer: str | None = None
    evidence_texts: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()
    recorded_state: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricResult(BaseModel):
    """One metric's score for one sample.

    ``judge_model`` and ``judge_version`` are recorded whenever a judge produced the score. A
    judge score whose provenance is unknown cannot be recalibrated, and an uncalibrated judge
    reporting confident nonsense at scale is a real failure mode rather than a hypothetical one.
    """

    model_config = ConfigDict(frozen=True)

    metric_id: str
    sample_id: str
    score: float
    passed: bool | None = None
    detail: str | None = None
    judge_model: str | None = None
    judge_version: str | None = None
