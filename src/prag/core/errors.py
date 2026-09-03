"""The typed error hierarchy.

Every failure path in the platform raises one of these. The API layer maps them to responses in
exactly one place, so adding a failure mode never means touching transport code.

Two things deliberately absent:

**Abstention is not an error.** Declining to answer because knowledge confidence sits below the
tenant floor is a success state, reported as such in metrics. It travels in the response
envelope as an ``Abstention``, never as a raised exception. Modelling it as an error would put
it on the same path as a provider outage and make the abstention rate impossible to read.

**Degradation is not an error.** Skipping the reranker because the budget controller said so is
normal operation at degradation level 1. It is logged, and it shows up in evaluation as
budget-driven quality loss, but nothing raises.

Every error carries a stable ``reason_code``. Those codes appear in traces, metrics labels, and
client responses, so they are part of the platform's contract: rename one and you break a
dashboard.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AbstentionRequired",
    "AclRecheckMismatch",
    "AdapterLoadError",
    "BudgetExhausted",
    "ConfigurationError",
    "ContextError",
    "ContextOverflow",
    "ContextValidationFailed",
    "DeadlineExceeded",
    "EligibilityBlocked",
    "GenerationError",
    "GroundingFailed",
    "GuardrailBlocked",
    "IsolationViolation",
    "ParametricError",
    "PragError",
    "ProviderUnavailable",
    "RetrievalError",
    "RetrievalTotalFailure",
    "SchemaValidationFailed",
    "Severity",
    "SourceUnavailable",
    "StorageError",
    "ToolProvenanceDenied",
]

from enum import StrEnum


class Severity(StrEnum):
    """How much a failure matters, independent of which subsystem raised it."""

    #: Expected, handled, visible only in metrics.
    INFO = "info"
    #: Degraded the answer. The client is told.
    WARNING = "warning"
    #: The request failed. Recoverable by retry or rephrase.
    ERROR = "error"
    #: Pages a human. Isolation and injection failures live here.
    CRITICAL = "critical"


class PragError(Exception):
    """Base class for every failure the platform raises deliberately.

    Subclasses set ``reason_code`` and ``severity`` as class attributes. Instances carry
    structured ``context`` that is safe to put in a log line, meaning it holds identifiers and
    measurements, never query text, evidence text, or credentials. Redaction happens at the
    logging boundary, but the discipline starts here: if it should not appear in a log, do not
    attach it to an error.
    """

    reason_code: str = "internal_error"
    severity: Severity = Severity.ERROR

    #: Whether retrying the identical request could plausibly succeed. Consulted by the retry
    #: policy engine; a ``False`` here means the policy will not burn budget on a second attempt.
    retryable: bool = False

    def __init__(self, detail: str, /, **context: Any) -> None:
        super().__init__(detail)
        self.detail = detail
        self.context: dict[str, Any] = context

    def __str__(self) -> str:
        if not self.context:
            return self.detail
        rendered = ", ".join(f"{k}={v!r}" for k, v in sorted(self.context.items()))
        return f"{self.detail} ({rendered})"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(reason_code={self.reason_code!r}, detail={self.detail!r})"

    def as_dict(self) -> dict[str, Any]:
        """Structured form for logs, spans, and the API error mapper."""
        return {
            "reason_code": self.reason_code,
            "severity": str(self.severity),
            "detail": self.detail,
            "retryable": self.retryable,
            "context": dict(self.context),
        }


# ---------------------------------------------------------------------------
# Configuration and startup
# ---------------------------------------------------------------------------


class ConfigurationError(PragError):
    """Invalid or internally inconsistent configuration.

    Raised at startup, never mid-request. An invalid config must fail the process rather than
    surface at 3 a.m. on a code path nobody exercised.
    """

    reason_code = "configuration_invalid"
    severity = Severity.CRITICAL


# ---------------------------------------------------------------------------
# Budget and time
# ---------------------------------------------------------------------------


class DeadlineExceeded(PragError):
    """A node or external call ran past its deadline.

    Deadlines are derived from the remaining request budget rather than being fixed per call,
    which is what stops per-node timeouts from summing to more than the request has.
    """

    reason_code = "deadline_exceeded"
    severity = Severity.WARNING
    retryable = False


class BudgetExhausted(PragError):
    """The request ran out of wall-clock, token, or cost budget.

    Reaching this means the degradation ladder was walked to its end without finding a viable
    cheaper path. It is the terminal state of budget enforcement, not its first response.
    """

    reason_code = "budget_exceeded"
    severity = Severity.WARNING


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


class GuardrailBlocked(PragError):
    """A guardrail returned BLOCK and no safe partial answer exists."""

    reason_code = "guardrail_blocked"
    severity = Severity.ERROR

    def __init__(self, detail: str, /, *, guardrail: str, phase: str, **context: Any) -> None:
        super().__init__(detail, guardrail=guardrail, phase=phase, **context)
        self.guardrail = guardrail
        self.phase = phase


class ToolProvenanceDenied(PragError):
    """A tool call was rejected because its authorising content came from retrieved evidence.

    This is the layer that holds when the model has already been fooled by a document-borne
    injection, so a rise in this counter is a signal worth alerting on even though each
    individual denial is the system working correctly.
    """

    reason_code = "tool_provenance_denied"
    severity = Severity.CRITICAL


class IsolationViolation(PragError):
    """Cross-tenant leakage was detected: an ACL recheck mismatch, or a canary sighting.

    This fails the request hard and pages a human. There is no degraded path, because a request
    that may have crossed a tenant boundary cannot be made safe after the fact. Of the risks in
    the specification this is the one that cannot be walked back.
    """

    reason_code = "isolation_violation"
    severity = Severity.CRITICAL


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class RetrievalError(PragError):
    """Base for retrieval failures."""

    reason_code = "retrieval_failed"
    severity = Severity.ERROR
    retryable = True


class SourceUnavailable(RetrievalError):
    """A single knowledge source is unreachable or its circuit breaker is open.

    Whether this fails the request depends on the leg: an optional leg dropping produces a
    coverage warning, a required leg dropping is a plan failure.
    """

    reason_code = "source_unavailable"
    severity = Severity.WARNING


class AclRecheckMismatch(IsolationViolation):
    """A candidate survived source-level filtering that the independent recheck rejected.

    The recheck exists precisely because it does not trust the source's filter. Reaching here
    means one of the five isolation layers failed, which is an audit of the whole filter path.
    """

    reason_code = "acl_recheck_mismatch"


class RetrievalTotalFailure(RetrievalError):
    """Every required leg failed.

    The caller decides what to do: answer parametrically with explicit marking that the
    documents could not be reached, or abstain if the query targets private data.
    """

    reason_code = "retrieval_total_failure"
    severity = Severity.CRITICAL
    retryable = False


# ---------------------------------------------------------------------------
# Parametric tier
# ---------------------------------------------------------------------------


class ParametricError(PragError):
    """Base for parametric-tier failures."""

    reason_code = "parametric_failed"
    severity = Severity.WARNING


class AdapterLoadError(ParametricError):
    """An adapter could not be loaded: object store error, or checksum mismatch.

    Never fatal. The request proceeds without the adapter, downgraded to non-parametric, and the
    degradation is logged. A checksum mismatch additionally means the artifact needs re-uploading
    from the training output.
    """

    reason_code = "adapter_load_failed"
    retryable = True


class EligibilityBlocked(ParametricError):
    """Knowledge was proposed for parameterization and the eligibility gate refused.

    Carries *every* blocking reason rather than the first, because a curator fixing one blocker
    only to hit the next learns nothing about whether the knowledge is fundamentally ineligible.
    """

    reason_code = "parametric_ineligible"
    severity = Severity.INFO

    def __init__(self, detail: str, /, *, reasons: list[str], **context: Any) -> None:
        super().__init__(detail, reasons=reasons, **context)
        self.reasons = reasons


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------


class ContextError(PragError):
    """Base for context-assembly failures."""

    reason_code = "context_failed"


class ContextOverflow(ContextError):
    """Packing could not meet the coverage floor within the token budget.

    Raised only after the compression ladder, window escalation, and map-reduce have all been
    exhausted. Ordinarily this surfaces as a coverage warning instead.
    """

    reason_code = "context_overflow"
    severity = Severity.WARNING


class ContextValidationFailed(ContextError):
    """Assembled context failed validation and re-retrieval did not fix it."""

    reason_code = "context_validation_failed"
    severity = Severity.WARNING

    def __init__(self, detail: str, /, *, failure_reasons: list[str], **context: Any) -> None:
        super().__init__(detail, failure_reasons=failure_reasons, **context)
        self.failure_reasons = failure_reasons


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


class GenerationError(PragError):
    """Base for generation failures."""

    reason_code = "generation_failed"
    retryable = True


class ProviderUnavailable(GenerationError):
    """A model provider errored, timed out, or its breaker is open.

    Retry is permitted only before the first token has been yielded. Once any chunk has reached
    the client, a retry would duplicate output, so the stream fails instead.
    """

    reason_code = "provider_unavailable"


class SchemaValidationFailed(GenerationError):
    """Generated output does not satisfy its schema, after one constrained repair attempt."""

    reason_code = "schema_invalid"
    retryable = False


class GroundingFailed(GenerationError):
    """Claims in the answer are not entailed by the evidence they cite.

    The mitigation is to strip the claim or regenerate once with a tightened prompt. Raising
    means both were tried.
    """

    reason_code = "grounding_failed"
    severity = Severity.ERROR
    retryable = False


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


class StorageError(PragError):
    """A persistence backend failed.

    Raised inside ``storage`` and translated by callers into a domain-level failure. Business
    logic never sees a driver exception, which is what keeps SQL out of every other package.
    """

    reason_code = "storage_failed"
    retryable = True


# ---------------------------------------------------------------------------
# Control flow
# ---------------------------------------------------------------------------


class AbstentionRequired(PragError):
    """Signals the graph to route to the abstention path.

    This is control flow, not a failure, and the one case where an exception carries a success
    outcome. It exists so that a deeply nested check can redirect the request without every
    intervening layer having to thread an optional abstention back up by hand. The graph
    interpreter catches it and converts it into an ``Abstention`` in the envelope; it must never
    escape to the API layer as an error.
    """

    reason_code = "abstained"
    severity = Severity.INFO

    def __init__(
        self,
        detail: str,
        /,
        *,
        abstention_code: str,
        suggested_action: str | None = None,
        **context: Any,
    ) -> None:
        super().__init__(
            detail,
            abstention_code=abstention_code,
            suggested_action=suggested_action,
            **context,
        )
        self.abstention_code = abstention_code
        self.suggested_action = suggested_action
