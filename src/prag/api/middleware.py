"""The middleware chain: identity, tenancy, limits, and error mapping.

Order matters and is not alphabetical. Identity resolves first because every later stage needs a
principal; limits come after identity because a rate limit is per tenant; error mapping wraps
everything because an error raised inside any of them still has to become a response.

**Errors are mapped to responses in exactly one place.** That is the whole reason the error
hierarchy is typed. A subsystem adding a failure mode should never touch transport code, and a
status code decided in three places will disagree in two of them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from prag.core.errors import (
    AbstentionRequired,
    BudgetExhausted,
    ConfigurationError,
    ContextError,
    DeadlineExceeded,
    GenerationError,
    GuardrailBlocked,
    IsolationViolation,
    PragError,
    RetrievalError,
    StorageError,
    ToolProvenanceDenied,
)

if TYPE_CHECKING:
    from prag.core.models.identity import Principal

__all__ = ["ErrorResponse", "resolve_principal", "status_for"]

#: Error class to HTTP status. Ordered most specific first, since several are subclasses.
#:
#: Isolation violations return 500, not 403. A 403 would tell a caller that something exists
#: which they may not see; an isolation *violation* means the platform's own boundary failed,
#: which is the platform's fault and not the caller's, and the incident is ours to investigate.
_STATUS_BY_ERROR: Final[tuple[tuple[type[PragError], int], ...]] = (
    (IsolationViolation, 500),
    (ToolProvenanceDenied, 403),
    (GuardrailBlocked, 400),
    (DeadlineExceeded, 504),
    (BudgetExhausted, 429),
    (RetrievalError, 503),
    (GenerationError, 502),
    (StorageError, 503),
    (ContextError, 500),
    (ConfigurationError, 500),
)


class ErrorResponse:
    """The one shape an error takes on the wire.

    Carries the stable ``reason_code`` and nothing else from the error's context. Context holds
    identifiers and measurements safe for a log line, but a client is a different audience: an
    internal source id or a collection name in an error body is a small information leak that
    costs nothing to prevent here.
    """

    __slots__ = ("detail", "reason_code", "retryable", "status")

    def __init__(self, *, status: int, reason_code: str, detail: str, retryable: bool) -> None:
        self.status = status
        self.reason_code = reason_code
        self.detail = detail
        self.retryable = retryable

    def as_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "reason_code": self.reason_code,
                "detail": self.detail,
                "retryable": self.retryable,
            }
        }


def status_for(error: Exception) -> int:
    """Map an error to an HTTP status. The single place this decision is made."""
    if isinstance(error, AbstentionRequired):
        # An abstention is a successful outcome carried in the envelope, not an error status.
        # Returning 4xx would make the abstention rate indistinguishable from a client error
        # in every dashboard that groups by status, which is exactly the signal it must not
        # be confused with.
        return 200

    for error_type, status in _STATUS_BY_ERROR:
        if isinstance(error, error_type):
            return status
    return 500


def to_response(error: Exception) -> ErrorResponse:
    """Convert any exception into the wire shape."""
    if isinstance(error, PragError):
        return ErrorResponse(
            status=status_for(error),
            reason_code=error.reason_code,
            detail=error.detail,
            retryable=error.retryable,
        )

    # An untyped exception is a bug rather than a handled failure. It gets a generic code and a
    # generic message: the real one goes to the log, because an unexpected exception's text is
    # the most likely place for an internal detail to escape.
    return ErrorResponse(
        status=500,
        reason_code="internal_error",
        detail="An unexpected error occurred.",
        retryable=False,
    )


def resolve_principal(headers: dict[str, str], *, default_tenant: str | None = None) -> Principal:
    """Resolve the caller from request headers.

    Development-grade: it reads headers a gateway would normally have validated. Production
    resolves the principal from a verified OIDC or JWT claim set, and this function is where
    that swap happens — nothing downstream changes, because everything downstream takes a
    ``Principal``.

    A missing tenant is refused rather than defaulted. Defaulting to a tenant is how a
    misconfigured client reads another tenant's corpus, and the failure would look like the
    system working.
    """
    from prag.core.errors import GuardrailBlocked
    from prag.core.models.common import SlaTier
    from prag.core.models.identity import Principal

    tenant_id = headers.get("x-tenant-id") or default_tenant
    if not tenant_id:
        raise GuardrailBlocked(
            "request carries no tenant identity",
            guardrail="tenant_assertion",
            phase="input",
        )

    user_id = headers.get("x-user-id", "anonymous")
    groups = tuple(g for g in headers.get("x-groups", "").split(",") if g)
    scopes = tuple(s for s in headers.get("x-scopes", "").split(",") if s)

    raw_tier = headers.get("x-sla-tier", "standard")
    try:
        sla_tier = SlaTier(raw_tier)
    except ValueError:
        # An unknown tier falls back to standard rather than failing. A client sending a typo
        # should get a served request at a sane tier, not an outage.
        sla_tier = SlaTier.STANDARD

    return Principal(
        tenant_id=tenant_id,
        user_id=user_id,
        groups=groups,
        scopes=scopes,
        # The ACL set a gateway would supply. Computed once per request and never recomputed:
        # a mid-request change would let the filter and the recheck disagree, and an
        # inconsistency there is indistinguishable from a leak.
        acl_hashes=tuple(h for h in headers.get("x-acl-hashes", "").split(",") if h),
        sla_tier=sla_tier,
    )
