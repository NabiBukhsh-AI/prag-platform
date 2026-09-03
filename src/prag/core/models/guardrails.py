"""Guardrail payloads and verdicts.

A guardrail returns one of three things: ALLOW, MODIFY with a replacement payload, or BLOCK with
a reason. MODIFY exists because the useful response to detected PII is usually redaction rather
than refusal, and forcing that choice into a binary would make the chain either useless or
unusable.

Guardrails are side-effect free apart from emitting security events. A guardrail that mutates
shared state cannot be run twice, cannot be run out of order, and cannot be tested in isolation
— and the chain does all three.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prag.core.errors import Severity
from prag.core.models.common import GuardrailPhase

__all__ = [
    "GuardrailPayload",
    "GuardrailVerdict",
    "VerdictAction",
]


class VerdictAction(StrEnum):
    ALLOW = "allow"
    MODIFY = "modify"
    BLOCK = "block"


class GuardrailPayload(BaseModel):
    """What a guardrail inspects.

    One shape for all four phases rather than a type per phase. The phases inspect different
    fields, but a single shape is what lets the chain be a list the configuration can reorder,
    and lets a guardrail be moved between phases without rewriting it.
    """

    model_config = ConfigDict(frozen=True)

    phase: GuardrailPhase
    request_id: str
    tenant_id: str

    #: The user's query, at the INPUT phase.
    query: str | None = None
    #: Retrieved text, at the RETRIEVAL phase. Untrusted by definition.
    evidence_texts: tuple[str, ...] = ()
    #: The rendered prompt, at PRE_GENERATION.
    prompt_regions: tuple[str, ...] = ()
    #: Generated text, at OUTPUT. May be a partial buffer during streaming.
    answer: str | None = None
    #: Set while streaming, so an output guardrail knows it is seeing a sentence rather than a
    #: finished answer and can defer checks that need the whole text.
    partial: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class GuardrailVerdict(BaseModel):
    """One guardrail's decision.

    ``modified_payload`` carries the replacement when the action is MODIFY. Later guardrails in
    the chain see the modified version, so ordering matters and is configuration rather than
    code.
    """

    model_config = ConfigDict(frozen=True)

    guardrail: str
    phase: GuardrailPhase
    action: VerdictAction
    severity: Severity = Severity.INFO
    #: Stable code, safe to put in a metric label or return to a client.
    reason_code: str | None = None
    #: Human-readable detail. May contain sensitive fragments, so it is redacted before logging
    #: and never returned to a client verbatim.
    detail: str | None = None
    modified_payload: GuardrailPayload | None = None
    latency_ms: int = Field(default=0, ge=0)

    @property
    def blocked(self) -> bool:
        return self.action is VerdictAction.BLOCK

    @property
    def is_security_event(self) -> bool:
        """Whether this verdict should be published as a security event.

        Anything blocking, and anything at warning severity or above. A quiet ALLOW is not
        interesting; an ALLOW that noticed something is.
        """
        return self.blocked or self.severity in (Severity.WARNING, Severity.CRITICAL)
