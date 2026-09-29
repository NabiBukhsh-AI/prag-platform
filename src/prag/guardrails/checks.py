"""Input and output guardrails.

Each one implements the ``Guardrail`` protocol and nothing else: it inspects a payload and returns
a verdict, and any change it wants is a replacement payload in that verdict. None of them holds
state between calls, which is what lets the chain run them in any configured order and lets each
be tested alone.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from prag.core.errors import Severity
from prag.core.models.common import GuardrailPhase
from prag.core.models.guardrails import GuardrailPayload, GuardrailVerdict, VerdictAction
from prag.guardrails.detectors import find_first, find_pii, find_secret, redact_pii

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "CitationValidationGuardrail",
    "LeakageGuardrail",
    "PatternGuardrail",
    "PayloadLimitsGuardrail",
    "PiiGuardrail",
]

_MARKER = re.compile(r"\s?\[([EM]\d+)\]")


def _inspected(payload: GuardrailPayload) -> str:
    """The text a phase inspects: the query in, the evidence at retrieval, the answer out."""
    match payload.phase:
        case GuardrailPhase.INPUT:
            return payload.query or ""
        case GuardrailPhase.RETRIEVAL:
            return "\n".join(payload.evidence_texts)
        case GuardrailPhase.PRE_GENERATION:
            return "\n".join(payload.prompt_regions)
        case GuardrailPhase.OUTPUT:
            return payload.answer or ""


def _allow(name: str, phase: GuardrailPhase, **fields: object) -> GuardrailVerdict:
    return GuardrailVerdict(guardrail=name, phase=phase, action=VerdictAction.ALLOW, **fields)


class PatternGuardrail:
    """Blocks when any of a pattern family matches.

    One class for the injection and instruction-override checks rather than two, since they
    differ only in the patterns and the reason code. The matched span goes in ``detail``, which
    is redacted before logging: an incident review needs to see what triggered the block.
    """

    def __init__(
        self,
        name: str,
        patterns: Sequence[re.Pattern[str]],
        *,
        phase: GuardrailPhase = GuardrailPhase.INPUT,
        reason_code: str,
        severity: Severity = Severity.WARNING,
    ) -> None:
        self.name = name
        self.phase = phase
        self.severity = severity
        self._patterns = tuple(patterns)
        self._reason_code = reason_code

    async def check(self, payload: GuardrailPayload) -> GuardrailVerdict:
        match = find_first(self._patterns, _inspected(payload))
        if match is None:
            return _allow(self.name, self.phase)
        return GuardrailVerdict(
            guardrail=self.name,
            phase=self.phase,
            action=VerdictAction.BLOCK,
            severity=self.severity,
            reason_code=self._reason_code,
            detail=match.group()[:200],
        )


class PiiGuardrail:
    """Detects PII in the query, and redacts it only when tenant policy asks.

    Not redacted before generation by default: a user asking about their own account needs the
    model to see the account number. Logs are redacted at the logging boundary regardless, and
    this verdict records which kinds were present so the trace says so without quoting them.
    """

    name = "pii"
    phase = GuardrailPhase.INPUT
    severity = Severity.INFO

    def __init__(self, *, redact_before_generation: bool = False) -> None:
        self._redact = redact_before_generation

    async def check(self, payload: GuardrailPayload) -> GuardrailVerdict:
        query = payload.query or ""
        kinds = find_pii(query)
        if not kinds:
            return _allow(self.name, self.phase)
        if not self._redact:
            return _allow(
                self.name, self.phase, reason_code="pii_detected", detail=",".join(kinds)
            )
        redacted, _ = redact_pii(query)
        return GuardrailVerdict(
            guardrail=self.name,
            phase=self.phase,
            action=VerdictAction.MODIFY,
            reason_code="pii_redacted",
            detail=",".join(kinds),
            modified_payload=payload.model_copy(update={"query": redacted}),
        )


class PayloadLimitsGuardrail:
    """Rejects oversized queries and ones carrying control characters.

    Control characters are rejected rather than stripped. There is no legitimate reason for a
    query to contain them, and the illegitimate ones — terminal escapes in a log viewer, NULs
    that truncate a downstream C string — are better refused than half-cleaned.
    """

    name = "payload_limits"
    phase = GuardrailPhase.INPUT
    severity = Severity.INFO

    _CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    def __init__(self, *, max_chars: int = 8_000) -> None:
        self._max_chars = max_chars

    async def check(self, payload: GuardrailPayload) -> GuardrailVerdict:
        query = payload.query or ""
        reason = (
            "payload_too_large"
            if len(query) > self._max_chars
            else "payload_encoding"
            if self._CONTROL.search(query)
            else None
        )
        if reason is None:
            return _allow(self.name, self.phase)
        return GuardrailVerdict(
            guardrail=self.name,
            phase=self.phase,
            action=VerdictAction.BLOCK,
            severity=self.severity,
            reason_code=reason,
        )


class CitationValidationGuardrail:
    """Strips citation markers that do not resolve to evidence in this request's context.

    An invalid citation is a hallucination even when the claim beside it happens to be true: it
    asserts a source said something the source was never shown to say. The claim stays and the
    marker goes, because the grounding verifier has already judged the claim on its own.

    Expects ``metadata["evidence_markers"]`` and ``metadata["memory_markers"]``: the markers of
    the evidence groups and memory items actually in context.
    """

    name = "citation_validation"
    phase = GuardrailPhase.OUTPUT
    severity = Severity.WARNING

    async def check(self, payload: GuardrailPayload) -> GuardrailVerdict:
        answer = payload.answer or ""
        # Two namespaces, each resolved only against its own items: an evidence marker against
        # the groups in context, a memory marker against the memory items in context. A marker
        # that crosses the boundary resolves to nothing and is stripped.
        valid = set(payload.metadata.get("evidence_markers", ())) | set(
            payload.metadata.get("memory_markers", ())
        )
        invalid = sorted({m for m in _MARKER.findall(answer) if m not in valid})
        if not invalid:
            return _allow(self.name, self.phase)

        stripped = _MARKER.sub(lambda m: m.group() if m.group(1) in valid else "", answer)
        return GuardrailVerdict(
            guardrail=self.name,
            phase=self.phase,
            action=VerdictAction.MODIFY,
            severity=self.severity,
            reason_code="invalid_citation",
            detail=",".join(invalid),
            modified_payload=payload.model_copy(update={"answer": stripped}),
        )


class LeakageGuardrail:
    """Blocks output carrying secrets, exfiltration-shaped URLs, or another tenant's canary.

    A canary sighting is critical rather than merely blocked: a string seeded only in another
    tenant's corpus appearing in this tenant's answer proves an isolation break somewhere
    upstream, whatever the retrieval-phase checks concluded.
    """

    name = "leakage"
    phase = GuardrailPhase.OUTPUT
    severity = Severity.ERROR

    def __init__(self, *, canaries: Mapping[str, Sequence[str]] | None = None) -> None:
        self._canaries = {t: tuple(c) for t, c in (canaries or {}).items()}

    async def check(self, payload: GuardrailPayload) -> GuardrailVerdict:
        answer = payload.answer or ""
        for tenant_id, strings in self._canaries.items():
            if tenant_id != payload.tenant_id and any(s in answer for s in strings):
                return GuardrailVerdict(
                    guardrail=self.name,
                    phase=self.phase,
                    action=VerdictAction.BLOCK,
                    severity=Severity.CRITICAL,
                    reason_code="canary_detected",
                )

        kind = find_secret(answer)
        if kind is None:
            return _allow(self.name, self.phase)
        return GuardrailVerdict(
            guardrail=self.name,
            phase=self.phase,
            action=VerdictAction.BLOCK,
            severity=self.severity,
            reason_code="secret_detected",
            # The kind, never the span. Quoting a leaked key into a verdict would leak it again.
            detail=kind,
        )
