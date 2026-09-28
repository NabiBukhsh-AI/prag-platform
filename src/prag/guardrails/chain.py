"""The guardrail chain.

Guardrails are a middleware chain, not a graph stage. The chain runs the guardrails for one phase
in configured order, passes each MODIFY forward so later checks see the modified payload, and
stops at the first BLOCK.

It fails closed. A guardrail that raises is a BLOCK, not a skipped check: a chain that lets a
request through whenever a detector crashes has turned every detector bug into a bypass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from prag.core.errors import GuardrailBlocked, Severity
from prag.core.models.guardrails import GuardrailPayload, GuardrailVerdict, VerdictAction

if TYPE_CHECKING:
    from collections.abc import Sequence

    from prag.core.protocols.crosscutting import Guardrail

__all__ = ["ChainOutcome", "GuardrailChain"]


@dataclass(frozen=True, slots=True)
class ChainOutcome:
    """The payload after every MODIFY, and every verdict that produced it."""

    payload: GuardrailPayload
    verdicts: tuple[GuardrailVerdict, ...]

    @property
    def blocking(self) -> GuardrailVerdict | None:
        return next((v for v in self.verdicts if v.blocked), None)

    def raise_if_blocked(self, state: object = None) -> None:
        """Raise ``GuardrailBlocked`` carrying the reason code and never the detail.

        The detail may quote an injected span or name a secret's kind; a client is told that a
        guardrail refused and which one, and the rest stays in the trace. ``state`` rides on the
        exception so the caller can still publish the verdicts that caused the refusal.
        """
        verdict = self.blocking
        if verdict is not None:
            error = GuardrailBlocked(
                f"request refused by guardrail: {verdict.reason_code}",
                guardrail=verdict.guardrail,
                phase=str(verdict.phase),
            )
            error.state = state
            raise error


class GuardrailChain:
    """Runs the guardrails for a payload's phase, in order."""

    def __init__(self, guardrails: Sequence[Guardrail]) -> None:
        self._guardrails = tuple(guardrails)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(g.name for g in self._guardrails)

    async def run(self, payload: GuardrailPayload) -> ChainOutcome:
        verdicts: list[GuardrailVerdict] = []
        for guardrail in self._guardrails:
            if guardrail.phase is not payload.phase:
                continue

            started = time.monotonic()
            try:
                verdict = await guardrail.check(payload)
            except Exception as exc:
                verdict = GuardrailVerdict(
                    guardrail=guardrail.name,
                    phase=payload.phase,
                    action=VerdictAction.BLOCK,
                    severity=Severity.ERROR,
                    reason_code="guardrail_error",
                    detail=type(exc).__name__,
                )
            verdict = verdict.model_copy(
                update={"latency_ms": int((time.monotonic() - started) * 1000)}
            )
            verdicts.append(verdict)

            if verdict.blocked:
                break
            if verdict.action is VerdictAction.MODIFY and verdict.modified_payload is not None:
                payload = verdict.modified_payload

        return ChainOutcome(payload=payload, verdicts=tuple(verdicts))
