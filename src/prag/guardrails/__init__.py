"""Guardrails: the input and output chains, the retrieval screen, and the tool gate.

Configured by name. A name this package does not know fails at startup, because a misspelled
guardrail that silently never runs is a guardrail switched off by typo. A name it knows but does
not implement yet is reported in ``GuardrailSet.deferred``, so the gap is visible on the health
endpoint instead of being assumed covered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any

from prag.core.errors import ConfigurationError
from prag.core.models.common import GuardrailPhase
from prag.guardrails.chain import ChainOutcome, GuardrailChain
from prag.guardrails.checks import (
    CitationValidationGuardrail,
    LeakageGuardrail,
    PatternGuardrail,
    PayloadLimitsGuardrail,
    PiiGuardrail,
)
from prag.guardrails.detectors import INJECTION_PATTERNS, OVERRIDE_PATTERNS
from prag.guardrails.screen import RETRIEVAL_CHECKS, RetrievalScreen
from prag.guardrails.tools import ProvenanceGatedToolExecutor

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

__all__ = [
    "ChainOutcome",
    "CitationValidationGuardrail",
    "GuardrailChain",
    "GuardrailSet",
    "LeakageGuardrail",
    "PatternGuardrail",
    "PayloadLimitsGuardrail",
    "PiiGuardrail",
    "ProvenanceGatedToolExecutor",
    "RetrievalScreen",
    "build_guardrails",
]

#: Configured names enforced by construction elsewhere, so they have no chain member.
_STRUCTURAL = {
    # The tenant is resolved from the authenticated principal and no request field can name
    # another, so there is nothing for an assertion to compare.
    "tenant_assertion",
    # Claims are verified against evidence in the generate node before citations are bound.
    "grounding",
    # The envelope is a validated model; an invalid one cannot be constructed.
    "schema",
}

#: Configured names with no implementation yet, and what each is waiting for.
_DEFERRED = {
    "policy": "needs a tenant content-policy classifier",
    "poison_heuristics": "needs per-source embedding distributions to detect anomalies against",
}


@dataclass(frozen=True, slots=True)
class GuardrailSet:
    input_chain: GuardrailChain
    output_chain: GuardrailChain
    screen: RetrievalScreen
    deferred: dict[str, str] = field(default_factory=dict)


def build_guardrails(
    *,
    input_names: Sequence[str],
    retrieval_names: Sequence[str],
    output_names: Sequence[str],
    canaries: Mapping[str, Sequence[str]] | None = None,
    redact_pii_before_generation: bool = False,
    max_query_chars: int = 8_000,
) -> GuardrailSet:
    """Build every phase from its configured names, in configured order."""
    input_factories = {
        "injection": lambda: PatternGuardrail(
            "injection", INJECTION_PATTERNS, reason_code="prompt_injection"
        ),
        "instruction_override": lambda: PatternGuardrail(
            "instruction_override", OVERRIDE_PATTERNS, reason_code="instruction_override"
        ),
        "pii": lambda: PiiGuardrail(redact_before_generation=redact_pii_before_generation),
        "payload_limits": lambda: PayloadLimitsGuardrail(max_chars=max_query_chars),
    }
    output_factories = {
        "citation_validation": CitationValidationGuardrail,
        "leakage": lambda: LeakageGuardrail(canaries=canaries),
    }

    deferred: dict[str, str] = {}

    def resolve(
        phase: GuardrailPhase, names: Sequence[str], known: Mapping[str, Callable[[], Any]]
    ) -> list[Any]:
        built: list[Any] = []
        for name in names:
            if name in known:
                built.append(known[name]())
            elif name in _DEFERRED:
                deferred[f"{phase}.{name}"] = _DEFERRED[name]
            elif name not in _STRUCTURAL:
                raise ConfigurationError(
                    "unknown guardrail name", phase=str(phase), guardrail=name
                )
        return built

    input_chain = GuardrailChain(resolve(GuardrailPhase.INPUT, input_names, input_factories))
    output_chain = GuardrailChain(resolve(GuardrailPhase.OUTPUT, output_names, output_factories))
    # The screen takes check names rather than objects: its checks share one pass over each
    # group's text, which separate chain members could not.
    screen_checks = {name: partial(str, name) for name in RETRIEVAL_CHECKS}
    screen = RetrievalScreen(
        checks=resolve(GuardrailPhase.RETRIEVAL, retrieval_names, screen_checks),
        canaries=canaries,
    )
    return GuardrailSet(input_chain, output_chain, screen, deferred)
