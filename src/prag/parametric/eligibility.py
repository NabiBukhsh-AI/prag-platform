"""The Parametric Eligibility Gate.

No knowledge becomes parametric without passing it. A pure function over a ``KnowledgeRecord``,
evaluated at ingestion and again at every retraining cycle, so the two must agree — which is why
nothing here consults live state.

Governance first, economics second. Every hard blocker is reported, never only the first, and no
economic argument overrides one: knowledge that cannot be filtered per request, revoked in time,
or quoted verbatim is ineligible at any price.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from prag.core.models.parametric import (
    EligibilityResult,
    KnowledgeClass,
    KnowledgeRecord,
    ParametricEconomics,
)

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["BLOCKERS", "EligibilityGate"]

#: Classes that can never be parameterized, whatever else is true of them. Session and user
#: memory are per-principal; real-time and dynamic knowledge change faster than weights can.
_NEVER_PARAMETRIC = frozenset(
    {
        KnowledgeClass.SESSION,
        KnowledgeClass.LONG_TERM_USER,
        KnowledgeClass.REAL_TIME,
        KnowledgeClass.NON_PARAMETRIC_DYNAMIC,
    }
)

#: Every blocker code, for the tests and for anyone grouping gate results over time.
BLOCKERS = (
    "knowledge_class_not_parameterizable",
    "acl_narrower_than_tenant",
    "revocation_sla_shorter_than_retrain_cadence",
    "requires_exact_quotation",
    "half_life_below_floor",
    "per_claim_provenance_without_shadowing",
    "contested",
    "contains_pii",
)


class EligibilityGate:
    """Implements ``ParametricEligibilityGate``."""

    def __init__(
        self,
        *,
        min_half_life_days: float = 90.0,
        retrain_cadence_hours: float = 168.0,
        min_queries_per_period: float = 500.0,
        savings_threshold: float = 0.6,
        coherence_floor: float = 0.55,
        provenance_shadowing_enabled: bool = True,
        block_if_pii: bool = True,
        block_if_acl_narrower_than_tenant: bool = True,
        block_if_requires_exact_quotation: bool = True,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._min_half_life = min_half_life_days
        self._cadence = retrain_cadence_hours
        self._min_volume = min_queries_per_period
        self._savings = savings_threshold
        self._coherence = coherence_floor
        self._shadowing = provenance_shadowing_enabled
        self._block_pii = block_if_pii
        self._block_acl = block_if_acl_narrower_than_tenant
        self._block_quote = block_if_requires_exact_quotation
        self._clock = clock

    def evaluate(
        self, record: KnowledgeRecord, economics: ParametricEconomics
    ) -> EligibilityResult:
        blockers = self._blockers(record)
        now = int(self._clock() * 1000)
        if blockers:
            # Economics are not weighed at all: a policy refusal will not change as traffic
            # grows, and reporting a cost comparison beside it would suggest that it might.
            return EligibilityResult(
                eligible=False, blocking_reasons=tuple(blockers), evaluated_at_ms=now
            )

        amortized = economics.amortized_parametric_cost_usd
        nonparametric = economics.per_query_nonparametric_cost_usd
        economic: list[str] = []
        if economics.expected_queries_per_period < self._min_volume:
            economic.append("volume_below_floor")
        if not amortized < nonparametric * self._savings:
            economic.append("savings_below_threshold")
        if economics.cluster_coherence_score < self._coherence:
            economic.append("cluster_incoherent")

        return EligibilityResult(
            eligible=not economic,
            economic_reasons=tuple(economic),
            amortized_cost_usd=None if amortized == float("inf") else amortized,
            nonparametric_cost_usd=nonparametric,
            evaluated_at_ms=now,
        )

    def _blockers(self, record: KnowledgeRecord) -> list[str]:
        """Every hard blocker that applies, in a fixed order."""
        checks = (
            record.knowledge_class in _NEVER_PARAMETRIC,
            # Parameters cannot be filtered per request, so anything some users of the tenant
            # may not see can never be in weights that serve the tenant.
            self._block_acl and record.acl_narrower_than_tenant,
            record.revocation_sla_hours is not None and record.revocation_sla_hours < self._cadence,
            self._block_quote and record.requires_exact_quotation,
            record.estimated_half_life_days < self._min_half_life,
            record.requires_per_claim_provenance and not self._shadowing,
            record.contested,
            self._block_pii and record.contains_pii,
        )
        return [code for code, blocked in zip(BLOCKERS, checks, strict=True) if blocked]
