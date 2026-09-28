"""The retrieval-phase guardrails.

The ones that matter most and are most often missing. Every check here runs on every retrieved
group, at retrieval time rather than only at ingest, because sources mutate: a clean document
can be edited into an injection vector after it was indexed, and an index filter can regress
without any test noticing.

Four checks, most severe first: a canary sighting fails the whole request, since it proves a
tenant boundary broke; an ACL mismatch drops the group and pages, since it proves the index filter
is wrong; a quarantined source and a document-borne injection drop the group.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prag.core.errors import IsolationViolation, Severity
from prag.core.models.common import GuardrailPhase
from prag.core.models.guardrails import GuardrailVerdict, ScreenResult, VerdictAction
from prag.guardrails.detectors import INJECTION_PATTERNS, OVERRIDE_PATTERNS, find_first

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from prag.core.models.identity import Principal
    from prag.core.models.retrieval import EvidenceGroup

__all__ = ["RETRIEVAL_CHECKS", "RetrievalScreen"]

#: The retrieval-phase checks this screen implements, by their configuration names.
RETRIEVAL_CHECKS = frozenset({"doc_injection", "acl_recheck", "canary", "source_health"})

#: Visible to every principal within a tenant, matching the index-side ACL filter.
_PUBLIC_ACL = "public"


class RetrievalScreen:
    """Implements ``EvidenceScreen``."""

    def __init__(
        self,
        *,
        checks: Iterable[str] = RETRIEVAL_CHECKS,
        canaries: Mapping[str, Sequence[str]] | None = None,
        quarantined_sources: Iterable[str] = (),
    ) -> None:
        self._checks = frozenset(checks)
        self._canaries = {t: tuple(c) for t, c in (canaries or {}).items()}
        # Mutable on purpose: quarantine is an operational action taken while serving, and a
        # restart to drop a poisoned source would leave it serving until the deploy lands.
        self._quarantined = set(quarantined_sources)

    def quarantine(self, source_id: str) -> None:
        self._quarantined.add(source_id)

    def reinstate(self, source_id: str) -> None:
        """Human action only. Nothing in the request path calls this."""
        self._quarantined.discard(source_id)

    async def screen(self, principal: Principal, groups: Sequence[EvidenceGroup]) -> ScreenResult:
        kept: list[EvidenceGroup] = []
        verdicts: list[GuardrailVerdict] = []
        for group in groups:
            verdict = self._check(principal, group)
            if verdict is None:
                kept.append(group)
            else:
                verdicts.append(verdict)
        return ScreenResult(kept=tuple(kept), verdicts=tuple(verdicts))

    def _check(self, principal: Principal, group: EvidenceGroup) -> GuardrailVerdict | None:
        texts = [t for m in group.members for t in (m.text, m.parent_text) if t]

        if "canary" in self._checks:
            for tenant_id, strings in self._canaries.items():
                if tenant_id != principal.tenant_id and any(s in t for s in strings for t in texts):
                    # Raised, not dropped. Answering from what is left would serve a request
                    # that has already crossed a tenant boundary.
                    raise IsolationViolation(
                        "another tenant's canary appeared in retrieved evidence",
                        group_id=group.group_id,
                        requesting_tenant=principal.tenant_id,
                    )

        if "acl_recheck" in self._checks:
            allowed = {*principal.acl_hashes, _PUBLIC_ACL}
            for member in group.members:
                meta = member.metadata
                # An unknown owner is a mismatch. The recheck exists because it does not trust
                # the filter, and "the index did not say" is not evidence of access.
                if meta.tenant_id != principal.tenant_id or meta.acl_hash not in allowed:
                    return self._drop(
                        group, "acl_recheck", "acl_recheck_mismatch", Severity.CRITICAL
                    )

        if "source_health" in self._checks and any(
            m.source_id in self._quarantined for m in group.members
        ):
            return self._drop(group, "source_health", "source_quarantined", Severity.INFO)

        if "doc_injection" in self._checks:
            for text in texts:
                if find_first((*INJECTION_PATTERNS, *OVERRIDE_PATTERNS), text) is not None:
                    return self._drop(
                        group, "doc_injection", "document_injection", Severity.WARNING
                    )

        return None

    @staticmethod
    def _drop(
        group: EvidenceGroup, name: str, reason_code: str, severity: Severity
    ) -> GuardrailVerdict:
        return GuardrailVerdict(
            guardrail=name,
            phase=GuardrailPhase.RETRIEVAL,
            action=VerdictAction.BLOCK,
            severity=severity,
            reason_code=reason_code,
            subject_id=group.group_id,
            # The source, so the health review knows where to look; never the text.
            detail=group.representative.source_id,
        )
