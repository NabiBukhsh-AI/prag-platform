"""Fusion through the whole platform: conflicts surfaced in the answer, and the abstention gates.

Two independent documents disagree about the retention window. With comparable authority the
answer must present both; with a clear authority gap it must prefer the stronger and still name
the weaker. In both cases the conflict is in the envelope as structure, and the prose is
generated from that structure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prag.api import build_platform
from prag.config import PragSettings
from prag.core.errors import AbstentionRequired
from prag.core.models.fusion import ConflictKind, ConflictResolution

pytestmark = pytest.mark.e2e

SEED = Path(__file__).resolve().parents[2] / "eval" / "seed" / "corpus" / "tenant-local" / "public"
RUNBOOK = (SEED / "runbook.incident-response.md").read_text(encoding="utf-8")
TENANT = "tenant-local"
WIKI = """# Team Wiki

## Data retention

Incident records are retained for 90 days and then archived to cold storage, where they remain
queryable for a further 12 months before permanent deletion.
"""


async def a_platform(*, wiki_authority: float):
    platform = build_platform(PragSettings())
    await platform.ingest_markdown(
        RUNBOOK, document_id="runbook.incident-response", tenant_id=TENANT, authority=0.8
    )
    await platform.ingest_markdown(
        WIKI, document_id="team.wiki", tenant_id=TENANT, authority=wiki_authority
    )
    return platform


async def ask(platform, query: str):
    state = platform.request_state(query, {"x-tenant-id": TENANT, "x-user-id": "u"})
    return await platform.answer(state)


class TestConflictSurfacing:
    async def test_comparable_authority_presents_both_positions(self) -> None:
        platform = await a_platform(wiki_authority=0.75)
        run = await ask(platform, "how long are incident records retained")
        envelope = run.state.result

        assert envelope is not None
        (conflict,) = [c for c in envelope.conflicts if c.kind is ConflictKind.SOURCE_VS_SOURCE]
        assert conflict.resolution is ConflictResolution.SURFACED
        assert {p.origin_id for p in conflict.positions} == {
            "runbook.incident-response",
            "team.wiki",
        }
        assert "Sources disagree" in envelope.answer
        assert "30 days" in envelope.answer
        assert "90 days" in envelope.answer

    async def test_a_clear_authority_gap_prefers_the_stronger_and_names_the_weaker(self) -> None:
        platform = await a_platform(wiki_authority=0.3)
        run = await ask(platform, "how long are incident records retained")
        envelope = run.state.result

        assert envelope is not None
        (conflict,) = envelope.conflicts
        assert conflict.resolution is ConflictResolution.AUTHORITY_WINS
        assert conflict.positions[0].origin_id == "runbook.incident-response"
        assert "lower-authority source (team.wiki)" in envelope.answer

    async def test_agreeing_sources_raise_no_conflict(self) -> None:
        platform = build_platform(PragSettings())
        await platform.ingest_markdown(
            RUNBOOK, document_id="runbook.incident-response", tenant_id=TENANT
        )
        run = await ask(platform, "how long are incident records retained")
        assert run.state.result is not None
        assert run.state.result.conflicts == ()


class TestGates:
    async def test_an_off_topic_retrieval_abstains_instead_of_answering_noise(self) -> None:
        """The corpus has nothing on this; returning its nearest sentence would be noise."""
        platform = build_platform(PragSettings())
        await platform.ingest_markdown(
            RUNBOOK, document_id="runbook.incident-response", tenant_id=TENANT
        )
        with pytest.raises(AbstentionRequired) as excinfo:
            await ask(platform, "what is the Project Nightjar budget")
        assert excinfo.value.abstention_code == "knowledge_below_floor"
