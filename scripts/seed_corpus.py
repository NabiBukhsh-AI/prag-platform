#!/usr/bin/env python
"""Seed the local corpus.

The whole point of the local stack is that `docker compose up` plus this script yields a working
system on a laptop, with no cloud dependency and no model download. Everything it uses is the
same code the API uses — the ingestion path here is `Platform.ingest_markdown`, not a
reimplementation, because two ingestion paths is two places for the chunking to differ.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prag.api import build_platform
from prag.config import PragSettings

SEED_DOCUMENTS: dict[str, str] = {
    "runbook.incident-response": """# Incident Response

## Severity levels

Sev-1 means a total outage affecting every tenant of the platform. It pages the on-call lead
immediately and opens a bridge call for the duration of the incident.

Sev-2 means degraded service for a subset of tenants. It is paged during business hours only,
with a four hour response target rather than fifteen minutes.

## Escalation

For a sev-1 incident the on-call lead must be paged within 15 minutes of detection. If the page
is unacknowledged after 5 minutes, escalation moves to the engineering manager, and after a
further 10 minutes to the director of engineering.

Escalation is automatic and does not require anyone to make a judgement call at three in the
morning, which is the entire point of having a written policy.

## Data retention

Incident records are retained for 30 days and then archived to cold storage, where they remain
queryable for a further 12 months before permanent deletion.

## Bridge calls

A sev-1 bridge call opens automatically when the page fires. The incident commander is whoever
holds the primary rota at that moment, not whoever happens to be awake and not whoever is most
senior. Handing command to a more senior person who has just joined costs the context the
commander has already built, and that context is usually worth more than the seniority.

The bridge stays open until the incident is downgraded or resolved. A scribe records decisions
with timestamps, because a postmortem written from memory reliably reconstructs a tidier
sequence of events than the one that actually happened.

## Communication

Customer communication for a sev-1 goes out within 30 minutes of confirmation, whether or not
the cause is known. Saying "we are investigating" early beats saying nothing accurately later:
silence during an outage is read as absence rather than as diligence.

Status page updates continue at 30 minute intervals until resolution, even when the update is
that nothing has changed. A gap in updates is indistinguishable, from outside, from a team that
has stopped working on the problem.

## Postmortems

Every sev-1 requires a written postmortem within five working days. Sev-2 incidents require one
only when they recur within a quarter, because a single degraded afternoon rarely teaches
anything a recurring one does not teach better.

Postmortems are blameless by policy. The purpose is a change to the system, not a change to
whoever was on call, and a review that produces the second outcome will not produce the first.
""",
    "policy.access-control": """# Access Control Policy

## Roles

The platform defines five roles: viewer, contributor, curator, tenant admin, and platform admin.
A curator manages sources and authority scores. Adapter promotion and authority overrides require
curator or higher and are audit-logged.

## Tenant isolation

Isolation is enforced at five independent layers, and any single layer failing must not produce a
breach. Row-level security scopes every query by tenant. The vector index filters on payload.
Candidate processing rechecks ACLs without trusting the index. Adapter selection filters by
tenant scope before scoring. Every cache key carries the tenant id.

## Secrets

Secrets live in a dedicated manager and are rotated on schedule. They are never stored in
environment variables in production and never committed to a repository.

## Authority scoring

Every registered source carries an authority score between zero and one, and a set of
domain-scoped overrides. A source that is the system of record for legal matters may be
worthless on engineering ones, and collapsing that into a single number forces one of those two
judgements to be wrong everywhere it is applied.

Authority scores are set by a curator and carry the curator's identity and a timestamp.
Authority is the strongest signal in conflict resolution, so a score with no accountable owner
is a way for one person's opinion to silently outrank a system of record.

## Staleness

Sources move through a staleness state machine: fresh, aging, stale, expired, reindexing.
Entering the stale state enqueues a reindex automatically. Entering expired makes the source
unusable in strict mode, and usable elsewhere only with an explicit age warning attached to the
answer.

Staleness is measured against each source's expected half-life rather than against a fixed
window. A two-year-old constitutional provision is fresh; a two-day-old exchange rate is not.

## Right to erasure

Erasure is a first-class workflow rather than a manual procedure. It removes the document from
object storage, tombstones and purges it from every index, purges cache entries by document id,
identifies every adapter trained on it, revokes those adapters, and queues the affected clusters
for retraining. The whole chain is recorded in the audit log.

Revocation degrades to the non-parametric path. It never degrades to serving stale weights.
""",
}


async def main() -> int:
    parser = argparse.ArgumentParser(description="Seed the local prag corpus.")
    parser.add_argument("--tenant", default="tenant-local", help="tenant to seed under")
    parser.add_argument("--query", default="how quickly must a sev-1 be escalated")
    args = parser.parse_args()

    platform = build_platform(PragSettings())

    total = 0
    for document_id, content in SEED_DOCUMENTS.items():
        indexed = await platform.ingest_markdown(
            content, document_id=document_id, tenant_id=args.tenant, source_id="kb.seed"
        )
        total += indexed
        print(f"  indexed {indexed:3d} chunks from {document_id}")

    print(f"\nSeeded {total} chunks for tenant {args.tenant!r}.\n")

    # Run one query end to end, so the script proves the system works rather than only that the
    # ingest half does. A seed that indexes successfully into an unqueryable index is the exact
    # failure this check exists to catch.
    from prag.api.http.app import _state_for

    headers = {"x-tenant-id": args.tenant, "x-user-id": "seed"}
    run = await platform.engine.run(_state_for(platform, args.query, headers))
    envelope = run.state.result

    print(f"Query: {args.query}")
    if envelope is None:
        print("  no envelope produced")
        return 1
    if envelope.abstained:
        print(f"  abstained: {envelope.abstention.reason_code if envelope.abstention else '?'}")
        return 1

    print(f"  answer: {envelope.answer}")
    print(
        f"  citations: {len(envelope.citations)} | groundedness: "
        f"{envelope.grounding.groundedness:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
