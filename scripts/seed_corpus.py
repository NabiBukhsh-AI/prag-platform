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

ROOT = Path(__file__).resolve().parent.parent

#: The clean seed documents, shared with the evaluation corpus so the quickstart and the golden
#: set answer from the same text. Listed rather than globbed: the corpus also holds adversarial
#: fixtures that have no place in a quickstart.
SEED_DIR = ROOT / "eval" / "seed" / "corpus" / "tenant-local" / "public"
SEED_DOCUMENTS: dict[str, str] = {
    name: (SEED_DIR / f"{name}.md").read_text(encoding="utf-8")
    for name in ("runbook.incident-response", "policy.access-control")
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
    headers = {"x-tenant-id": args.tenant, "x-user-id": "seed"}
    run = await platform.answer(platform.request_state(args.query, headers))
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
