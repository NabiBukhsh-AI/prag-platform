"""Running evaluation cases against a live platform.

Here rather than in ``prag.evaluation`` because it needs the platform, and only the composition
root may see the platform. The evaluation package stays importable by anything — including an
online sampler running out of process — because it knows nothing about how a sample was made.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from prag.core.errors import AbstentionRequired, GuardrailBlocked, IsolationViolation
from prag.evaluation import sample_from_state

if TYPE_CHECKING:
    from prag.api.composition import Platform
    from prag.core.models.events import EvalSample
    from prag.core.models.state import RequestState
    from prag.evaluation import GoldenCase, Outcome

__all__ = ["ingest_corpus", "run_case"]


async def ingest_corpus(platform: Platform, corpus_dir: str | Path) -> int:
    """Ingest every ``{tenant}/{acl_hash}/{document_id}.md`` under ``corpus_dir``.

    Ownership and access come from the path rather than from front matter, so a fixture cannot
    claim a tenant its location does not give it.
    """
    total = 0
    for path in sorted(Path(corpus_dir).glob("*/*/*.md")):
        tenant_id, acl_hash = path.parent.parent.name, path.parent.name
        total += await platform.ingest_markdown(
            path.read_text(encoding="utf-8"),
            document_id=path.stem,
            tenant_id=tenant_id,
            source_id=f"kb.{tenant_id}",
            acl_hash=acl_hash,
        )
    return total


async def run_case(platform: Platform, case: GoldenCase) -> EvalSample:
    """Run one case through the same path the HTTP endpoint takes, and sample the result.

    Raised outcomes are captured rather than propagated: an adversarial case that ends in a
    guardrail block or an isolation violation has *passed* if that was the expected outcome.
    """
    headers = {
        "x-tenant-id": case.tenant_id,
        "x-user-id": "eval",
        "x-acl-hashes": ",".join(case.acl_hashes),
    }
    state: RequestState | None = None
    outcome: Outcome
    try:
        run = await platform.answer(platform.request_state(case.query, headers))
        state = run.state
        outcome = "abstained" if state.result is None or state.result.abstained else "answered"
    except AbstentionRequired:
        outcome = "abstained"
    except GuardrailBlocked:
        outcome = "blocked"
    except IsolationViolation:
        outcome = "isolation_violation"

    return sample_from_state(case, outcome=outcome, state=state)
