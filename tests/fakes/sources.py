"""An in-memory ``KnowledgeSource``."""

from __future__ import annotations

import time
from collections.abc import Sequence

from prag.core.errors import SourceUnavailable
from prag.core.models.common import (
    Deadline,
    HealthState,
    HealthStatus,
    SourceCapabilities,
    VolatilityClass,
)
from prag.core.models.identity import Principal
from prag.core.models.retrieval import (
    Candidate,
    ChunkMetadata,
    LegResult,
    LegStatus,
    RetrievalLeg,
)

__all__ = ["InMemoryKnowledgeSource", "make_candidate"]


def make_candidate(
    text: str,
    *,
    candidate_id: str,
    document_id: str = "doc-1",
    acl_hash: str = "public",
    authority: float = 0.5,
    lineage_root: str | None = None,
    age_ms: int = 0,
    score: float = 1.0,
) -> Candidate:
    """Build a candidate for tests, with sensible defaults for the fields under test.

    ``lineage_root`` defaults to the document id, which is the common case: a chunk derives from
    its own document. Tests exercising the independence correction override it, because that is
    exactly what a syndicated document looks like — different document ids, one root.
    """
    now = int(time.time() * 1000)
    return Candidate(
        candidate_id=candidate_id,
        chunk_id=f"chunk-{candidate_id}",
        document_id=document_id,
        document_version="v1",
        source_id="fake.memory",
        text=text,
        raw_score=score,
        metadata=ChunkMetadata(
            authority=authority,
            created_at_ms=now - age_ms,
            updated_at_ms=now - age_ms,
            acl_hash=acl_hash,
            volatility_class=VolatilityClass.SLOW,
            lineage_root=lineage_root or document_id,
            embedding_version="fake-embed-v1",
        ),
    )


class InMemoryKnowledgeSource:
    """A knowledge source backed by a list, with controllable failure behaviour.

    Retrieval is naive substring matching. That is enough: the contract suite tests the
    *contract* — deadline honouring, ACL filtering, top-k truncation, partial results — not
    retrieval quality, which is what the evaluation suite is for.
    """

    def __init__(
        self,
        candidates: Sequence[Candidate] = (),
        *,
        source_id: str = "fake.memory",
        latency_ms: int = 0,
        fail: bool = False,
        health_state: HealthState = HealthState.HEALTHY,
    ) -> None:
        self.source_id = source_id
        self.capabilities = SourceCapabilities(
            supports_filters=True,
            supports_vectors=True,
            supports_text=True,
            max_top_k=100,
        )
        self._candidates = list(candidates)
        #: Simulated per-call latency, for exercising deadline behaviour without sleeping.
        self._latency_ms = latency_ms
        self._fail = fail
        self._health_state = health_state
        self.retrieve_calls = 0

    async def retrieve(
        self,
        leg: RetrievalLeg,
        principal: Principal,
        deadline: Deadline,
    ) -> LegResult:
        self.retrieve_calls += 1
        started = time.monotonic()

        if self._fail:
            raise SourceUnavailable("fake source configured to fail", source_id=self.source_id)

        # Report PARTIAL rather than raising when the work would not fit. Honouring a deadline
        # by truncating is correct behaviour, and it has to be distinguishable from both a clean
        # success and a failure or coverage warnings cannot be attributed accurately.
        truncated = self._latency_ms > deadline.remaining_ms

        # ACL filtering happens here, at the source, not downstream. Filtering late means
        # fetching data the caller may not see, and fetched data can leak through a log line or
        # a cache entry even when it never reaches the response.
        allowed = set(principal.acl_hashes) | {"public"}
        matched = [c for c in self._candidates if c.metadata.acl_hash in allowed]

        matched = matched[: max(1, leg.top_k // 2) if truncated else leg.top_k]

        elapsed = int((time.monotonic() - started) * 1000)
        return LegResult(
            leg_id=leg.leg_id,
            source_id=self.source_id,
            status=LegStatus.PARTIAL if truncated else LegStatus.OK,
            candidates=tuple(matched),
            latency_ms=elapsed,
        )

    async def health(self) -> HealthStatus:
        return HealthStatus(
            state=self._health_state,
            checked_at_ms=int(time.time() * 1000),
            latency_ms=0,
        )
