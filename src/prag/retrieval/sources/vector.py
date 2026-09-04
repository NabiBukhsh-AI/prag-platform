"""Dense retrieval as a ``KnowledgeSource``.

This is where parent-child retrieval pays off. The vector matched is the child's — precise,
about one thing — and the candidate that comes back carries the parent as its context text. The
model reads a coherent section; the index matched a specific paragraph. Neither had to
compromise on size for the other's benefit.

Two obligations are enforced here rather than trusted to a caller. The ACL filter is built from
the principal and applied *at the store*, so content the caller may not see is never fetched.
And the deadline is honoured by truncating rather than overrunning, because a leg that spends
200 ms it did not have has taken that time from the reranker or the model.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from prag.core.errors import DeadlineExceeded, SourceUnavailable
from prag.core.models.common import (
    EmbeddingPurpose,
    HealthState,
    HealthStatus,
    SourceCapabilities,
    VolatilityClass,
)
from prag.core.models.retrieval import (
    Candidate,
    ChunkMetadata,
    LegResult,
    LegStatus,
)
from prag.ingestion.indexing import payload_to_fields
from prag.storage.vectorstore.filters import acl_filter

if TYPE_CHECKING:
    from prag.core.models.common import Deadline
    from prag.core.models.identity import Principal
    from prag.core.models.retrieval import RetrievalLeg
    from prag.core.protocols.evidence import EmbeddingProvider
    from prag.core.protocols.retrieval import VectorStore

__all__ = ["VectorKnowledgeSource"]

#: Reserved for the embedding call, out of the leg's deadline. Embedding must finish with time
#: left for the search, or the leg spends its whole budget preparing to do the work.
_EMBED_SHARE = 0.4


class VectorKnowledgeSource:
    """Dense retrieval over one vector collection."""

    def __init__(
        self,
        *,
        source_id: str,
        store: VectorStore,
        embedder: EmbeddingProvider,
        collection: str,
        max_top_k: int = 100,
    ) -> None:
        self.source_id = source_id
        self.capabilities = SourceCapabilities(
            supports_filters=True,
            supports_vectors=True,
            # Dense retrieval matches meaning, not terms. Claiming text support would let the
            # planner route identifier-heavy queries here, where it reliably returns plausible
            # neighbours instead of the exact match a lexical index would have found.
            supports_text=False,
            supports_traversal=False,
            max_top_k=max_top_k,
        )
        self._store = store
        self._embedder = embedder
        self._collection = collection

    async def retrieve(
        self,
        leg: RetrievalLeg,
        principal: Principal,
        deadline: Deadline,
    ) -> LegResult:
        started = deadline.elapsed_ms
        query = leg.query_text
        if not query.strip():
            raise SourceUnavailable(
                "vector retrieval requires query text on the leg",
                source_id=self.source_id,
                leg_id=leg.leg_id,
            )

        # The principal's filter is merged over whatever the leg asked for, never under it. A
        # leg cannot widen its own ACL scope by supplying a competing tenant_id.
        filters = {
            **leg.filters,
            **acl_filter(principal.tenant_id, principal.acl_hashes),
        }

        # Running out of time is not the source failing. Wrapping a deadline breach as
        # SourceUnavailable would open the circuit breaker on a perfectly healthy source, and
        # a slow request would then take that source down for every other request too.
        if deadline.expired:
            return self._out_of_time(leg, started, deadline)

        try:
            vectors = await self._embedder.embed(
                [query],
                EmbeddingPurpose.QUERY,
                deadline.share(_EMBED_SHARE, label=f"{leg.leg_id}.embed"),
            )
        except DeadlineExceeded:
            return self._out_of_time(leg, started, deadline)
        except Exception as exc:
            raise SourceUnavailable(
                "embedding failed", source_id=self.source_id, leg_id=leg.leg_id
            ) from exc

        if deadline.expired:
            return self._out_of_time(leg, started, deadline)

        try:
            hits = await self._store.search(
                self._collection,
                vectors[0],
                min(leg.top_k, self.capabilities.max_top_k),
                filters,
                deadline,
            )
        except DeadlineExceeded:
            return self._out_of_time(leg, started, deadline)
        except Exception as exc:
            raise SourceUnavailable(
                "vector search failed", source_id=self.source_id, leg_id=leg.leg_id
            ) from exc

        candidates = tuple(
            self._to_candidate(hit, leg_id=leg.leg_id, rank=rank) for rank, hit in enumerate(hits)
        )
        return LegResult(
            leg_id=leg.leg_id,
            source_id=self.source_id,
            status=LegStatus.OK,
            candidates=candidates,
            latency_ms=int(deadline.elapsed_ms - started),
        )

    def _out_of_time(self, leg: RetrievalLeg, started: float, deadline: Deadline) -> LegResult:
        """Report a truncated leg.

        PARTIAL rather than FAILED, and usable: the plan may have other legs that did return,
        and a coverage warning on a thinner answer beats failing the whole request.
        """
        return LegResult(
            leg_id=leg.leg_id,
            source_id=self.source_id,
            status=LegStatus.PARTIAL,
            latency_ms=int(deadline.elapsed_ms - started),
            error_reason_code="deadline_exceeded",
        )

    def _to_candidate(self, hit, *, leg_id: str, rank: int) -> Candidate:
        """Map a store hit into a domain candidate.

        The vendor type stops here. Nothing outside this method knows what a ``ScoredPoint``
        is, which is what makes swapping pgvector for Qdrant an adapter change.
        """
        fields = payload_to_fields(hit.payload)
        now = int(hit.payload.get("updated_at_ms", 0)) or 0

        return Candidate(
            candidate_id=f"{self.source_id}:{hit.point_id}",
            chunk_id=fields["chunk_id"] or hit.point_id,
            document_id=fields["document_id"],
            document_version=fields["document_version"],
            source_id=self.source_id,
            text=fields["text"],
            # The child matched; the parent is what the model will read.
            parent_text=fields["parent_text"],
            raw_score=hit.score,
            rank_by_leg={leg_id: rank},
            metadata=ChunkMetadata(
                authority=fields["authority"],
                created_at_ms=now,
                updated_at_ms=now,
                acl_hash=fields["acl_hash"],
                volatility_class=_volatility(hit.payload.get("volatility_class")),
                lineage_root=fields["lineage_root"],
                title=" > ".join(fields["heading_path"]) or None,
                embedding_version=fields["embedding_version"],
            ),
        )

    async def health(self) -> HealthStatus:
        """Report on the backing collection.

        An empty collection is ``DEGRADED`` rather than healthy. It answers every query with
        nothing, which from the outside is indistinguishable from a corpus that genuinely has no
        match — and that ambiguity is exactly what makes a failed seed hard to notice.
        """
        checked = int(time.time() * 1000)
        try:
            info = await self._store.collection_info(self._collection)
        except Exception as exc:
            return HealthStatus(
                state=HealthState.UNAVAILABLE, checked_at_ms=checked, detail=str(exc)
            )

        if info.vector_count == 0:
            return HealthStatus(
                state=HealthState.DEGRADED,
                checked_at_ms=checked,
                detail=f"collection {self._collection!r} is empty",
            )
        return HealthStatus(state=HealthState.HEALTHY, checked_at_ms=checked)


def _volatility(raw: object) -> VolatilityClass:
    try:
        return VolatilityClass(str(raw))
    except ValueError:
        return VolatilityClass.SLOW
