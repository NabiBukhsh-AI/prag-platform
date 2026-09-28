"""BM25 lexical retrieval as a ``KnowledgeSource``.

It exists because dense retrieval reliably fails on exactly the queries an enterprise corpus is
full of. An embedding places "sev-1" near "incident severity" and near "outage" — useful — but it
also places "sev-1" near "sev-2", and it has no way to prefer an exact identifier match over a
semantically adjacent one. Asked for error code `E4011`, a dense index confidently returns
`E4012`.

BM25 has the opposite failure: it cannot match meaning at all. Neither is a substitute for the
other, and running both is not redundancy — it is the two halves of a working retriever. The
rank fusion that combines them needs no score calibration precisely because their scores are
incommensurable.

Implemented in-process over the same records the vector store holds. OpenSearch is the target at
scale, and the protocol is what makes that a config change.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from typing import TYPE_CHECKING, Any

from prag.core.errors import DeadlineExceeded, SourceUnavailable
from prag.core.models.common import (
    HealthState,
    HealthStatus,
    SourceCapabilities,
    VolatilityClass,
)
from prag.core.models.retrieval import Candidate, ChunkMetadata, LegResult, LegStatus
from prag.ingestion.indexing import payload_to_fields
from prag.storage.vectorstore.filters import acl_filter, matches

if TYPE_CHECKING:
    from collections.abc import Sequence

    from prag.core.models.common import Deadline
    from prag.core.models.identity import Principal
    from prag.core.models.retrieval import RetrievalLeg

__all__ = ["InMemoryLexicalIndex", "LexicalKnowledgeSource"]

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-_.]*")

#: BM25 term-frequency saturation. Above it, more repetitions of a term stop adding much — which
#: is what stops a document that says "escalation" forty times from beating one that answers the
#: question once.
_K1 = 1.2
#: Length normalisation strength. 0.75 is the conventional value: it discounts long documents
#: without punishing them so hard that a thorough section loses to a one-line stub.
_B = 0.75


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens, keeping hyphens and dots.

    Keeping them is the point. Splitting on them would turn ``sev-1`` into ``sev`` and ``1``,
    and ``E4011.2`` into three meaningless fragments — destroying exactly the identifiers this
    source exists to match.
    """
    return _TOKEN.findall(text.lower())


class InMemoryLexicalIndex:
    """A BM25 index over chunk payloads.

    In-process and exact. At seed-corpus scale that is faster than building an inverted index,
    and it keeps the local stack free of another service. OpenSearch replaces it behind the
    same source, and the conformance suite is what keeps the two honest about their contract.
    """

    def __init__(self) -> None:
        self._documents: dict[str, dict[str, Any]] = {}
        self._tokens: dict[str, list[str]] = {}
        self._frequencies: dict[str, Counter[str]] = {}
        #: Documents containing each term, for inverse document frequency.
        self._document_frequency: Counter[str] = Counter()
        self._total_length = 0

    def index(self, documents: Sequence[dict[str, Any]]) -> int:
        """Index or replace documents, returning how many were written."""
        for payload in documents:
            doc_id = str(payload.get("chunk_id") or payload.get("point_id") or "")
            if not doc_id:
                continue

            if doc_id in self._tokens:
                self._remove(doc_id)

            # The heading path is indexed with the body, for the same reason it is embedded with
            # it: two sections can both say "30 days", and only the heading tells them apart.
            heading = " ".join(payload.get("heading_path", ()) or ())
            tokens = _tokenize(f"{heading} {payload.get('text', '')}")

            self._documents[doc_id] = dict(payload)
            self._tokens[doc_id] = tokens
            self._frequencies[doc_id] = Counter(tokens)
            self._document_frequency.update(set(tokens))
            self._total_length += len(tokens)

        return len(documents)

    def _remove(self, doc_id: str) -> None:
        tokens = self._tokens.pop(doc_id, [])
        self._document_frequency.subtract(set(tokens))
        self._document_frequency += Counter()  # drop zero and negative counts
        self._total_length -= len(tokens)
        self._frequencies.pop(doc_id, None)
        self._documents.pop(doc_id, None)

    def delete(self, doc_ids: Sequence[str]) -> None:
        for doc_id in doc_ids:
            self._remove(doc_id)

    def search(
        self, query: str, top_k: int, filters: dict[str, Any] | None
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """BM25 over the filtered subset.

        Filters are applied before scoring, for the same reason they are in the vector store:
        scoring content the caller may not see means fetching it, and fetched data can leak
        through a log line or a timing difference even when it never reaches the response.
        """
        if not self._documents:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        candidates = [
            doc_id for doc_id, payload in self._documents.items() if matches(payload, filters)
        ]
        if not candidates:
            return []

        average_length = self._total_length / max(1, len(self._tokens))
        total_docs = len(self._tokens)
        scored: list[tuple[str, float, dict[str, Any]]] = []

        for doc_id in candidates:
            frequencies = self._frequencies[doc_id]
            length = len(self._tokens[doc_id])
            score = 0.0

            for term in query_tokens:
                term_frequency = frequencies.get(term, 0)
                if term_frequency == 0:
                    continue
                containing = self._document_frequency.get(term, 0)
                # Smoothed IDF, which stays positive for a term present in every document. The
                # unsmoothed form goes negative there and would let a common term *subtract*
                # from a document's score, ranking a document below one that lacks the term.
                idf = math.log(1 + (total_docs - containing + 0.5) / (containing + 0.5))
                denominator = term_frequency + _K1 * (
                    1 - _B + _B * length / max(1e-9, average_length)
                )
                score += idf * (term_frequency * (_K1 + 1)) / denominator

            if score > 0.0:
                scored.append((doc_id, score, self._documents[doc_id]))

        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_k]

    @property
    def size(self) -> int:
        return len(self._documents)


class LexicalKnowledgeSource:
    """Term-matching retrieval over an in-process BM25 index."""

    def __init__(
        self,
        *,
        source_id: str = "lexical.primary",
        index: InMemoryLexicalIndex | None = None,
        max_top_k: int = 100,
    ) -> None:
        self.source_id = source_id
        self.capabilities = SourceCapabilities(
            supports_filters=True,
            # Text, not vectors. Declaring it honestly is what lets the planner send this source
            # the expanded variant and the dense source the rewritten one — the two want
            # opposite things from the same query.
            supports_vectors=False,
            supports_text=True,
            supports_traversal=False,
            max_top_k=max_top_k,
        )
        self.index = index or InMemoryLexicalIndex()

    async def retrieve(
        self, leg: RetrievalLeg, principal: Principal, deadline: Deadline
    ) -> LegResult:
        started = deadline.elapsed_ms

        if not leg.query_text.strip():
            raise SourceUnavailable(
                "lexical retrieval requires query text on the leg",
                source_id=self.source_id,
                leg_id=leg.leg_id,
            )

        if deadline.expired:
            return LegResult(
                leg_id=leg.leg_id,
                source_id=self.source_id,
                status=LegStatus.PARTIAL,
                latency_ms=int(deadline.elapsed_ms - started),
                error_reason_code="deadline_exceeded",
            )

        filters = {**leg.filters, **acl_filter(principal.tenant_id, principal.acl_hashes)}

        try:
            hits = self.index.search(
                leg.query_text, min(leg.top_k, self.capabilities.max_top_k), filters
            )
        except DeadlineExceeded:
            return LegResult(
                leg_id=leg.leg_id,
                source_id=self.source_id,
                status=LegStatus.PARTIAL,
                latency_ms=int(deadline.elapsed_ms - started),
                error_reason_code="deadline_exceeded",
            )
        except Exception as exc:
            raise SourceUnavailable(
                "lexical search failed", source_id=self.source_id, leg_id=leg.leg_id
            ) from exc

        return LegResult(
            leg_id=leg.leg_id,
            source_id=self.source_id,
            status=LegStatus.OK,
            candidates=tuple(
                self._to_candidate(doc_id, score, payload, leg_id=leg.leg_id, rank=rank)
                for rank, (doc_id, score, payload) in enumerate(hits)
            ),
            latency_ms=int(deadline.elapsed_ms - started),
        )

    def _to_candidate(
        self,
        doc_id: str,
        score: float,
        payload: dict[str, Any],
        *,
        leg_id: str,
        rank: int,
    ) -> Candidate:
        fields = payload_to_fields(payload)
        stamp = int(payload.get("updated_at_ms", 0)) or 0

        return Candidate(
            candidate_id=f"{self.source_id}:{doc_id}",
            chunk_id=fields["chunk_id"] or doc_id,
            document_id=fields["document_id"],
            document_version=fields["document_version"],
            source_id=self.source_id,
            text=fields["text"],
            parent_text=fields["parent_text"],
            raw_score=score,
            rank_by_leg={leg_id: rank},
            metadata=ChunkMetadata(
                authority=fields["authority"],
                created_at_ms=stamp,
                updated_at_ms=stamp,
                acl_hash=fields["acl_hash"],
                tenant_id=fields["tenant_id"],
                volatility_class=_volatility(payload.get("volatility_class")),
                lineage_root=fields["lineage_root"],
                title=" > ".join(fields["heading_path"]) or None,
                embedding_version=fields["embedding_version"],
            ),
        )

    async def health(self) -> HealthStatus:
        checked = int(time.time() * 1000)
        if self.index.size == 0:
            # Empty answers everything with nothing, which is indistinguishable from a corpus
            # with no match — the ambiguity that makes a failed seed hard to notice.
            return HealthStatus(
                state=HealthState.DEGRADED,
                checked_at_ms=checked,
                detail="lexical index is empty",
            )
        return HealthStatus(state=HealthState.HEALTHY, checked_at_ms=checked)


def _volatility(raw: object) -> VolatilityClass:
    try:
        return VolatilityClass(str(raw))
    except ValueError:
        return VolatilityClass.SLOW
