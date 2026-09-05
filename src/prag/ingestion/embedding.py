"""A hashing ``EmbeddingProvider`` for the local stack.

Deterministic, model-free, and dependency-free. It exists so that ``docker compose up`` plus a
seed script yields a working system on a laptop — no model download, no API key, no GPU.

It is not a semantic embedder. Hashed token vectors capture lexical overlap and nothing else:
"the escalation window" and "how fast do we page someone" share no tokens and will not match.
That limitation is stated plainly because the alternative — quietly shipping a bad embedder
that looks like a good one — makes retrieval quality untraceable. Register a real provider for
anything that needs meaning; this one is for making the plumbing runnable and testable.

Determinism is the property worth having here. The same text embeds identically across runs and
processes, which is what recorded-state replay and every retrieval assertion depend on.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from prag.core.models.common import Deadline, EmbeddingPurpose

__all__ = ["HashingEmbeddingProvider"]

_WORD = re.compile(r"[a-z0-9][a-z0-9\-_.]*")

#: Character n-gram length. Four is short enough to survive inflection and long enough to
#: avoid matching unrelated words on a shared prefix.
_NGRAM_SIZE = 4
#: An n-gram match is weaker evidence than a whole-word match, and is weighted accordingly.
_NGRAM_WEIGHT = 0.35
#: Tokens at or below this length are their own best signal; splitting them produces grams
#: that match almost anything.
_MIN_GRAM_TOKEN = 5


class HashingEmbeddingProvider:
    """Bag-of-hashed-tokens embeddings, L2 normalized.

    Each content token is hashed into a bucket and accumulated, so two texts sharing tokens
    point in similar directions. Crude, but it makes cosine similarity behave the way retrieval
    expects — which is enough to exercise every stage around it.
    """

    def __init__(
        self,
        *,
        dimensions: int = 256,
        model_id: str = "local.hashing",
        model_version: str = "v1",
    ) -> None:
        if dimensions <= 0:
            raise ValueError(f"dimensions must be positive, got {dimensions}")
        self.model_id = model_id
        self.model_version = model_version
        self.dimensions = dimensions
        self.embed_calls = 0
        self.largest_batch = 0

    async def embed(
        self,
        texts: Sequence[str],
        purpose: EmbeddingPurpose,
        deadline: Deadline,
    ) -> Sequence[Sequence[float]]:
        self.embed_calls += 1
        self.largest_batch = max(self.largest_batch, len(texts))
        deadline.raise_if_expired()
        return [self._vector(text, purpose) for text in texts]

    def _vector(self, text: str, purpose: EmbeddingPurpose) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = _WORD.findall(text.lower())

        for token in tokens:
            bucket, sign = self._bucket(token)
            vector[bucket] += sign

            # Character n-grams alongside the whole token, so morphological variants share
            # buckets: "escalate", "escalation" and "escalated" overlap on their stem instead of
            # being three unrelated hashes. Without this, a question phrased with one inflection
            # misses a document phrased with another — which is the single most common way a
            # purely lexical matcher fails on questions people actually ask.
            #
            # Weighted below whole tokens, because an n-gram match is weaker evidence than a
            # word match and should not outvote one.
            for gram in self._char_grams(token):
                gram_bucket, gram_sign = self._bucket(gram)
                vector[gram_bucket] += gram_sign * _NGRAM_WEIGHT

        # A small purpose-dependent component, so query and document embeddings of identical
        # text differ without becoming unrelated. Real asymmetric models behave this way, and a
        # provider that ignored purpose entirely would let a caller pass the wrong one for free
        # and never learn it had.
        marker, marker_sign = self._bucket(f"__{purpose}__")
        vector[marker] += marker_sign * 0.1

        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            # Empty or punctuation-only text. A zero vector scores zero against everything,
            # which ranks it last rather than failing the batch it was part of.
            return vector
        return [v / norm for v in vector]

    @staticmethod
    def _char_grams(token: str) -> tuple[str, ...]:
        """Overlapping character n-grams of a token, or nothing for a short one."""
        if len(token) < _MIN_GRAM_TOKEN:
            return ()
        return tuple(token[i : i + _NGRAM_SIZE] for i in range(len(token) - _NGRAM_SIZE + 1))

    def _bucket(self, token: str) -> tuple[int, float]:
        """Map a token to a bucket and a sign.

        The sign is a second hash bit, and it matters: without it every token adds positively
        and unrelated texts drift toward a common direction, so everything looks mildly similar
        to everything. Signed hashing keeps unrelated vectors near-orthogonal.
        """
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dimensions, 1.0 if (value >> 63) & 1 else -1.0
