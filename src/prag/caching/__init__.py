"""Cache tiers, key construction, and invalidation.

Every correctness rule lives in key construction rather than at the call sites. A caching rule
that depends on every caller remembering it gets broken once and then stays broken invisibly,
and the symptom — a well-formed answer that is simply wrong for this request — is the hardest
kind to notice.
"""

from prag.caching.keys import (
    NEVER_CACHE_REASONS,
    analysis_key,
    embedding_key,
    exact_answer_key,
    is_cacheable,
    retrieval_key,
    ttl_for_volatility,
)

__all__ = [
    "NEVER_CACHE_REASONS",
    "analysis_key",
    "embedding_key",
    "exact_answer_key",
    "is_cacheable",
    "retrieval_key",
    "ttl_for_volatility",
]
