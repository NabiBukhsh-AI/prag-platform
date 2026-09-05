"""Dedup, fusion, reranking and selection — one pipeline, deliberately.

They ship together because each stage's output is the next one's only consumer, and a network
boundary between them would buy nothing while costing a serialization round trip inside the
retrieval budget.

Near-duplicates are linked rather than dropped. The link is what stops the agreement signal in
fusion from counting one fact several times, which is the failure that makes a system most
confident exactly where it is most wrong.
"""

from prag.evidence.dedup import group_candidates, normalized_fingerprint, token_overlap

__all__ = ["group_candidates", "normalized_fingerprint", "token_overlap"]
