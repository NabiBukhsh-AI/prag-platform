"""Token estimation.

The heuristic default deliberately avoids a tokenizer dependency in the ingestion path. The
budget it feeds is itself approximate — an evidence cap of 8000 tokens is a policy choice, not a
measurement — so paying a model-loading cost and a per-call tokenization cost to be exactly
right about an approximate number is a poor trade at ingest scale.

Where exactness matters, at the context-packing boundary against a specific model, a real
tokenizer is registered through the ``TokenCounter`` protocol and nothing else changes.
"""

from __future__ import annotations

import re

__all__ = ["HeuristicTokenCounter", "estimate_tokens"]

#: Roughly the observed characters-per-token ratio for English prose across common BPE
#: vocabularies. Code and non-Latin scripts run denser, which is why the estimator adjusts
#: rather than applying one ratio everywhere.
_CHARS_PER_TOKEN_PROSE = 4.0
_CHARS_PER_TOKEN_DENSE = 2.8

#: Characters that tokenize poorly: punctuation runs, symbols, and CJK, which is close to one
#: token per character in most vocabularies.
_DENSE_PATTERN = re.compile(r"[^\sA-Za-z0-9]|[　-鿿]")


def estimate_tokens(text: str) -> int:
    """Estimate the token count of a string.

    Biased slightly high for dense text. Under-estimating is the more expensive error: it
    overfills the context window, which either truncates evidence silently or fails the request
    at the provider, whereas over-estimating just leaves a little headroom unused.
    """
    if not text:
        return 0

    dense_chars = len(_DENSE_PATTERN.findall(text))
    plain_chars = len(text) - dense_chars

    estimate = plain_chars / _CHARS_PER_TOKEN_PROSE + dense_chars / _CHARS_PER_TOKEN_DENSE
    return max(1, round(estimate))


class HeuristicTokenCounter:
    """The default ``TokenCounter``: character-class based, no model required."""

    def count(self, text: str) -> int:
        return estimate_tokens(text)
