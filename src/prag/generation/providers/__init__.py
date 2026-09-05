"""``LLMProvider`` adapters, one module per backend.

No provider type, exception, or token-counting quirk escapes an adapter. If a caller can tell
which backend served a request, the abstraction has failed and swapping vLLM for a hosted API
stops being a config change.
"""

from prag.generation.providers.local import LocalExtractiveProvider

__all__ = ["LocalExtractiveProvider"]
