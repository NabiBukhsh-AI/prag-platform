"""Model routing, providers, streaming, grounding, and tools.

Two rules carry this package. **Temperature 0 for anything evidence-grounded** — sampling on an
extraction task buys nothing and costs faithfulness. And **citations are attached only after an
entailment check** — a plausible citation on an unsupported claim converts an unverified
statement into an apparently verified one, which is worse than leaving it uncited.
"""

from prag.generation.grounding import (
    Claim,
    HeuristicGroundingVerifier,
    cited_markers,
    entailment_score,
    extract_claims,
    strip_invalid_markers,
)
from prag.generation.profiles import (
    DEFAULT_PROFILES,
    FallbackChain,
    GenerationProfile,
    ModelOption,
    PolicyModelRouter,
    profile_for,
)
from prag.generation.providers import LocalExtractiveProvider
from prag.generation.streaming import (
    CorrectionEvent,
    CorrectionKind,
    SentenceBuffer,
    StreamEvent,
    buffered_stream,
)

__all__ = [
    "DEFAULT_PROFILES",
    "Claim",
    "CorrectionEvent",
    "CorrectionKind",
    "FallbackChain",
    "GenerationProfile",
    "HeuristicGroundingVerifier",
    "LocalExtractiveProvider",
    "ModelOption",
    "PolicyModelRouter",
    "SentenceBuffer",
    "StreamEvent",
    "buffered_stream",
    "cited_markers",
    "entailment_score",
    "extract_claims",
    "profile_for",
    "strip_invalid_markers",
]
