"""Every cross-boundary interface in the platform.

All of them are ``typing.Protocol``: structurally typed, async where I/O is involved, and free
of vendor types. Structural typing rather than inheritance means an implementation does not
import the protocol to satisfy it, so a subsystem never depends on ``core`` at runtime merely to
declare that it conforms.

Every implementation must pass the shared conformance suite in ``tests/contract/``. That suite
is parameterized over the registered implementations, so adding a new ``KnowledgeSource``
without making it pass fails CI. This is the mechanism that keeps provider abstraction honest
instead of aspirational — an abstraction verified only by one implementation is a description of
that implementation.

**Add the protocol here first, then implement it elsewhere.** A protocol written after its first
implementation describes what that implementation happened to do.
"""

from prag.core.protocols.crosscutting import (
    CacheTier,
    Evaluator,
    GraphNode,
    Guardrail,
    MemoryStore,
)
from prag.core.protocols.evidence import (
    ContextBuilder,
    ContextValidator,
    EmbeddingProvider,
    EvidenceGrouper,
    PromptRenderer,
    Reranker,
)
from prag.core.protocols.fusion import ConfidenceCalibrator, FusionPolicy
from prag.core.protocols.generation import GroundingVerifier, LLMProvider, ModelRouter
from prag.core.protocols.ingestion import Chunker, Extractor, TokenCounter
from prag.core.protocols.intelligence import (
    QueryAnalyzer,
    QueryTransformer,
    StrategyRouter,
)
from prag.core.protocols.parametric import (
    AdapterSelector,
    AdapterStore,
    ParametricEligibilityGate,
)
from prag.core.protocols.retrieval import (
    KnowledgeSource,
    LexicalStore,
    RetrievalOrchestrator,
    VectorStore,
)
from prag.core.protocols.storage import (
    AdapterRepository,
    EvalRepository,
    KnowledgeRepository,
    LineageRepository,
)

__all__ = [
    "AdapterRepository",
    "AdapterSelector",
    "AdapterStore",
    "CacheTier",
    "Chunker",
    "ConfidenceCalibrator",
    "ContextBuilder",
    "ContextValidator",
    "EmbeddingProvider",
    "EvalRepository",
    "Evaluator",
    "EvidenceGrouper",
    "Extractor",
    "FusionPolicy",
    "GraphNode",
    "GroundingVerifier",
    "Guardrail",
    "KnowledgeRepository",
    "KnowledgeSource",
    "LLMProvider",
    "LexicalStore",
    "LineageRepository",
    "MemoryStore",
    "ModelRouter",
    "ParametricEligibilityGate",
    "PromptRenderer",
    "QueryAnalyzer",
    "QueryTransformer",
    "Reranker",
    "RetrievalOrchestrator",
    "StrategyRouter",
    "TokenCounter",
    "VectorStore",
]
