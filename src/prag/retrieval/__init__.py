"""Plan construction and execution over knowledge sources.

Sources are registered from config and read through the ``KnowledgeSource`` protocol, so the
planner and orchestrator branch on declared capabilities rather than on source ids. Adding a
source is a config entry plus an implementation.
"""

from prag.retrieval.fusion import fuse_legs, reciprocal_rank_fusion
from prag.retrieval.orchestrator import CircuitBreaker, ParallelRetrievalOrchestrator
from prag.retrieval.planner import SourcePlanner, SourceSpec, build_plan
from prag.retrieval.sources import (
    InMemoryLexicalIndex,
    LexicalKnowledgeSource,
    VectorKnowledgeSource,
)

__all__ = [
    "CircuitBreaker",
    "InMemoryLexicalIndex",
    "LexicalKnowledgeSource",
    "ParallelRetrievalOrchestrator",
    "SourcePlanner",
    "SourceSpec",
    "VectorKnowledgeSource",
    "build_plan",
    "fuse_legs",
    "reciprocal_rank_fusion",
]
