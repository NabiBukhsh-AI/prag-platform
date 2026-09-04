"""Plan construction and execution over knowledge sources.

Sources are registered from config and read through the ``KnowledgeSource`` protocol, so the
planner and orchestrator branch on declared capabilities rather than on source ids. Adding a
source is a config entry plus an implementation.
"""

from prag.retrieval.sources import VectorKnowledgeSource

__all__ = ["VectorKnowledgeSource"]
