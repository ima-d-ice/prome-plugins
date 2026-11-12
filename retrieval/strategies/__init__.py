"""Retrieval strategies of the Prome RAG ecosystem.

Every strategy answers with the shared
:class:`~retrieval.contracts.models.RetrievalResponse`, so one strategy can be
swapped for another without touching the caller.
"""

from retrieval.strategies.hierarchical import (
    ChildStage,
    ChildStageResult,
    HierarchicalConfig,
    HierarchicalIndexConfig,
    HierarchicalIndexer,
    HierarchicalRetrievalStrategy,
    ParentStage,
    ParentStageResult,
)

__all__ = [
    "ChildStage",
    "ChildStageResult",
    "HierarchicalConfig",
    "HierarchicalIndexConfig",
    "HierarchicalIndexer",
    "HierarchicalRetrievalStrategy",
    "ParentStage",
    "ParentStageResult",
]
