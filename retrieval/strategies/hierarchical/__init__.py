"""Hierarchical (coarse-to-fine) retrieval.

The strategy selects the parent sections that look relevant to a query and
then retrieves the evidence children *inside those sections only*, so the fine
stage never has to consider the whole corpus.

Both levels live in the same collection and the same named dense vector
space; the ``chunk_level`` metadata field tells them apart.
"""

from retrieval.strategies.hierarchical.child import ChildStage, ChildStageResult
from retrieval.strategies.hierarchical.config import (
    HierarchicalConfig,
    HierarchicalIndexConfig,
)
from retrieval.strategies.hierarchical.indexer import (
    HierarchicalIndexer,
    IndexReport,
    build_collection_config,
    validate_hierarchy,
)
from retrieval.strategies.hierarchical.metadata import (
    CHILD_COUNT_FIELD,
    CHILD_IDS_FIELD,
    CHILD_INDEX_FIELD,
    CHUNK_INDEX_FIELD,
    CHUNK_LEVEL_FIELD,
    DOCUMENT_ID_FIELD,
    HEADING_PATH_FIELD,
    LEVEL_CHILD,
    LEVEL_PARENT,
    PARENT_ID_FIELD,
    SOURCE_FIELD,
    child_filter,
    child_filter_for_parent,
    parent_filter,
    record_metadata,
)
from retrieval.strategies.hierarchical.parent import ParentStage, ParentStageResult
from retrieval.strategies.hierarchical.strategy import HierarchicalRetrievalStrategy

__all__ = [
    "CHILD_COUNT_FIELD",
    "CHILD_IDS_FIELD",
    "CHILD_INDEX_FIELD",
    "CHUNK_INDEX_FIELD",
    "CHUNK_LEVEL_FIELD",
    "DOCUMENT_ID_FIELD",
    "HEADING_PATH_FIELD",
    "LEVEL_CHILD",
    "LEVEL_PARENT",
    "PARENT_ID_FIELD",
    "SOURCE_FIELD",
    "ChildStage",
    "ChildStageResult",
    "HierarchicalConfig",
    "HierarchicalIndexConfig",
    "HierarchicalIndexer",
    "HierarchicalRetrievalStrategy",
    "IndexReport",
    "ParentStage",
    "ParentStageResult",
    "build_collection_config",
    "child_filter",
    "child_filter_for_parent",
    "parent_filter",
    "record_metadata",
    "validate_hierarchy",
]
