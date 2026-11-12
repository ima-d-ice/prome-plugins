"""Prome RAG Vector Store Ecosystem.

Backend independent contracts (:mod:`vector_stores.contracts`) plus the
adapter implementations.  The first adapter is Qdrant
(:mod:`vector_stores.qdrant`).

The contracts are dependency free: importing this package never imports a
vector database client.
"""

from vector_stores.contracts import (
    CollectionConfig,
    DistanceMetric,
    FilterOp,
    IndexConfig,
    IndexKind,
    MetadataFilter,
    PayloadFieldType,
    PayloadIndexSpec,
    SearchRequest,
    SearchResult,
    SpaceVectors,
    StoreCapabilities,
    StoredRecord,
    VectorSpaceConfig,
    VectorStore,
)

__all__ = [
    "CollectionConfig",
    "DistanceMetric",
    "FilterOp",
    "IndexConfig",
    "IndexKind",
    "MetadataFilter",
    "PayloadFieldType",
    "PayloadIndexSpec",
    "SearchRequest",
    "SearchResult",
    "SpaceVectors",
    "StoreCapabilities",
    "StoredRecord",
    "VectorSpaceConfig",
    "VectorStore",
]
