"""Shared contracts for Prome RAG vector stores.

The contracts are backend independent: they describe collections, stored
records, vector spaces, metadata filters, index descriptions and the
:class:`~vector_stores.contracts.protocols.VectorStore` protocol every
adapter implements.

They reuse the upstream ecosystems instead of duplicating them:

- :class:`chunkers.contracts.models.Chunk` for the stored chunk (identity,
  text, metadata and provenance);
- :class:`embeddings.contracts.models.EmbeddingSpaceDescriptor` for the
  embedding-space identity of a vector space;
- :class:`embeddings.contracts.models.RepresentationType` for the vector
  kinds (dense / sparse / multi-vector).

Nothing in this package imports a vector database client.
"""

from vector_stores.contracts.capabilities import StoreCapabilities
from vector_stores.contracts.exceptions import (
    BackendConnectionError,
    CollectionExistsError,
    CollectionNotFoundError,
    DimensionMismatchError,
    EmbeddingSpaceMismatchError,
    InvalidRecordError,
    RecordNotFoundError,
    UnsupportedFilterOperationError,
    UnsupportedIndexOperationError,
    UnsupportedVectorTypeError,
    VectorStoreConfigError,
    VectorStoreDependencyError,
    VectorStoreError,
)
from vector_stores.contracts.filters import (
    FIELD_OPS,
    LOGICAL_OPS,
    FilterOp,
    MetadataFilter,
    and_,
    eq,
    exists,
    gt,
    gte,
    is_in,
    lt,
    lte,
    ne,
    not_,
    not_in,
    or_,
    validate_filter,
)
from vector_stores.contracts.indexes import (
    HnswIndexConfig,
    IndexConfig,
    IndexDescription,
    IndexKind,
    IndexKindCapability,
    PayloadFieldType,
    PayloadIndexSpec,
    SparseIndexConfig,
)
from vector_stores.contracts.models import (
    ChunkProvenance,
    CollectionConfig,
    CollectionInfo,
    DistanceMetric,
    ScoreDirection,
    ScrollPage,
    SearchRequest,
    SearchResult,
    SpaceVectors,
    StoredRecord,
    VectorSpaceConfig,
)
from vector_stores.contracts.protocols import VectorStore, collect_records, iter_records
from vector_stores.contracts.validation import (
    assert_embedding_space_compatible,
    normalize_descriptor,
    validate_collection_config,
    validate_record,
    validate_search_request,
    validate_space_vectors,
    validate_vector_space,
)

__all__ = [
    "FIELD_OPS",
    "LOGICAL_OPS",
    "BackendConnectionError",
    "ChunkProvenance",
    "CollectionConfig",
    "CollectionExistsError",
    "CollectionInfo",
    "CollectionNotFoundError",
    "DimensionMismatchError",
    "DistanceMetric",
    "EmbeddingSpaceMismatchError",
    "FilterOp",
    "HnswIndexConfig",
    "IndexConfig",
    "IndexDescription",
    "IndexKind",
    "IndexKindCapability",
    "InvalidRecordError",
    "MetadataFilter",
    "PayloadFieldType",
    "PayloadIndexSpec",
    "RecordNotFoundError",
    "ScoreDirection",
    "ScrollPage",
    "SearchRequest",
    "SearchResult",
    "SpaceVectors",
    "SparseIndexConfig",
    "StoreCapabilities",
    "StoredRecord",
    "UnsupportedFilterOperationError",
    "UnsupportedIndexOperationError",
    "UnsupportedVectorTypeError",
    "VectorSpaceConfig",
    "VectorStore",
    "VectorStoreConfigError",
    "VectorStoreDependencyError",
    "VectorStoreError",
    "and_",
    "assert_embedding_space_compatible",
    "collect_records",
    "eq",
    "exists",
    "gt",
    "gte",
    "is_in",
    "iter_records",
    "lt",
    "lte",
    "ne",
    "normalize_descriptor",
    "not_",
    "not_in",
    "or_",
    "validate_collection_config",
    "validate_filter",
    "validate_record",
    "validate_search_request",
    "validate_space_vectors",
    "validate_vector_space",
]
