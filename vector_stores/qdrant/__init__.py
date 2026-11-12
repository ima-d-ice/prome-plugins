"""Qdrant adapter for the Prome vector-store contracts.

The adapter accepts precomputed embeddings only - it never embeds text and
never loads a model.  ``qdrant-client`` is an optional dependency: it is
imported lazily, so this package can be imported without it.

Example::

    from vector_stores.contracts import CollectionConfig, VectorSpaceConfig
    from vector_stores.qdrant import QdrantVectorStore

    store = QdrantVectorStore.local("/tmp/qdrant-data")
"""

from vector_stores.qdrant.store import (
    QdrantMode,
    QdrantStoreConfig,
    QdrantVectorStore,
)

__all__ = [
    "QdrantMode",
    "QdrantStoreConfig",
    "QdrantVectorStore",
]
