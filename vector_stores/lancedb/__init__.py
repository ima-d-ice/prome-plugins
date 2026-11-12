"""LanceDB adapter for the Prome vector-store contracts.

LanceDB is an embedded, local vector database: the data lives in a
directory, no server is needed and no hosted service is ever used.  The
adapter accepts precomputed embeddings only - it never embeds text and
never loads a model.

``lancedb`` is an optional dependency: it is imported lazily, so this
package can be imported without it.

Example::

    from vector_stores.contracts import CollectionConfig, SearchRequest
    from vector_stores.lancedb import LanceDBVectorStore

    store = LanceDBVectorStore.local("/tmp/lancedb-data")
    store.create_collection(config)
    store.upsert("docs", records)
    hits = store.search(
        SearchRequest(collection="docs", space="dense", dense=query_vector)
    )
"""

from vector_stores.lancedb.store import (
    LanceDBStoreConfig,
    LanceDBVectorStore,
)

__all__ = [
    "LanceDBStoreConfig",
    "LanceDBVectorStore",
]
