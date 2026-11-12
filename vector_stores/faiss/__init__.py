"""FAISS adapter for the Prome vector-store contracts.

FAISS is a similarity search library, not a document database.  This adapter
combines FAISS (dense vectors, nearest neighbour search) with SQLite
(records, metadata, provenance, collection configuration).

``faiss-cpu`` is an optional dependency: it is imported lazily, so this
package can be imported without it.

Example::

    from vector_stores.contracts import CollectionConfig, SearchRequest
    from vector_stores.faiss import FaissVectorStore

    store = FaissVectorStore.local("/tmp/faiss-data")
    store.create_collection(config)
    store.upsert("docs", records)
    hits = store.search(
        SearchRequest(collection="docs", space="dense", dense=query_vector)
    )
"""

from vector_stores.faiss.store import FaissStoreConfig, FaissVectorStore

__all__ = [
    "FaissStoreConfig",
    "FaissVectorStore",
]
