"""PostgreSQL (pgvector) adapter for the Prome vector-store contracts.

The adapter stores precomputed embeddings in a PostgreSQL table and searches
them with the pgvector operators.  It needs a PostgreSQL server with the
``vector`` extension installed.

``psycopg`` (v3) and ``pgvector`` are optional dependencies: they are
imported lazily, so this package can be imported without them.

Example::

    from vector_stores.contracts import CollectionConfig, SearchRequest
    from vector_stores.pgvector import PgVectorVectorStore

    store = PgVectorVectorStore.server("postgresql://user@host/db")
    store.create_collection(config)
    store.upsert("docs", records)
    hits = store.search(
        SearchRequest(collection="docs", space="dense", dense=query_vector)
    )
"""

from vector_stores.pgvector.store import PgVectorStoreConfig, PgVectorVectorStore

__all__ = [
    "PgVectorStoreConfig",
    "PgVectorVectorStore",
]
