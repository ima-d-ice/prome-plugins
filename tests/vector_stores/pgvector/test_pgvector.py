"""PostgreSQL (pgvector) adapter tests.

The server tests need a PostgreSQL server with the ``vector`` extension.
Point them at one with ``PROMERAG_PGVECTOR_TEST_DSN``; without it they skip
cleanly.  The DSN used by the CI-like local setup is
``postgresql://postgres:prome@localhost:55432/prome``.
"""

from __future__ import annotations

import os
import uuid

import pytest

from tests.vector_stores.conformance import *
from tests.vector_stores.conformance import (
    make_config,
    make_record,
    unit_vector,
)
from vector_stores.contracts import (
    DimensionMismatchError,
    IndexConfig,
    IndexKind,
    SearchRequest,
    UnsupportedVectorTypeError,
)

pytest.importorskip("psycopg")
pytest.importorskip("pgvector")

from vector_stores.pgvector import (
    PgVectorStoreConfig,
    PgVectorVectorStore,
)

TEST_DSN = os.environ.get(
    "PROMERAG_PGVECTOR_TEST_DSN", "postgresql://postgres:prome@localhost:55432/prome"
)


def _server_available() -> bool:
    try:
        import psycopg

        with psycopg.connect(TEST_DSN, connect_timeout=3) as connection:
            connection.execute("select 1")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _server_available(),
    reason=f"no PostgreSQL server with pgvector reachable at {TEST_DSN}",
)


@pytest.fixture()
def store_factory():
    """Return a factory building a store connected to the test server."""
    stores = []

    def factory():
        store = PgVectorVectorStore(PgVectorStoreConfig(dsn=TEST_DSN))
        stores.append(store)
        return store

    yield factory
    for store in stores:
        store.close()


@pytest.fixture()
def collection_name():
    """Return a unique collection name and drop it afterwards."""
    name = f"prome_test_{uuid.uuid4().hex[:12]}"
    try:
        yield name
    finally:
        cleanup = PgVectorVectorStore(PgVectorStoreConfig(dsn=TEST_DSN))
        try:
            cleanup.delete_collection(name)
        finally:
            cleanup.close()


def make_store() -> PgVectorVectorStore:
    """Return a store connected to the test server."""
    return PgVectorVectorStore(PgVectorStoreConfig(dsn=TEST_DSN))


# ----------------------------------------------------------------------
# backend specific behaviour
# ----------------------------------------------------------------------
def test_the_pgvector_version_is_detected():
    store = make_store()
    try:
        assert store.extension_version is not None
        assert store.extension_version[0].isdigit()
    finally:
        store.close()


def test_collection_is_a_table(store_factory, collection_name):
    store = store_factory()
    store.create_collection(make_config(collection_name))
    with store._connection.cursor() as cursor:
        cursor.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = %s AND table_name = %s",
            ("public", collection_name),
        )
        assert cursor.fetchone() is not None


def test_persistence_after_reopening(store_factory, collection_name):
    store = store_factory()
    store.create_collection(make_config(collection_name))
    store.upsert(collection_name, [make_record("c-0000", "a.html", seed=1)])
    store.close()

    reopened = make_store()
    try:
        assert reopened.count(collection_name) == 1
        record = reopened.get(collection_name, ["c-0000"])[0]
        assert record.chunk.id == "c-0000"
        assert record.chunk.doc_id == "a.html"
        assert record.metadata["tenant"] == "acme"
        results = reopened.search(
            SearchRequest(
                collection=collection_name, space="dense", dense=unit_vector(1), limit=5
            )
        )
        assert results[0].chunk_id == "c-0000"
    finally:
        reopened.close()


def test_foreign_model_is_rejected_after_a_restart(store_factory, collection_name):
    from vector_stores.contracts import EmbeddingSpaceMismatchError

    store = store_factory()
    store.create_collection(make_config(collection_name))
    store.close()

    reopened = make_store()
    try:
        record = make_record("c-0000")
        record.vectors["dense"].embedding_space = make_descriptor(model_id="other")
        with pytest.raises(EmbeddingSpaceMismatchError):
            reopened.upsert(collection_name, [record])
    finally:
        reopened.close()


def test_transactions_are_committed(store_factory, collection_name):
    store = store_factory()
    store.create_collection(make_config(collection_name))
    store.upsert(collection_name, [make_record("c-0000", seed=1)])
    # A second connection sees the committed data.
    other = make_store()
    try:
        assert other.count(collection_name) == 1
    finally:
        other.close()


def test_a_failed_statement_does_not_leave_a_partial_batch(
    store_factory, collection_name
):
    """The batch is written in one statement: it is atomic."""
    store = store_factory()
    store.create_collection(make_config(collection_name))
    good = make_record("c-0000", seed=1)
    bad = make_record("c-0001", seed=2)
    bad.vectors["dense"].dense = [0.1] * 3
    with pytest.raises(DimensionMismatchError):
        store.upsert(collection_name, [good, bad])
    assert store.count(collection_name) == 0


def test_hnsw_index_is_created(store_factory, collection_name):
    store = store_factory()
    store.create_collection(make_config(collection_name))
    store.upsert(
        collection_name, [make_record(f"c-{i:04d}", seed=i) for i in range(500)]
    )
    description = store.create_index(
        collection_name,
        IndexConfig(kind=IndexKind.HNSW, target="dense", build_params={"m": 16, "ef_construction": 64}),
    )
    assert description.kind is IndexKind.HNSW
    names = [index.name for index in store.list_indexes(collection_name)]
    assert any("hnsw" in str(name) for name in names)


def test_ivfflat_index_is_created(store_factory, collection_name):
    store = store_factory()
    store.create_collection(make_config(collection_name))
    store.upsert(
        collection_name, [make_record(f"c-{i:04d}", seed=i) for i in range(500)]
    )
    description = store.create_index(
        collection_name,
        IndexConfig(kind=IndexKind.IVF, target="dense", build_params={"lists": 8}),
    )
    assert description.kind is IndexKind.IVF
    names = [index.name for index in store.list_indexes(collection_name)]
    assert any("ivfflat" in str(name) for name in names)


def test_payload_indexes_are_created(store_factory, collection_name):
    store = store_factory()
    store.create_collection(make_config(collection_name))
    with store._connection.cursor() as cursor:
        cursor.execute(
            "SELECT indexname FROM pg_indexes WHERE tablename = %s", (collection_name,)
        )
        names = [row[0] for row in cursor.fetchall()]
    assert any("metadata_gin" in name for name in names)
    assert any("meta_tenant_idx" in name for name in names)


def test_metadata_filtering_uses_jsonb(store_factory, collection_name):
    from vector_stores.contracts import eq

    store = store_factory()
    store.create_collection(make_config(collection_name))
    store.upsert(
        collection_name,
        [
            make_record("c-0000", tenant="acme", seed=1),
            make_record("c-0001", tenant="other", seed=2),
        ],
    )
    assert store.count(collection_name, eq("tenant", "acme")) == 1
    results = store.search(
        SearchRequest(
            collection=collection_name,
            space="dense",
            dense=unit_vector(0),
            limit=5,
            filter=eq("tenant", "acme"),
        )
    )
    assert [result.chunk_id for result in results] == ["c-0000"]


def test_filter_values_are_parameterized(store_factory, collection_name):
    """A value that looks like SQL is bound as a parameter."""
    from vector_stores.contracts import eq

    store = store_factory()
    store.create_collection(make_config(collection_name))
    store.upsert(collection_name, [make_record("c-0000", tenant="acme", seed=1)])
    # The statement must not fail and must not match anything.
    assert store.count(collection_name, eq("tenant", "'; DROP TABLE x; --")) == 0
    assert store.count(collection_name, eq("tenant", "acme")) == 1


def test_sparse_vectors_are_stored_and_searched(store_factory, collection_name):
    from embeddings.contracts.models import RepresentationType
    from vector_stores.contracts import (
        CollectionConfig,
        SpaceVectors,
        VectorSpaceConfig,
    )

    store = store_factory()
    assert store.capabilities.sparse_search is True

    sparse_space = VectorSpaceConfig(
        name="sparse",
        vector_type=RepresentationType.SPARSE,
        embedding_space=make_descriptor(
            dimensions=None, representation_type=RepresentationType.SPARSE
        ),
    )
    config = CollectionConfig(
        name=collection_name, vector_spaces={"sparse": sparse_space}
    )
    store.create_collection(config)
    record = make_record("c-0000")
    record.vectors = {
        "sparse": SpaceVectors(
            space="sparse",
            embedding_space=sparse_space.embedding_space,
            sparse_indices=[1, 5],
            sparse_values=[0.5, 0.25],
        )
    }
    store.upsert(collection_name, [record])

    results = store.search(
        SearchRequest(
            collection=collection_name,
            space="sparse",
            sparse_indices=[1],
            sparse_values=[1.0],
            limit=5,
        )
    )
    assert results
    assert results[0].chunk_id == "c-0000"


def test_multi_vector_is_stored_but_not_searchable(store_factory, collection_name):
    from embeddings.contracts.models import RepresentationType
    from vector_stores.contracts import (
        CollectionConfig,
        SpaceVectors,
        VectorSpaceConfig,
    )

    store = store_factory()
    multi_space = VectorSpaceConfig(
        name="late",
        vector_type=RepresentationType.MULTI_VECTOR,
        embedding_space=make_descriptor(
            representation_type=RepresentationType.MULTI_VECTOR
        ),
        dimensions=16,
    )
    config = CollectionConfig(
        name=collection_name, vector_spaces={"late": multi_space}
    )
    store.create_collection(config)
    record = make_record("c-0000")
    record.vectors = {
        "late": SpaceVectors(
            space="late",
            embedding_space=multi_space.embedding_space,
            multi_vector=[unit_vector(1), unit_vector(2)],
        )
    }
    store.upsert(collection_name, [record])
    assert store.count(collection_name) == 1
    with pytest.raises(UnsupportedVectorTypeError):
        store.search(
            SearchRequest(
                collection=collection_name,
                space="late",
                multi_vector=[unit_vector(1)],
                limit=5,
            )
        )
    assert store.capabilities.multi_vector_search is False


def test_connection_failure_is_reported():
    from vector_stores.contracts import BackendConnectionError

    with pytest.raises(BackendConnectionError):
        PgVectorVectorStore(
            PgVectorStoreConfig(dsn="postgresql://nobody@127.0.0.1:1/nothing")
        )


def test_a_missing_extension_is_reported():
    """A database without the vector extension cannot be used."""
    from vector_stores.contracts import BackendConnectionError

    try:
        import psycopg

        with psycopg.connect(TEST_DSN, autocommit=True) as connection:
            connection.execute("drop extension if exists vector cascade")
    except Exception:
        pytest.skip("the test database does not allow dropping the extension")
    try:
        with pytest.raises(BackendConnectionError):
            make_store()
    finally:
        import psycopg

        with psycopg.connect(TEST_DSN, autocommit=True) as connection:
            connection.execute("create extension if not exists vector")
