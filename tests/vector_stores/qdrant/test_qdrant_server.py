"""Qdrant adapter tests against a running Qdrant server.

These tests are skipped when no server is reachable, so the default test run
never depends on Docker or on a background service.  Point them at a server
with ``QDRANT_URL`` (default ``http://localhost:6333``).

The server-only capabilities checked here are the ones the in-process local
mode cannot provide: payload indexes that are actually reported by the
backend, and the difference between an exact search and an HNSW search.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("qdrant_client")

from tests.vector_stores.qdrant.conftest import (
    dense_config,
    dense_record,
    dense_space,
    descriptor,
    multi_space,
    sparse_space,
    unit_vector,
)
from vector_stores.contracts import (
    CollectionConfig,
    EmbeddingSpaceMismatchError,
    IndexConfig,
    IndexKind,
    PayloadFieldType,
    PayloadIndexSpec,
    SearchRequest,
    SpaceVectors,
)
from vector_stores.qdrant import QdrantVectorStore

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


@pytest.fixture(scope="module")
def server_store():
    """Return a store connected to a running Qdrant server."""
    from qdrant_client import QdrantClient

    try:
        client = QdrantClient(url=QDRANT_URL, timeout=3)
        client.get_collections()
    except Exception as exc:  # noqa: BLE001 - any failure means "no server"
        pytest.skip(f"no Qdrant server reachable at {QDRANT_URL}: {exc}")

    store = QdrantVectorStore(client=client)
    try:
        yield store
    finally:
        store.close()


@pytest.fixture()
def collection_name(server_store):
    """Return a unique collection name and drop it afterwards."""
    name = f"prome-test-{uuid.uuid4().hex[:12]}"
    yield name
    server_store.delete_collection(name)


def test_server_create_upsert_and_search(server_store, collection_name):
    server_store.create_collection(dense_config(collection_name))
    server_store.upsert(
        collection_name,
        [
            dense_record("c-0000", "a.html", seed=1, metadata={"tenant": "acme"}),
            dense_record("c-0001", "b.html", seed=2, metadata={"tenant": "other"}),
        ],
    )
    results = server_store.search(
        SearchRequest(
            collection=collection_name, space="dense", dense=unit_vector(1), limit=5
        )
    )
    assert results
    assert results[0].chunk_id == "c-0000"
    assert results[0].doc_id == "a.html"


def test_server_payload_index_is_reported(server_store, collection_name):
    """The server records the payload index; the local mode does not."""
    config = dense_config(
        collection_name,
        payload_indexes=(PayloadIndexSpec("tenant", PayloadFieldType.KEYWORD),),
    )
    server_store.create_collection(config)
    indexes = server_store.list_indexes(collection_name)
    payload_indexes = [i for i in indexes if i.kind is IndexKind.PAYLOAD]
    assert any(i.target == "tenant" for i in payload_indexes)


def test_server_create_and_delete_a_payload_index(server_store, collection_name):
    server_store.create_collection(dense_config(collection_name))
    server_store.create_index(
        collection_name,
        IndexConfig(
            kind=IndexKind.PAYLOAD,
            target="tenant",
            field_type=PayloadFieldType.KEYWORD,
        ),
    )
    targets = {i.target for i in server_store.list_indexes(collection_name)}
    assert "tenant" in targets
    assert server_store.delete_index(collection_name, "tenant") is True


def test_server_exact_and_hnsw_search(server_store, collection_name):
    server_store.create_collection(dense_config(collection_name))
    server_store.upsert(
        collection_name,
        [dense_record(f"c-{index:04d}", "a.html", seed=index) for index in range(20)],
    )
    exact = server_store.search(
        SearchRequest(
            collection=collection_name,
            space="dense",
            dense=unit_vector(0),
            limit=5,
            params={"exact": True},
        )
    )
    approximate = server_store.search(
        SearchRequest(
            collection=collection_name,
            space="dense",
            dense=unit_vector(0),
            limit=5,
            params={"hnsw_ef": 64},
        )
    )
    assert len(exact) == 5
    assert len(approximate) == 5


def test_server_sparse_and_multi_vector(server_store, collection_name):
    config = CollectionConfig(
        name=collection_name,
        vector_spaces={
            "dense": dense_space(),
            "sparse": sparse_space(),
            "late": multi_space(),
        },
    )
    server_store.create_collection(config)
    record = dense_record("c-0000", "a.html")
    record.vectors["sparse"] = SpaceVectors(
        space="sparse",
        embedding_space=sparse_space().embedding_space,
        sparse_indices=[1, 5],
        sparse_values=[0.5, 0.25],
    )
    record.vectors["late"] = SpaceVectors(
        space="late",
        embedding_space=multi_space().embedding_space,
        multi_vector=[[0.1] * 8, [0.2] * 8],
    )
    server_store.upsert(collection_name, [record])

    sparse_hits = server_store.search(
        SearchRequest(
            collection=collection_name,
            space="sparse",
            sparse_indices=[1],
            sparse_values=[1.0],
            limit=5,
        )
    )
    assert sparse_hits and sparse_hits[0].chunk_id == "c-0000"

    multi_hits = server_store.search(
        SearchRequest(
            collection=collection_name,
            space="late",
            multi_vector=[[0.1] * 8],
            limit=5,
        )
    )
    assert multi_hits and multi_hits[0].chunk_id == "c-0000"


def test_server_rejects_a_foreign_model(server_store, collection_name):
    server_store.create_collection(dense_config(collection_name))
    record = dense_record("c-0000")
    record.vectors["dense"].embedding_space = descriptor(model_id="other/model")
    with pytest.raises(EmbeddingSpaceMismatchError):
        server_store.upsert(collection_name, [record])
