"""Qdrant adapter tests using the persistent local mode (a directory, no server).

These tests verify that a collection, its configuration and its records
survive a restart of the store.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qdrant_client")

from tests.vector_stores.qdrant.conftest import (
    dense_config,
    dense_record,
    unit_vector,
)
from vector_stores.contracts import EmbeddingSpaceMismatchError, SearchRequest
from vector_stores.qdrant import QdrantVectorStore


def test_collection_and_records_survive_a_restart(tmp_path):
    path = str(tmp_path / "qdrant-data")

    store = QdrantVectorStore.local(path)
    store.create_collection(dense_config())
    store.upsert(
        "docs",
        [
            dense_record("c-0000", "a.html", seed=1, metadata={"tenant": "acme"}),
            dense_record("c-0001", "b.html", seed=2),
        ],
    )
    store.close()

    reopened = QdrantVectorStore.local(path)
    try:
        info = reopened.describe_collection("docs")
        assert info.name == "docs"
        assert info.vector_spaces["dense"].dimensions == 8
        assert info.point_count == 2

        results = reopened.search(
            SearchRequest(
                collection="docs", space="dense", dense=unit_vector(1), limit=5
            )
        )
        assert results[0].chunk_id == "c-0000"
        assert results[0].doc_id == "a.html"

        fetched = reopened.get("docs", ["c-0001"])
        assert fetched[0].chunk.id == "c-0001"
        assert fetched[0].chunk.metadata["source"] == "b.html"
    finally:
        reopened.close()


def test_the_stored_configuration_is_reused_after_a_restart(tmp_path):
    """The declared configuration is restored from the collection metadata."""
    path = str(tmp_path / "qdrant-data")

    store = QdrantVectorStore.local(path)
    store.create_collection(dense_config())
    store.close()

    reopened = QdrantVectorStore.local(path)
    try:
        # The configuration is read back from the backend: a record of another
        # model is still rejected after the restart.
        from tests.vector_stores.qdrant.conftest import descriptor

        record = dense_record("c-0000")
        record.vectors["dense"].embedding_space = descriptor(model_id="other/model")
        with pytest.raises(EmbeddingSpaceMismatchError):
            reopened.upsert("docs", [record])
    finally:
        reopened.close()


def test_a_collection_created_outside_the_adapter_is_reported(tmp_path):
    """A foreign collection has no Prome configuration: the adapter says so."""
    from qdrant_client import QdrantClient, models

    from vector_stores.contracts import VectorStoreConfigError

    path = str(tmp_path / "qdrant-data")
    client = QdrantClient(path=path)
    client.create_collection(
        "foreign",
        vectors_config=models.VectorParams(
            size=4, distance=models.Distance.COSINE
        ),
    )
    client.close()

    store = QdrantVectorStore.local(path)
    try:
        with pytest.raises(VectorStoreConfigError):
            store.describe_collection("foreign")
        # The collection is still listed, without a Prome configuration.
        names = [info.name for info in store.list_collections()]
        assert "foreign" in names
    finally:
        store.close()
