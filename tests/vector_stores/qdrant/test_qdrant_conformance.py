"""Qdrant adapter conformance tests.

The shared conformance suite is imported here and run against the Qdrant
adapter, next to the adapter specific tests of ``test_qdrant_store.py``.
"""

from __future__ import annotations

import pytest

from tests.vector_stores.conformance import *
from tests.vector_stores.conformance import make_config, make_record

pytest.importorskip("qdrant_client")

from vector_stores.qdrant import QdrantVectorStore


@pytest.fixture()
def store_factory():
    """Return a factory building an ephemeral (in-memory) Qdrant store."""
    stores = []

    def factory():
        store = QdrantVectorStore.memory()
        stores.append(store)
        return store

    yield factory
    for store in stores:
        store.close()


@pytest.fixture()
def persistent_store_factory(tmp_path):
    """Return a factory building a persistent (local directory) Qdrant store."""

    def factory():
        return QdrantVectorStore.local(str(tmp_path / "qdrant-data"))

    return factory


def test_persistence_after_reopening(persistent_store_factory):
    store = persistent_store_factory()
    store.create_collection(make_config())
    store.upsert("conformance", [make_record("c-0000", "a.html", seed=1)])
    store.close()

    reopened = persistent_store_factory()
    try:
        assert reopened.count("conformance") == 1
        record = reopened.get("conformance", ["c-0000"])[0]
        assert record.chunk.id == "c-0000"
        assert record.metadata["tenant"] == "acme"
    finally:
        reopened.close()
