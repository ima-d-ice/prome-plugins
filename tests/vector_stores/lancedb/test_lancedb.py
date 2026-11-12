"""LanceDB adapter tests.

The shared conformance suite is imported here and run against the LanceDB
adapter; the backend specific tests follow.
"""

from __future__ import annotations

import pytest

from tests.vector_stores.conformance import *
from tests.vector_stores.conformance import (
    make_config,
    make_record,
    unit_vector,
)
from vector_stores.contracts import (
    IndexConfig,
    IndexKind,
    SearchRequest,
    UnsupportedFilterOperationError,
    UnsupportedIndexOperationError,
)

pytest.importorskip("lancedb")

from vector_stores.lancedb import LanceDBVectorStore


@pytest.fixture()
def store_factory():
    """Return a factory building an ephemeral LanceDB store."""
    stores = []

    def factory():
        store = LanceDBVectorStore.memory()
        stores.append(store)
        return store

    yield factory
    for store in stores:
        store.close()


@pytest.fixture()
def persistent_store_factory(tmp_path):
    """Return a factory building a persistent LanceDB store."""

    def factory():
        return LanceDBVectorStore.local(str(tmp_path / "lancedb-data"))

    return factory


# ----------------------------------------------------------------------
# backend specific behaviour
# ----------------------------------------------------------------------
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
        assert record.chunk.doc_id == "a.html"
        assert record.metadata["tenant"] == "acme"
        # the declared configuration survives the restart
        info = reopened.describe_collection("conformance")
        assert info.vector_spaces["dense"].dimensions == 16
    finally:
        reopened.close()


def test_foreign_model_is_rejected_after_a_restart(persistent_store_factory):
    from vector_stores.contracts import EmbeddingSpaceMismatchError

    store = persistent_store_factory()
    store.create_collection(make_config())
    store.close()

    reopened = persistent_store_factory()
    try:
        record = make_record("c-0000")
        record.vectors["dense"].embedding_space = make_descriptor(model_id="other")
        with pytest.raises(EmbeddingSpaceMismatchError):
            reopened.upsert("conformance", [record])
    finally:
        reopened.close()


def test_sparse_vectors_are_stored_but_not_searchable(store):
    """LanceDB has no sparse index: storage and search are different things."""
    from embeddings.contracts.models import RepresentationType
    from tests.vector_stores.conformance import make_descriptor
    from vector_stores.contracts import (
        CollectionConfig,
        SpaceVectors,
        UnsupportedVectorTypeError,
        VectorSpaceConfig,
    )

    sparse_space = VectorSpaceConfig(
        name="sparse",
        vector_type=RepresentationType.SPARSE,
        embedding_space=make_descriptor(
            dimensions=None, representation_type=RepresentationType.SPARSE
        ),
    )
    config = CollectionConfig(name="sparse-docs", vector_spaces={"sparse": sparse_space})
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
    assert store.upsert("sparse-docs", [record]) == 1

    fetched = store.get("sparse-docs", ["c-0000"])[0]
    assert fetched.vectors["sparse"].sparse_indices == [1, 5]
    assert fetched.vectors["sparse"].sparse_values == [0.5, 0.25]

    with pytest.raises(UnsupportedVectorTypeError):
        store.search(
            SearchRequest(
                collection="sparse-docs",
                space="sparse",
                sparse_indices=[1],
                sparse_values=[1.0],
            )
        )


def test_sparse_search_is_not_advertised(store):
    assert store.capabilities.sparse_search is False
    assert store.capabilities.supports(IndexKind.SPARSE_INVERTED) is False


def test_multi_vector_search_is_supported(store):
    """LanceDB answers a multi-vector query natively (max-sim style)."""
    from embeddings.contracts.models import RepresentationType
    from tests.vector_stores.conformance import make_descriptor
    from vector_stores.contracts import (
        CollectionConfig,
        SpaceVectors,
        VectorSpaceConfig,
    )

    multi_space = VectorSpaceConfig(
        name="late",
        vector_type=RepresentationType.MULTI_VECTOR,
        embedding_space=make_descriptor(
            representation_type=RepresentationType.MULTI_VECTOR
        ),
        dimensions=16,
    )
    config = CollectionConfig(name="multi-docs", vector_spaces={"late": multi_space})
    store.create_collection(config)

    record = make_record("c-0000")
    record.vectors = {
        "late": SpaceVectors(
            space="late",
            embedding_space=multi_space.embedding_space,
            multi_vector=[unit_vector(1), unit_vector(2)],
        )
    }
    store.upsert("multi-docs", [record])

    results = store.search(
        SearchRequest(
            collection="multi-docs",
            space="late",
            multi_vector=[unit_vector(1)],
            limit=5,
        )
    )
    assert results
    assert results[0].chunk_id == "c-0000"
    assert store.capabilities.multi_vector_search is True


def test_native_vector_index_is_created_and_used(store):
    store.create_collection(make_config())
    store.upsert("conformance", [make_record(f"c-{i:04d}", seed=i) for i in range(600)])
    description = store.create_index(
        "conformance",
        IndexConfig(
            kind=IndexKind.HNSW,
            target="dense",
            build_params={"num_partitions": 4, "m": 16, "ef_construction": 100},
        ),
    )
    assert description.kind is IndexKind.HNSW
    assert description.build_params["index_type"] == "IVF_HNSW_FLAT"

    targets = {index.target for index in store.list_indexes("conformance")}
    assert "dense" in targets


def test_unsupported_vector_index_type_is_rejected(store):
    store.create_collection(make_config())
    store.upsert("conformance", [make_record("c-0000")])
    with pytest.raises(UnsupportedIndexOperationError):
        store.create_index(
            "conformance",
            IndexConfig(
                kind=IndexKind.HNSW,
                target="dense",
                backend_options={"index_type": "IVF_MADE_UP"},
            ),
        )


def test_sparse_index_creation_is_rejected(store):
    store.create_collection(make_config())
    with pytest.raises(UnsupportedIndexOperationError):
        store.create_index(
            "conformance", IndexConfig(kind=IndexKind.SPARSE_INVERTED, target="dense")
        )


def test_filter_on_an_undeclared_field_is_reported(store):
    """A filter that cannot be pushed down must raise, never be ignored."""
    store.create_collection(make_config())
    store.upsert("conformance", [make_record("c-0000")])
    with pytest.raises(UnsupportedFilterOperationError):
        store.count("conformance", __import__("vector_stores.contracts", fromlist=["eq"]).eq("unknown_field", "x"))


def test_query_time_parameters_are_accepted(store):
    store.create_collection(make_config())
    store.upsert("conformance", [make_record(f"c-{i:04d}", seed=i) for i in range(600)])
    store.create_index(
        "conformance",
        IndexConfig(kind=IndexKind.HNSW, target="dense", build_params={"num_partitions": 4}),
    )
    results = store.search(
        SearchRequest(
            collection="conformance",
            space="dense",
            dense=unit_vector(3),
            limit=5,
            params={"nprobes": 8},
        )
    )
    assert len(results) == 5
