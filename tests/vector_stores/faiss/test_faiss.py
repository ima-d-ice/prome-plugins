"""FAISS adapter tests.

The shared conformance suite is imported here and run against the FAISS
adapter; the backend specific tests follow.
"""

from __future__ import annotations

import sys

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
    UnsupportedIndexOperationError,
    UnsupportedVectorTypeError,
)

# NOTE: ``faiss`` is deliberately NOT imported at module level.  The
# ``faiss-cpu`` wheel links its own copy of the OpenMP runtime, and so does
# torch: loading both in the same process aborts it ("OMP: Error #15").  The
# adapter imports FAISS lazily and the fixture below skips the suite when
# torch has already been loaded - which is the case in the full test suite.
# Run this suite on its own (``pytest tests/vector_stores/faiss``) to
# exercise it.
from vector_stores.faiss import FaissVectorStore

_SKIP_REASON = (
    "faiss-cpu and torch both link the OpenMP runtime and cannot be loaded in "
    "the same process (macOS aborts with 'OMP: Error #15'); run "
    "'pytest tests/vector_stores/faiss' on its own"
)


def _torch_is_loaded() -> bool:
    """Return True when torch (and its OpenMP runtime) is already loaded."""
    return "torch" in sys.modules


def _parser_tests_are_collected(request) -> bool:
    """Return True when the session also runs the parser tests.

    The docling parser imports torch, and torch and FAISS cannot share a
    process.  Checking the collected items makes the guard independent of
    the order the test paths are given on the command line.
    """
    for item in request.session.items:
        path = str(item.fspath).replace("\\", "/")
        if "/tests/parsers/" in path or path.endswith("/tests/parsers"):
            return True
    return False


@pytest.fixture(autouse=True)
def _faiss_runtime_guard(request):
    """Skip the whole module when FAISS cannot be loaded safely.

    ``faiss-cpu`` and torch each link their own copy of ``libomp``: loading
    both in the same process aborts the interpreter (it is not a catchable
    error).  The guard keeps the rest of the test suite usable and tells the
    user how to run this suite.
    """
    if _torch_is_loaded():
        pytest.skip(_SKIP_REASON)
    if _parser_tests_are_collected(request):
        pytest.skip(
            _SKIP_REASON
            + " (this session also collects the parser tests, which import torch)"
        )
    pytest.importorskip("faiss")


@pytest.fixture()
def store_factory():
    """Return a factory building an ephemeral FAISS store."""
    stores = []

    def factory():
        store = FaissVectorStore.memory()
        stores.append(store)
        return store

    yield factory
    for store in stores:
        store.close()


@pytest.fixture()
def persistent_store_factory(tmp_path):
    """Return a factory building a persistent FAISS store."""

    def factory():
        return FaissVectorStore.local(str(tmp_path / "faiss-data"))

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
        assert record.chunk.text == "hello world"
        assert record.metadata["tenant"] == "acme"
        assert record.chunk.metadata["source"] == "a.html"
        results = reopened.search(
            SearchRequest(
                collection="conformance", space="dense", dense=unit_vector(1), limit=5
            )
        )
        assert results[0].chunk_id == "c-0000"
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


def test_sqlite_is_the_authoritative_store(tmp_path):
    """The records live in SQLite, the index is a derived artifact."""
    from vector_stores.faiss.mapping import DATABASE_NAME

    store = FaissVectorStore.local(str(tmp_path / "faiss-data"))
    try:
        store.create_collection(make_config())
        store.upsert("conformance", [make_record("c-0000", seed=1)])
        directory = tmp_path / "faiss-data" / "conformance"
        assert (directory / DATABASE_NAME).exists()
        assert list(directory.glob("index-*.faiss"))
    finally:
        store.close()


def test_a_missing_index_is_rebuilt_from_sqlite(tmp_path):
    """A deleted index file is rebuilt from the authoritative records."""
    from vector_stores.faiss.mapping import DATABASE_NAME

    store = FaissVectorStore.local(str(tmp_path / "faiss-data"))
    try:
        store.create_collection(make_config())
        store.upsert("conformance", [make_record("c-0000", seed=1)])
        directory = tmp_path / "faiss-data" / "conformance"
        for path in directory.glob("index-*.faiss"):
            path.unlink()
        store.close()

        reopened = FaissVectorStore.local(str(tmp_path / "faiss-data"))
        try:
            assert reopened.count("conformance") == 1
            results = reopened.search(
                SearchRequest(
                    collection="conformance",
                    space="dense",
                    dense=unit_vector(1),
                    limit=5,
                )
            )
            assert results[0].chunk_id == "c-0000"
            assert (directory / DATABASE_NAME).exists()
        finally:
            reopened.close()
    finally:
        store.close()


def test_a_corrupt_index_is_rebuilt_from_sqlite(tmp_path):
    store = FaissVectorStore.local(str(tmp_path / "faiss-data"))
    try:
        store.create_collection(make_config())
        store.upsert("conformance", [make_record("c-0000", seed=1)])
        directory = tmp_path / "faiss-data" / "conformance"
        for path in directory.glob("index-*.faiss"):
            path.write_bytes(b"not a faiss index")
        store.close()

        reopened = FaissVectorStore.local(str(tmp_path / "faiss-data"))
        try:
            results = reopened.search(
                SearchRequest(
                    collection="conformance",
                    space="dense",
                    dense=unit_vector(1),
                    limit=5,
                )
            )
            assert results[0].chunk_id == "c-0000"
        finally:
            reopened.close()
    finally:
        store.close()


def test_index_rebuild_after_a_deletion(store):
    """The index stays consistent with SQLite after a deletion."""
    store.create_collection(make_config())
    store.upsert(
        "conformance",
        [make_record(f"c-{index:04d}", seed=index) for index in range(20)],
    )
    assert store.delete("conformance", ["c-0000"]) == 1
    assert store.count("conformance") == 19
    results = store.search(
        SearchRequest(
            collection="conformance", space="dense", dense=unit_vector(0), limit=5
        )
    )
    assert all(result.chunk_id != "c-0000" for result in results)
    store.rebuild_index("conformance")
    assert store.count("conformance") == 19


def test_upsert_replaces_the_vector(store):
    store.create_collection(make_config())
    store.upsert("conformance", [make_record("c-0000", seed=1)])
    store.upsert("conformance", [make_record("c-0000", seed=9)])
    assert store.count("conformance") == 1
    results = store.search(
        SearchRequest(
            collection="conformance", space="dense", dense=unit_vector(9), limit=1
        )
    )
    assert results[0].chunk_id == "c-0000"
    assert results[0].score > 0.99


def test_filtered_search_returns_the_true_top_k(store):
    """A filtered search is exact over the matching records."""
    store.create_collection(make_config())
    store.upsert("conformance", [make_record(f"c-{i:04d}", seed=i) for i in range(50)])
    from vector_stores.contracts import eq

    filtered = store.search(
        SearchRequest(
            collection="conformance",
            space="dense",
            dense=unit_vector(0),
            limit=5,
            filter=eq("tenant", "acme"),
        )
    )
    assert len(filtered) == 5
    assert all(result.metadata["tenant"] == "acme" for result in filtered)
    # the closest matching record is the one that is closest in the corpus
    exact = store.search(
        SearchRequest(
            collection="conformance",
            space="dense",
            dense=unit_vector(0),
            limit=50,
            params={"exact": True},
        )
    )
    expected = [r for r in exact if r.metadata["tenant"] == "acme"][:5]
    assert [r.chunk_id for r in filtered] == [r.chunk_id for r in expected]


def test_hnsw_index_is_created(store):
    store.create_collection(make_config())
    store.upsert("conformance", [make_record(f"c-{i:04d}", seed=i) for i in range(200)])
    description = store.create_index(
        "conformance",
        IndexConfig(
            kind=IndexKind.HNSW,
            target="dense",
            build_params={"m": 16, "ef_construct": 100},
        ),
    )
    assert description.kind is IndexKind.HNSW
    targets = {index.target for index in store.list_indexes("conformance")}
    assert "dense" in targets


def test_ivf_index_is_created(store):
    store.create_collection(make_config())
    store.upsert("conformance", [make_record(f"c-{i:04d}", seed=i) for i in range(200)])
    description = store.create_index(
        "conformance",
        IndexConfig(kind=IndexKind.HNSW, target="dense", build_params={"index_type": "ivf", "nlist": 8}),
    )
    assert description.build_params["index_type"] == "ivf"


def test_ivf_index_needs_enough_vectors(store):
    store.create_collection(make_config())
    store.upsert("conformance", [make_record("c-0000")])
    from vector_stores.contracts import VectorStoreConfigError

    with pytest.raises(VectorStoreConfigError):
        store.create_index(
            "conformance",
            IndexConfig(kind=IndexKind.HNSW, target="dense", build_params={"index_type": "ivf", "nlist": 64}),
        )


def test_multi_vector_and_sparse_spaces_are_rejected(store):
    from embeddings.contracts.models import RepresentationType
    from vector_stores.contracts import CollectionConfig, VectorSpaceConfig

    multi_space = VectorSpaceConfig(
        name="late",
        vector_type=RepresentationType.MULTI_VECTOR,
        embedding_space=make_descriptor(
            representation_type=RepresentationType.MULTI_VECTOR
        ),
        dimensions=16,
    )
    config = CollectionConfig(name="multi", vector_spaces={"late": multi_space})
    with pytest.raises(UnsupportedVectorTypeError):
        store.create_collection(config)


def test_sparse_search_is_not_advertised(store):
    assert store.capabilities.sparse_search is False
    assert store.capabilities.multi_vector_search is False
    assert store.capabilities.supports(IndexKind.SPARSE_INVERTED) is False
    assert store.capabilities.supports(IndexKind.MULTI_VECTOR) is False


def test_sparse_index_creation_is_rejected(store):
    store.create_collection(make_config())
    with pytest.raises(UnsupportedIndexOperationError):
        store.create_index(
            "conformance", IndexConfig(kind=IndexKind.SPARSE_INVERTED, target="dense")
        )


def test_payload_index_creation(store):
    store.create_collection(make_config())
    description = store.create_index(
        "conformance",
        IndexConfig(kind=IndexKind.PAYLOAD, target="tenant"),
    )
    assert description.kind is IndexKind.PAYLOAD
    targets = {index.target for index in store.list_indexes("conformance")}
    assert "tenant" in targets


def test_payload_index_on_an_undeclared_field_is_allowed(store):
    """SQLite can index any metadata field, declared or not."""
    store.create_collection(make_config())
    description = store.create_index(
        "conformance", IndexConfig(kind=IndexKind.PAYLOAD, target="extra_field")
    )
    assert description.kind is IndexKind.PAYLOAD
    targets = {index.target for index in store.list_indexes("conformance")}
    assert "extra_field" in targets


def test_payload_index_with_an_invalid_field_name_is_rejected(store):
    from vector_stores.contracts import VectorStoreConfigError

    store.create_collection(make_config())
    with pytest.raises(VectorStoreConfigError):
        store.create_index(
            "conformance",
            IndexConfig(kind=IndexKind.PAYLOAD, target="not an identifier"),
        )


def test_query_time_parameters_are_accepted(store):
    store.create_collection(make_config())
    store.upsert("conformance", [make_record(f"c-{i:04d}", seed=i) for i in range(200)])
    store.create_index(
        "conformance",
        IndexConfig(kind=IndexKind.HNSW, target="dense", build_params={"m": 16}),
    )
    results = store.search(
        SearchRequest(
            collection="conformance",
            space="dense",
            dense=unit_vector(3),
            limit=5,
            params={"hnsw_ef": 64},
        )
    )
    assert len(results) == 5
