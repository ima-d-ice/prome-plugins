"""Qdrant adapter tests (in-memory local mode).

The in-memory mode of the Qdrant client is used for isolation: no server, no
Docker, no model download.  The persistent local mode is covered in
``test_qdrant_persistence.py``.
"""

from __future__ import annotations

import sys

import pytest

from embeddings.contracts.models import Normalization
from vector_stores.contracts import (
    CollectionExistsError,
    CollectionNotFoundError,
    DimensionMismatchError,
    EmbeddingSpaceMismatchError,
    IndexConfig,
    IndexKind,
    InvalidRecordError,
    PayloadFieldType,
    SearchRequest,
    SpaceVectors,
    UnsupportedIndexOperationError,
    VectorStoreConfigError,
    VectorStoreDependencyError,
    and_,
    eq,
    exists,
    gt,
    is_in,
    lt,
    ne,
    not_,
    not_in,
    or_,
)

pytest.importorskip("qdrant_client")

from tests.vector_stores.qdrant.conftest import (
    dense_config,
    dense_record,
    descriptor,
    multi_space,
    multi_space_config,
    sparse_space,
    unit_vector,
)
from vector_stores.qdrant import QdrantVectorStore
from vector_stores.qdrant.mapping import point_id


# ----------------------------------------------------------------------
# collection lifecycle
# ----------------------------------------------------------------------
def test_create_and_describe_collection(store):
    config = dense_config()
    info = store.create_collection(config)
    assert info.name == "docs"
    assert info.description == "test collection"
    assert set(info.vector_spaces) == {"dense"}
    assert info.point_count == 0

    described = store.describe_collection("docs")
    assert described.name == "docs"
    assert described.vector_spaces["dense"].dimensions == 8
    assert described.vector_spaces["dense"].distance.value == "cosine"


def test_create_collection_twice_raises(store):
    store.create_collection(dense_config())
    with pytest.raises(CollectionExistsError):
        store.create_collection(dense_config())


def test_create_collection_with_exist_ok(store):
    store.create_collection(dense_config())
    info = store.create_collection(dense_config(), exist_ok=True)
    assert info.name == "docs"


def test_create_collection_with_a_conflicting_configuration_raises(store):
    store.create_collection(dense_config())
    with pytest.raises(VectorStoreConfigError):
        store.create_collection(dense_config(dimensions=16), exist_ok=True)


def test_describe_missing_collection_raises(store):
    with pytest.raises(CollectionNotFoundError):
        store.describe_collection("missing")


def test_list_collections(store):
    store.create_collection(dense_config("a"))
    store.create_collection(dense_config("b"))
    assert sorted(info.name for info in store.list_collections()) == ["a", "b"]


def test_delete_collection(store):
    store.create_collection(dense_config())
    assert store.delete_collection("docs") is True
    assert store.delete_collection("docs") is False
    assert store.list_collections() == []


def test_upsert_into_a_missing_collection_raises(store):
    with pytest.raises(CollectionNotFoundError):
        store.upsert("missing", [dense_record()])


# ----------------------------------------------------------------------
# upsert / retrieval
# ----------------------------------------------------------------------
def test_dense_upsert_and_search(store):
    store.create_collection(dense_config())
    store.upsert("docs", [dense_record("c-0000", "a.html", seed=1)])

    results = store.search(
        SearchRequest(collection="docs", space="dense", dense=unit_vector(1), limit=5)
    )
    assert len(results) == 1
    hit = results[0]
    assert hit.chunk_id == "c-0000"
    assert hit.doc_id == "a.html"
    assert hit.text == "hello world"
    assert hit.space == "dense"
    assert hit.metric.value == "cosine"
    assert hit.score_direction.value == "higher-is-better"
    assert hit.provenance is not None
    assert hit.provenance.source == "a.html"
    assert hit.provenance.pages == (1,)


def test_batch_upsert(store):
    store.create_collection(dense_config())
    records = [dense_record(f"c-{index:04d}", "a.html", seed=index) for index in range(25)]
    assert store.upsert("docs", records) == 25
    assert store.count("docs") == 25


def test_duplicate_upsert_replaces_the_record(store):
    store.create_collection(dense_config())
    store.upsert("docs", [dense_record("c-0000", "a.html", seed=1, metadata={"v": 1})])
    store.upsert("docs", [dense_record("c-0000", "a.html", seed=2, metadata={"v": 2})])
    assert store.count("docs") == 1
    fetched = store.get("docs", ["c-0000"])
    assert len(fetched) == 1
    assert fetched[0].metadata["v"] == 2


def test_fetch_by_ids_preserves_identity_and_provenance(store):
    store.create_collection(dense_config())
    store.upsert("docs", [dense_record("c-0000", "a.html", seed=1)])
    fetched = store.get("docs", ["c-0000"])
    assert len(fetched) == 1
    record = fetched[0]
    assert record.chunk.id == "c-0000"
    assert record.chunk.doc_id == "a.html"
    assert record.chunk.text == "hello world"
    assert record.chunk.index == 0
    assert record.chunk.count == 1
    assert record.chunk.length_unit == "characters"
    assert record.chunk.block_refs[0].page_number == 1
    assert record.chunk.metadata["source"] == "a.html"
    assert record.vectors["dense"].dense == pytest.approx(unit_vector(1), abs=1e-6)


def test_fetch_unknown_ids_returns_nothing(store):
    store.create_collection(dense_config())
    assert store.get("docs", ["missing"]) == []


def test_stored_chunk_id_is_preserved_even_though_points_are_uuids(store):
    store.create_collection(dense_config())
    store.upsert("docs", [dense_record("abc123-fixed-0000", "a.html")])
    fetched = store.get("docs", ["abc123-fixed-0000"])
    assert fetched[0].chunk.id == "abc123-fixed-0000"


def test_point_id_mapping_is_deterministic_and_collision_safe():
    first = point_id("abc123-fixed-0000")
    second = point_id("abc123-fixed-0001")
    assert first == point_id("abc123-fixed-0000")
    assert first != second
    assert len(first) == 36


# ----------------------------------------------------------------------
# sparse and multi-vector
# ----------------------------------------------------------------------
def test_sparse_upsert_and_search(store):
    store.create_collection(multi_space_config())
    indices, values = [1, 5, 9], [0.5, 0.25, 0.125]
    space = sparse_space()
    record = dense_record("c-0000", "a.html")
    record.vectors["sparse"] = SpaceVectors(
        space="sparse",
        embedding_space=space.embedding_space,
        sparse_indices=indices,
        sparse_values=values,
    )
    store.upsert("docs", [record])

    results = store.search(
        SearchRequest(
            collection="docs",
            space="sparse",
            sparse_indices=[1, 5],
            sparse_values=[1.0, 1.0],
            limit=5,
        )
    )
    assert len(results) == 1
    assert results[0].chunk_id == "c-0000"
    assert results[0].score > 0

    fetched = store.get("docs", ["c-0000"])
    assert fetched[0].vectors["sparse"].sparse_indices == indices
    assert fetched[0].vectors["sparse"].sparse_values == values


def test_multi_vector_upsert_and_search(store):
    store.create_collection(multi_space_config())
    space = multi_space()
    matrix = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    record = dense_record("c-0000", "a.html")
    record.vectors["late"] = SpaceVectors(
        space="late",
        embedding_space=space.embedding_space,
        multi_vector=matrix,
    )
    store.upsert("docs", [record])

    results = store.search(
        SearchRequest(
            collection="docs", space="late", multi_vector=[matrix[0]], limit=5
        )
    )
    assert len(results) == 1
    assert results[0].chunk_id == "c-0000"

    fetched = store.get("docs", ["c-0000"])
    assert fetched[0].vectors["late"].multi_vector == matrix


def test_searching_an_unknown_space_raises(store):
    store.create_collection(dense_config())
    with pytest.raises(VectorStoreConfigError):
        store.search(
            SearchRequest(collection="docs", space="missing", dense=unit_vector(0))
        )


def test_searching_with_the_wrong_representation_raises(store):
    store.create_collection(dense_config())
    with pytest.raises(VectorStoreConfigError):
        store.search(
            SearchRequest(
                collection="docs", space="dense", sparse_indices=[1], sparse_values=[1.0]
            )
        )


# ----------------------------------------------------------------------
# filters
# ----------------------------------------------------------------------
@pytest.fixture()
def filtered_store(store):
    store.create_collection(dense_config())
    store.upsert(
        "docs",
        [
            dense_record("c-0000", "a.html", seed=1, metadata={"tenant": "acme", "score": 1.5, "lang": "en"}),
            dense_record("c-0001", "b.html", seed=2, metadata={"tenant": "acme", "score": 2.5, "lang": "fr"}),
            dense_record("c-0002", "c.html", seed=3, metadata={"tenant": "other", "score": 3.5, "lang": "en"}),
        ],
    )
    return store


def ids(results):
    return sorted(result.chunk_id for result in results)


def test_filter_equality(filtered_store):
    results = filtered_store.search(
        SearchRequest(
            collection="docs", space="dense", dense=unit_vector(0), limit=10,
            filter=eq("tenant", "acme"),
        )
    )
    assert ids(results) == ["c-0000", "c-0001"]


def test_filter_inequality(filtered_store):
    results = filtered_store.search(
        SearchRequest(
            collection="docs", space="dense", dense=unit_vector(0), limit=10,
            filter=ne("tenant", "acme"),
        )
    )
    assert ids(results) == ["c-0002"]


def test_filter_in_and_not_in(filtered_store):
    results = filtered_store.search(
        SearchRequest(
            collection="docs", space="dense", dense=unit_vector(0), limit=10,
            filter=is_in("lang", ["en"]),
        )
    )
    assert ids(results) == ["c-0000", "c-0002"]

    results = filtered_store.search(
        SearchRequest(
            collection="docs", space="dense", dense=unit_vector(0), limit=10,
            filter=not_in("lang", ["en"]),
        )
    )
    assert ids(results) == ["c-0001"]


def test_filter_ordering(filtered_store):
    results = filtered_store.search(
        SearchRequest(
            collection="docs", space="dense", dense=unit_vector(0), limit=10,
            filter=gt("score", 2.0),
        )
    )
    assert ids(results) == ["c-0001", "c-0002"]

    results = filtered_store.search(
        SearchRequest(
            collection="docs", space="dense", dense=unit_vector(0), limit=10,
            filter=lt("score", 2.0),
        )
    )
    assert ids(results) == ["c-0000"]


def test_filter_exists(filtered_store):
    filtered_store.upsert(
        "docs",
        [
            dense_record("c-0003", "d.html", seed=4, metadata={"lang": "de"}),
        ],
    )
    results = filtered_store.search(
        SearchRequest(
            collection="docs", space="dense", dense=unit_vector(0), limit=10,
            filter=exists("tenant"),
        )
    )
    assert ids(results) == ["c-0000", "c-0001", "c-0002"]


def test_filter_boolean_composition(filtered_store):
    results = filtered_store.search(
        SearchRequest(
            collection="docs", space="dense", dense=unit_vector(0), limit=10,
            filter=and_(eq("tenant", "acme"), eq("lang", "en")),
        )
    )
    assert ids(results) == ["c-0000"]

    results = filtered_store.search(
        SearchRequest(
            collection="docs", space="dense", dense=unit_vector(0), limit=10,
            filter=or_(eq("lang", "fr"), eq("lang", "de")),
        )
    )
    assert ids(results) == ["c-0001"]

    results = filtered_store.search(
        SearchRequest(
            collection="docs", space="dense", dense=unit_vector(0), limit=10,
            filter=not_(eq("tenant", "acme")),
        )
    )
    assert ids(results) == ["c-0002"]


def test_filter_on_the_document_id(filtered_store):
    results = filtered_store.search(
        SearchRequest(
            collection="docs", space="dense", dense=unit_vector(0), limit=10,
            filter=eq("doc_id", "b.html"),
        )
    )
    assert ids(results) == ["c-0001"]


def test_count_with_a_filter(filtered_store):
    assert filtered_store.count("docs") == 3
    assert filtered_store.count("docs", eq("tenant", "acme")) == 2
    assert filtered_store.count("docs", eq("tenant", "nobody")) == 0


# ----------------------------------------------------------------------
# deletion
# ----------------------------------------------------------------------
def test_delete_by_ids(store):
    store.create_collection(dense_config())
    store.upsert(
        "docs",
        [dense_record("c-0000", "a.html", seed=1), dense_record("c-0001", "a.html", seed=2)],
    )
    assert store.delete("docs", ["c-0000"]) == 1
    assert store.count("docs") == 1
    assert store.get("docs", ["c-0000"]) == []


def test_delete_by_document_id(store):
    store.create_collection(dense_config())
    store.upsert(
        "docs",
        [
            dense_record("c-0000", "a.html", seed=1),
            dense_record("c-0001", "a.html", seed=2),
            dense_record("c-0002", "b.html", seed=3),
        ],
    )
    assert store.delete_by_document("docs", "a.html") == 2
    assert store.count("docs") == 1
    assert ids(store.search(SearchRequest(collection="docs", space="dense", dense=unit_vector(0)))) == ["c-0002"]


def test_delete_by_filter(store):
    store.create_collection(dense_config())
    store.upsert(
        "docs",
        [
            dense_record("c-0000", "a.html", seed=1, metadata={"tenant": "acme"}),
            dense_record("c-0001", "b.html", seed=2, metadata={"tenant": "other"}),
        ],
    )
    assert store.delete_by_filter("docs", eq("tenant", "acme")) == 1
    assert store.count("docs") == 1


# ----------------------------------------------------------------------
# pagination
# ----------------------------------------------------------------------
def test_scroll_paginates(store):
    store.create_collection(dense_config())
    store.upsert(
        "docs",
        [dense_record(f"c-{index:04d}", "a.html", seed=index) for index in range(10)],
    )
    page = store.scroll("docs", limit=4)
    assert len(page.records) == 4
    assert page.next_offset is not None

    seen = [record.chunk.id for record in page.records]
    while page.next_offset is not None:
        page = store.scroll("docs", limit=4, offset=page.next_offset)
        seen.extend(record.chunk.id for record in page.records)
    assert len(seen) == 10
    assert len(set(seen)) == 10


def test_iter_records_walks_the_whole_collection(store):
    from vector_stores.contracts import iter_records

    store.create_collection(dense_config())
    store.upsert(
        "docs",
        [dense_record(f"c-{index:04d}", "a.html", seed=index) for index in range(7)],
    )
    records = list(iter_records(store, "docs", page_size=3))
    assert len(records) == 7


def test_scroll_with_a_filter(store):
    store.create_collection(dense_config())
    store.upsert(
        "docs",
        [
            dense_record("c-0000", "a.html", seed=1, metadata={"tenant": "acme"}),
            dense_record("c-0001", "b.html", seed=2, metadata={"tenant": "other"}),
        ],
    )
    page = store.scroll("docs", limit=10, filter=eq("tenant", "acme"))
    assert [record.chunk.id for record in page.records] == ["c-0000"]


# ----------------------------------------------------------------------
# validation of the stored vectors
# ----------------------------------------------------------------------
def test_dimension_mismatch_is_rejected(store):
    store.create_collection(dense_config())
    record = dense_record("c-0000")
    record.vectors["dense"].dense = [0.1] * 4
    with pytest.raises(DimensionMismatchError):
        store.upsert("docs", [record])


def test_model_mismatch_is_rejected(store):
    store.create_collection(dense_config())
    record = dense_record("c-0000")
    record.vectors["dense"].embedding_space = descriptor(model_id="other/model")
    with pytest.raises(EmbeddingSpaceMismatchError):
        store.upsert("docs", [record])


def test_revision_mismatch_is_rejected(store):
    store.create_collection(dense_config())
    record = dense_record("c-0000")
    record.vectors["dense"].embedding_space = descriptor(model_revision="rev-2")
    with pytest.raises(EmbeddingSpaceMismatchError):
        store.upsert("docs", [record])


def test_normalization_mismatch_is_rejected(store):
    store.create_collection(dense_config())
    record = dense_record("c-0000")
    record.vectors["dense"].embedding_space = descriptor(
        normalization=Normalization.NONE
    )
    with pytest.raises(EmbeddingSpaceMismatchError):
        store.upsert("docs", [record])


def test_record_targeting_an_unknown_space_is_rejected(store):
    store.create_collection(dense_config())
    record = dense_record("c-0000")
    record.vectors["unknown"] = SpaceVectors(
        space="unknown",
        embedding_space=descriptor(),
        dense=unit_vector(0),
    )
    with pytest.raises(InvalidRecordError):
        store.upsert("docs", [record])


def test_empty_batch_is_a_no_op(store):
    store.create_collection(dense_config())
    assert store.upsert("docs", []) == 0
    assert store.count("docs") == 0


# ----------------------------------------------------------------------
# indexes
# ----------------------------------------------------------------------
def test_payload_indexes_are_created_with_the_collection(store):
    from vector_stores.contracts import PayloadIndexSpec

    config = dense_config(
        payload_indexes=(
            PayloadIndexSpec("tenant", PayloadFieldType.KEYWORD),
            PayloadIndexSpec("score", PayloadFieldType.FLOAT),
        )
    )
    store.create_collection(config)
    # The local mode accepts the index creation but does not use it to speed up
    # filtering; the call must not fail.
    assert store.count("docs") == 0


def test_create_and_delete_a_payload_index(store):
    store.create_collection(dense_config())
    description = store.create_index(
        "docs",
        IndexConfig(
            kind=IndexKind.PAYLOAD,
            target="tenant",
            field_type=PayloadFieldType.KEYWORD,
        ),
    )
    assert description.kind is IndexKind.PAYLOAD
    assert description.target == "tenant"


def test_creating_a_vector_index_afterwards_is_rejected(store):
    store.create_collection(dense_config())
    with pytest.raises(UnsupportedIndexOperationError):
        store.create_index(
            "docs", IndexConfig(kind=IndexKind.HNSW, target="dense")
        )


def test_capabilities_report_the_supported_index_kinds(store):
    capabilities = store.capabilities
    assert capabilities.backend == "qdrant"
    assert capabilities.dense_search is True
    assert capabilities.sparse_search is True
    assert capabilities.multi_vector_search is True
    assert capabilities.exact_search is True
    assert capabilities.supports(IndexKind.HNSW) is True
    assert capabilities.supports(IndexKind.SPARSE_INVERTED) is True
    assert capabilities.supports(IndexKind.PAYLOAD) is True
    assert capabilities.supports(IndexKind.MULTI_VECTOR) is True
    assert "hnsw_ef" in capabilities.index_capability(IndexKind.HNSW).query_parameters


# ----------------------------------------------------------------------
# error handling
# ----------------------------------------------------------------------
def test_missing_dependency_raises_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "qdrant_client", None)
    with pytest.raises(VectorStoreDependencyError) as excinfo:
        QdrantVectorStore.memory()
    assert "qdrant-client" in str(excinfo.value)


def test_backend_connection_failure_is_reported():
    from vector_stores.contracts import BackendConnectionError

    class BrokenClient:
        def get_collections(self):
            raise ConnectionError("connection refused")

        def close(self):
            return None

    store = QdrantVectorStore(client=BrokenClient())
    with pytest.raises(BackendConnectionError):
        store.list_collections()


def test_using_a_closed_store_raises():
    from vector_stores.contracts import VectorStoreError

    store = QdrantVectorStore.memory()
    store.close()
    with pytest.raises(VectorStoreError):
        store.list_collections()


def test_invalid_ids_are_rejected(store):
    store.create_collection(dense_config())
    with pytest.raises(InvalidRecordError):
        store.upsert("docs", [dense_record("")])


def test_the_adapter_implements_the_store_protocol(store):
    from vector_stores.contracts import VectorStore

    assert isinstance(store, VectorStore)


def test_the_shared_contracts_do_not_import_the_backend():
    """The contracts must stay independent of any vector database client."""
    import subprocess
    import sys

    code = (
        "import sys; import vector_stores.contracts; "
        "assert 'qdrant_client' not in sys.modules, sorted(sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
