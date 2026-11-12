"""Reusable conformance tests shared by every vector-store adapter.

The tests in this module are collected by the backend test modules, which
import them and provide the ``store`` fixture (see
``tests/vector_stores/conftest.py``).  They only use the shared contracts,
so an adapter that passes them behaves like every other adapter:

- collections, records, ids, pagination and counts;
- metadata and provenance preservation;
- embedding-space compatibility enforcement;
- the portable filter language;
- index description and the recall of the approximate search against the
  exact-search baseline.

Backend specific behaviour (persistence formats, native index types, sparse
support) lives in the backend test modules.
"""

from __future__ import annotations

import math

import pytest

from chunkers.contracts.models import BlockReference, Chunk
from embeddings.contracts.models import (
    EmbeddingSpaceDescriptor,
    Normalization,
    RepresentationType,
)
from vector_stores.contracts import (
    CollectionConfig,
    DistanceMetric,
    EmbeddingSpaceMismatchError,
    IndexConfig,
    IndexKind,
    InvalidRecordError,
    PayloadFieldType,
    PayloadIndexSpec,
    SearchRequest,
    SpaceVectors,
    StoredRecord,
    UnsupportedFilterOperationError,
    VectorSpaceConfig,
    eq,
    exists,
    gt,
    gte,
    is_in,
    lt,
    lte,
    ne,
    not_in,
)

MODEL = "conformance/model"
REVISION = "rev-1"
DIMENSIONS = 16

#: The metadata fields every adapter declares and can therefore filter on.
FILTERABLE_FIELDS = ("tenant", "category", "score", "rank")


def unit_vector(seed: int, dimensions: int = DIMENSIONS) -> list[float]:
    """Return a deterministic unit vector for ``seed``."""
    raw = [
        math.sin(seed * (index + 1) * 0.7351) + 0.05 * (index + 1) * (seed % 3 + 1)
        for index in range(dimensions)
    ]
    norm = math.sqrt(sum(value * value for value in raw)) or 1.0
    return [value / norm for value in raw]


def make_descriptor(
    *,
    model_id: str = MODEL,
    model_revision: str | None = REVISION,
    dimensions: int | None = DIMENSIONS,
    representation_type: RepresentationType = RepresentationType.DENSE,
) -> EmbeddingSpaceDescriptor:
    return EmbeddingSpaceDescriptor(
        model_id=model_id,
        model_revision=model_revision,
        dimensions=dimensions,
        representation_type=representation_type,
        normalization=Normalization.L2,
    )


def make_space(
    name: str = "dense",
    *,
    dimensions: int = DIMENSIONS,
    distance: DistanceMetric = DistanceMetric.COSINE,
    model_id: str = MODEL,
    model_revision: str | None = REVISION,
) -> VectorSpaceConfig:
    return VectorSpaceConfig(
        name=name,
        vector_type=RepresentationType.DENSE,
        embedding_space=make_descriptor(
            model_id=model_id, model_revision=model_revision, dimensions=dimensions
        ),
        dimensions=dimensions,
        distance=distance,
    )


def make_config(name: str = "conformance", **kwargs) -> CollectionConfig:
    """Return a collection configuration with declared filterable fields."""
    return CollectionConfig(
        name=name,
        vector_spaces={"dense": make_space(**kwargs)},
        description="conformance collection",
        payload_indexes=(
            PayloadIndexSpec("tenant", PayloadFieldType.KEYWORD),
            PayloadIndexSpec("category", PayloadFieldType.KEYWORD),
            PayloadIndexSpec("score", PayloadFieldType.FLOAT),
            PayloadIndexSpec("rank", PayloadFieldType.INTEGER),
        ),
    )


def make_chunk(
    chunk_id: str = "abc123-fixed-0000",
    doc_id: str = "doc.html",
    *,
    index: int = 0,
    count: int = 2,
    text: str = "hello world",
) -> Chunk:
    return Chunk(
        id=chunk_id,
        doc_id=doc_id,
        text=text,
        index=index,
        count=count,
        length=len(text),
        length_unit="characters",
        block_refs=[BlockReference(0, 1, "text", 0, len(text))],
        heading="Intro",
        metadata={"source": doc_id, "pages": [1], "strategy": "fixed"},
    )


def make_record(
    chunk_id: str = "abc123-fixed-0000",
    doc_id: str = "doc.html",
    *,
    seed: int = 0,
    tenant: str = "acme",
    category: str = "docs",
    score: float = 1.5,
    rank: int = 3,
    space: VectorSpaceConfig | None = None,
    model_id: str = MODEL,
    dimensions: int = DIMENSIONS,
) -> StoredRecord:
    space = space or make_space(dimensions=dimensions)
    descriptor = make_descriptor(
        model_id=model_id, dimensions=dimensions, representation_type=space.vector_type
    )
    return StoredRecord(
        chunk=make_chunk(chunk_id, doc_id),
        vectors={
            space.name: SpaceVectors(
                space=space.name,
                embedding_space=descriptor,
                dense=unit_vector(seed, dimensions),
            )
        },
        metadata={
            "tenant": tenant,
            "category": category,
            "score": score,
            "rank": rank,
        },
    )


def make_corpus(count: int = 24) -> list[StoredRecord]:
    """Return a deterministic corpus of records spread over three documents."""
    records = []
    for index in range(count):
        records.append(
            make_record(
                f"chunk-{index:05d}",
                f"doc-{index % 3}.html",
                seed=index,
                tenant="acme" if index % 2 == 0 else "other",
                category="docs" if index % 3 == 0 else "code",
                score=float(index),
                rank=index,
            )
        )
    return records


# ----------------------------------------------------------------------
# core required behaviour
# ----------------------------------------------------------------------
def test_create_collection(store, collection):
    config = make_config(collection)
    info = store.create_collection(config)
    assert info.name == collection
    assert info.description == "conformance collection"
    assert set(info.vector_spaces) == {"dense"}
    assert info.vector_spaces["dense"].dimensions == DIMENSIONS
    assert info.point_count == 0


def test_create_collection_twice_raises(store, collection):
    store.create_collection(make_config(collection))
    with pytest.raises(Exception) as excinfo:
        store.create_collection(make_config(collection))
    assert "already exists" in str(excinfo.value).lower()


def test_describe_and_list_collections(store, collection):
    store.create_collection(make_config(collection))
    info = store.describe_collection(collection)
    assert info.name == collection
    assert info.backend is not None
    names = [item.name for item in store.list_collections()]
    assert collection in names


def test_delete_collection(store, collection):
    store.create_collection(make_config(collection))
    assert store.delete_collection(collection) is True
    assert store.delete_collection(collection) is False
    assert collection not in [item.name for item in store.list_collections()]


def test_missing_collection_raises(store, collection):
    from vector_stores.contracts import CollectionNotFoundError

    with pytest.raises(CollectionNotFoundError):
        store.describe_collection("missing")
    with pytest.raises(CollectionNotFoundError):
        store.count("missing")
    with pytest.raises(CollectionNotFoundError):
        store.upsert("missing", [make_record()])


def test_upsert_and_count(store, collection):
    store.create_collection(make_config(collection))
    assert store.upsert(collection, [make_record("c-0000", "a.html", seed=1)]) == 1
    assert store.count(collection) == 1


def test_batch_upsert(store, collection):
    store.create_collection(make_config(collection))
    records = make_corpus(12)
    assert store.upsert(collection, records) == 12
    assert store.count(collection) == 12


def test_empty_batch_is_a_no_op(store, collection):
    store.create_collection(make_config(collection))
    assert store.upsert(collection, []) == 0
    assert store.count(collection) == 0


def test_fetch_by_id_preserves_identity_and_provenance(store, collection):
    store.create_collection(make_config(collection))
    store.upsert(collection, [make_record("c-0000", "a.html", seed=1)])
    fetched = store.get(collection, ["c-0000"])
    assert len(fetched) == 1
    record = fetched[0]
    assert record.chunk.id == "c-0000"
    assert record.chunk.doc_id == "a.html"
    assert record.chunk.text == "hello world"
    assert record.chunk.index == 0
    assert record.chunk.count == 2
    assert record.chunk.length_unit == "characters"
    assert record.chunk.heading == "Intro"
    assert record.chunk.block_refs[0].page_number == 1
    assert record.chunk.block_refs[0].block_index == 0
    assert record.chunk.metadata["source"] == "a.html"
    assert record.chunk.metadata["pages"] == [1]


def test_fetch_unknown_id_returns_nothing(store, collection):
    store.create_collection(make_config(collection))
    assert store.get(collection, ["missing"]) == []


def test_metadata_is_preserved(store, collection):
    store.create_collection(make_config(collection))
    store.upsert(collection, [make_record("c-0000", tenant="acme", score=2.5, rank=7)])
    record = store.get(collection, ["c-0000"])[0]
    assert record.metadata["tenant"] == "acme"
    assert record.metadata["category"] == "docs"
    assert record.metadata["score"] == 2.5
    assert record.metadata["rank"] == 7


def test_duplicate_upsert_replaces_the_record(store, collection):
    store.create_collection(make_config(collection))
    store.upsert(collection, [make_record("c-0000", seed=1, score=1.0)])
    store.upsert(collection, [make_record("c-0000", seed=2, score=9.0)])
    assert store.count(collection) == 1
    record = store.get(collection, ["c-0000"])[0]
    assert record.metadata["score"] == 9.0


def test_delete_by_id(store, collection):
    store.create_collection(make_config(collection))
    store.upsert(
        collection,
        [make_record("c-0000", seed=1), make_record("c-0001", seed=2)],
    )
    assert store.delete(collection, ["c-0000"]) == 1
    assert store.count(collection) == 1
    assert store.get(collection, ["c-0000"]) == []


def test_delete_by_document_id(store, collection):
    store.create_collection(make_config(collection))
    store.upsert(
        collection,
        [
            make_record("c-0000", "a.html", seed=1),
            make_record("c-0001", "a.html", seed=2),
            make_record("c-0002", "b.html", seed=3),
        ],
    )
    assert store.delete_by_document(collection, "a.html") == 2
    assert store.count(collection) == 1


def test_delete_by_filter(store, collection):
    store.create_collection(make_config(collection))
    store.upsert(
        collection,
        [
            make_record("c-0000", tenant="acme", seed=1),
            make_record("c-0001", tenant="other", seed=2),
        ],
    )
    assert store.delete_by_filter(collection, eq("tenant", "acme")) == 1
    assert store.count(collection) == 1


def test_count_with_a_filter(store, collection):
    store.create_collection(make_config(collection))
    store.upsert(collection, make_corpus(12))
    assert store.count(collection, eq("tenant", "acme")) == 6
    assert store.count(collection, eq("tenant", "nobody")) == 0


def test_scroll_paginates_without_duplicates(store, collection):
    store.create_collection(make_config(collection))
    store.upsert(collection, make_corpus(10))
    seen: list[str] = []
    offset = None
    pages = 0
    while True:
        page = store.scroll(collection, limit=3, offset=offset)
        seen.extend(record.chunk.id for record in page.records)
        pages += 1
        if page.next_offset is None:
            break
        offset = page.next_offset
        assert pages < 20
    assert len(seen) == 10
    assert len(set(seen)) == 10


def test_scroll_with_a_filter(store, collection):
    store.create_collection(make_config(collection))
    store.upsert(collection, make_corpus(8))
    page = store.scroll(collection, limit=10, filter=eq("tenant", "acme"))
    assert len(page.records) == 4
    assert all(record.metadata["tenant"] == "acme" for record in page.records)


def test_iter_records_walks_the_whole_collection(store, collection):
    from vector_stores.contracts import iter_records

    store.create_collection(make_config(collection))
    store.upsert(collection, make_corpus(7))
    records = list(iter_records(store, collection, page_size=3))
    assert len(records) == 7


# ----------------------------------------------------------------------
# search
# ----------------------------------------------------------------------
def test_dense_exact_search_returns_the_closest_vector_first(store, collection):
    store.create_collection(make_config(collection))
    store.upsert(collection, make_corpus(24))
    results = store.search(
        SearchRequest(
            collection=collection,
            space="dense",
            dense=unit_vector(5),
            limit=5,
            params={"exact": True},
        )
    )
    assert results
    assert results[0].chunk_id == "chunk-00005"
    assert results[0].doc_id == "doc-2.html"
    assert results[0].space == "dense"
    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)


def test_search_results_preserve_the_payload(store, collection):
    store.create_collection(make_config(collection))
    store.upsert(collection, [make_record("c-0000", "a.html", seed=1)])
    result = store.search(
        SearchRequest(
            collection=collection, space="dense", dense=unit_vector(1), limit=1
        )
    )[0]
    assert result.chunk_id == "c-0000"
    assert result.doc_id == "a.html"
    assert result.text == "hello world"
    assert result.metadata["tenant"] == "acme"
    assert result.provenance is not None
    assert result.provenance.source == "a.html"
    assert result.provenance.pages == (1,)
    assert result.metric is not None
    assert result.score_direction is not None


def test_search_limit_is_respected(store, collection):
    store.create_collection(make_config(collection))
    store.upsert(collection, make_corpus(20))
    results = store.search(
        SearchRequest(
            collection=collection, space="dense", dense=unit_vector(0), limit=3
        )
    )
    assert len(results) == 3


def test_search_on_an_unknown_space_raises(store, collection):
    from vector_stores.contracts import VectorStoreConfigError

    store.create_collection(make_config(collection))
    with pytest.raises(VectorStoreConfigError):
        store.search(
            SearchRequest(
                collection=collection, space="missing", dense=unit_vector(0)
            )
        )


def test_dimension_mismatch_is_rejected(store, collection):
    from vector_stores.contracts import DimensionMismatchError

    store.create_collection(make_config(collection))
    record = make_record("c-0000")
    record.vectors["dense"].dense = [0.1] * (DIMENSIONS - 1)
    with pytest.raises(DimensionMismatchError):
        store.upsert(collection, [record])


def test_model_identity_mismatch_is_rejected(store, collection):
    store.create_collection(make_config(collection))
    record = make_record("c-0000")
    record.vectors["dense"].embedding_space = make_descriptor(model_id="other/model")
    with pytest.raises(EmbeddingSpaceMismatchError):
        store.upsert(collection, [record])


def test_model_revision_mismatch_is_rejected(store, collection):
    store.create_collection(make_config(collection))
    record = make_record("c-0000")
    record.vectors["dense"].embedding_space = make_descriptor(model_revision="rev-2")
    with pytest.raises(EmbeddingSpaceMismatchError):
        store.upsert(collection, [record])


def test_record_targeting_an_unknown_space_is_rejected(store, collection):
    store.create_collection(make_config(collection))
    record = make_record("c-0000")
    record.vectors["other"] = SpaceVectors(
        space="other",
        embedding_space=make_descriptor(),
        dense=unit_vector(0),
    )
    with pytest.raises(InvalidRecordError):
        store.upsert(collection, [record])


def test_empty_chunk_id_is_rejected(store, collection):
    store.create_collection(make_config(collection))
    with pytest.raises(InvalidRecordError):
        store.upsert(collection, [make_record("", seed=1)])


# ----------------------------------------------------------------------
# filters
# ----------------------------------------------------------------------
def test_filter_equality(store, collection):
    store.create_collection(make_config(collection))
    store.upsert(collection, make_corpus(10))
    results = store.search(
        SearchRequest(
            collection=collection,
            space="dense",
            dense=unit_vector(0),
            limit=10,
            filter=eq("tenant", "acme"),
        )
    )
    assert len(results) == 5
    assert all(result.metadata["tenant"] == "acme" for result in results)


def test_filter_inequality(store, collection):
    store.create_collection(make_config(collection))
    store.upsert(collection, make_corpus(10))
    results = store.search(
        SearchRequest(
            collection=collection,
            space="dense",
            dense=unit_vector(0),
            limit=10,
            filter=ne("tenant", "acme"),
        )
    )
    assert len(results) == 5
    assert all(result.metadata["tenant"] == "other" for result in results)


def test_filter_membership(store, collection):
    store.create_collection(make_config(collection))
    store.upsert(collection, make_corpus(10))
    results = store.search(
        SearchRequest(
            collection=collection,
            space="dense",
            dense=unit_vector(0),
            limit=10,
            filter=is_in("tenant", ["acme", "nobody"]),
        )
    )
    assert len(results) == 5

    results = store.search(
        SearchRequest(
            collection=collection,
            space="dense",
            dense=unit_vector(0),
            limit=10,
            filter=not_in("tenant", ["acme"]),
        )
    )
    assert len(results) == 5


def test_filter_ordering(store, collection):
    store.create_collection(make_config(collection))
    store.upsert(collection, make_corpus(10))
    results = store.search(
        SearchRequest(
            collection=collection,
            space="dense",
            dense=unit_vector(0),
            limit=10,
            filter=gt("rank", 5),
        )
    )
    assert len(results) == 4

    results = store.search(
        SearchRequest(
            collection=collection,
            space="dense",
            dense=unit_vector(0),
            limit=10,
            filter=lt("rank", 5),
        )
    )
    assert len(results) == 5

    results = store.search(
        SearchRequest(
            collection=collection,
            space="dense",
            dense=unit_vector(0),
            limit=10,
            filter=gte("rank", 8),
        )
    )
    assert len(results) == 2

    results = store.search(
        SearchRequest(
            collection=collection,
            space="dense",
            dense=unit_vector(0),
            limit=10,
            filter=lte("rank", 1),
        )
    )
    assert len(results) == 2


def test_filter_exists(store, collection):
    store.create_collection(make_config(collection))
    store.upsert(collection, make_corpus(6))
    results = store.search(
        SearchRequest(
            collection=collection,
            space="dense",
            dense=unit_vector(0),
            limit=10,
            filter=exists("tenant"),
        )
    )
    assert len(results) == 6


def test_filter_boolean_composition(store, collection):
    from vector_stores.contracts import and_, not_, or_

    store.create_collection(make_config(collection))
    store.upsert(collection, make_corpus(12))
    results = store.search(
        SearchRequest(
            collection=collection,
            space="dense",
            dense=unit_vector(0),
            limit=20,
            filter=and_(eq("tenant", "acme"), eq("category", "docs")),
        )
    )
    assert len(results) == 2

    results = store.search(
        SearchRequest(
            collection=collection,
            space="dense",
            dense=unit_vector(0),
            limit=20,
            filter=or_(eq("category", "code"), eq("rank", 0)),
        )
    )
    assert len(results) == 9

    results = store.search(
        SearchRequest(
            collection=collection,
            space="dense",
            dense=unit_vector(0),
            limit=20,
            filter=not_(eq("tenant", "acme")),
        )
    )
    assert len(results) == 6


def test_filter_on_the_document_id(store, collection):
    store.create_collection(make_config(collection))
    store.upsert(collection, make_corpus(9))
    results = store.search(
        SearchRequest(
            collection=collection,
            space="dense",
            dense=unit_vector(0),
            limit=20,
            filter=eq("doc_id", "doc-0.html"),
        )
    )
    assert len(results) == 3


def test_unsupported_filter_operation_is_reported(store, collection):
    """A filter the adapter cannot express must raise, never be ignored."""
    capabilities = store.capabilities
    from vector_stores.contracts import FilterOp

    if FilterOp.EXISTS in capabilities.filter_ops:
        pytest.skip("the adapter supports EXISTS")
    with pytest.raises(UnsupportedFilterOperationError):
        store.count(collection, exists("tenant"))


# ----------------------------------------------------------------------
# indexes
# ----------------------------------------------------------------------
def test_list_indexes_describes_the_indexes(store, collection):
    store.create_collection(make_config(collection))
    store.upsert(collection, make_corpus(4))
    indexes = store.list_indexes(collection)
    assert isinstance(indexes, list)
    for index in indexes:
        assert index.kind in tuple(IndexKind)
        assert index.target


def test_create_payload_index(store, collection):
    capabilities = store.capabilities
    if not capabilities.supports(IndexKind.PAYLOAD):
        pytest.skip("the adapter does not support payload indexes")
    store.create_collection(make_config(collection))
    description = store.create_index(
        collection,
        IndexConfig(
            kind=IndexKind.PAYLOAD,
            target="tenant",
            field_type=PayloadFieldType.KEYWORD,
        ),
    )
    assert description.kind is IndexKind.PAYLOAD
    assert description.target == "tenant"


def test_index_creation_validates_the_target(store, collection):
    capabilities = store.capabilities
    if not capabilities.supports(IndexKind.PAYLOAD):
        pytest.skip("the adapter does not support payload indexes")
    store.create_collection(make_config(collection))
    index = IndexConfig(
        kind=IndexKind.PAYLOAD,
        target="undeclared_field",
        field_type=PayloadFieldType.KEYWORD,
    )
    if store.capabilities.filter_requires_declared_fields:
        # The backend has a fixed schema: it must reject the field instead of
        # pretending the index exists.
        with pytest.raises(UnsupportedFilterOperationError):
            store.create_index(collection, index)
    else:
        description = store.create_index(collection, index)
        assert description.target == "undeclared_field"


def test_approximate_search_recall_against_the_exact_baseline(store, collection):
    """The approximate path must return (almost) the same neighbours."""
    capabilities = store.capabilities
    if not capabilities.supports(IndexKind.HNSW):
        pytest.skip("the adapter does not support an approximate dense index")
    store.create_collection(make_config(collection))
    store.upsert(collection, make_corpus(400))
    exact = store.search(
        SearchRequest(
            collection=collection,
            space="dense",
            dense=unit_vector(11),
            limit=10,
            params={"exact": True},
        )
    )
    approximate = store.search(
        SearchRequest(
            collection=collection,
            space="dense",
            dense=unit_vector(11),
            limit=10,
        )
    )
    assert len(exact) == 10
    assert len(approximate) == 10
    expected = {result.chunk_id for result in exact}
    found = {result.chunk_id for result in approximate}
    recall = len(expected & found) / len(expected)
    assert recall >= 0.8, f"recall@10 too low: {recall}"
