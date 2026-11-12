"""Contract tests: collection configuration, records, filters, requests, capabilities.

These tests are backend independent: they exercise the shared contracts only,
never a vector database client.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from chunkers.contracts.models import BlockReference, Chunk
from embeddings.contracts.models import (
    EmbeddingResult,
    EmbeddingSpaceDescriptor,
    Normalization,
    RepresentationType,
)
from vector_stores.contracts import (
    CollectionConfig,
    DistanceMetric,
    HnswIndexConfig,
    IndexConfig,
    IndexKind,
    MetadataFilter,
    PayloadFieldType,
    PayloadIndexSpec,
    ScoreDirection,
    SearchRequest,
    SearchResult,
    SpaceVectors,
    SparseIndexConfig,
    StoredRecord,
    VectorSpaceConfig,
    VectorStoreConfigError,
    VectorStoreError,
    and_,
    assert_embedding_space_compatible,
    eq,
    exists,
    gt,
    gte,
    is_in,
    lt,
    lte,
    ne,
    not_,
    not_in,
    or_,
    validate_collection_config,
    validate_filter,
    validate_record,
    validate_search_request,
    validate_space_vectors,
)

MODEL = "test/embedding-model"
REVISION = "rev-1"


def descriptor(
    *,
    model_id: str = MODEL,
    model_revision: str | None = REVISION,
    dimensions: int | None = 8,
    representation_type: RepresentationType = RepresentationType.DENSE,
    normalization: Normalization = Normalization.L2,
) -> EmbeddingSpaceDescriptor:
    return EmbeddingSpaceDescriptor(
        model_id=model_id,
        model_revision=model_revision,
        dimensions=dimensions,
        representation_type=representation_type,
        normalization=normalization,
    )


def dense_space(
    name: str = "dense",
    *,
    dimensions: int = 8,
    distance: DistanceMetric = DistanceMetric.COSINE,
    model_id: str = MODEL,
    model_revision: str | None = REVISION,
    normalization: Normalization = Normalization.L2,
    hnsw: HnswIndexConfig | None = None,
) -> VectorSpaceConfig:
    return VectorSpaceConfig(
        name=name,
        vector_type=RepresentationType.DENSE,
        embedding_space=descriptor(
            model_id=model_id,
            model_revision=model_revision,
            dimensions=dimensions,
            normalization=normalization,
        ),
        dimensions=dimensions,
        distance=distance,
        hnsw=hnsw,
    )


def sparse_space(name: str = "sparse") -> VectorSpaceConfig:
    return VectorSpaceConfig(
        name=name,
        vector_type=RepresentationType.SPARSE,
        embedding_space=descriptor(
            dimensions=None, representation_type=RepresentationType.SPARSE
        ),
    )


def multi_space(name: str = "late") -> VectorSpaceConfig:
    return VectorSpaceConfig(
        name=name,
        vector_type=RepresentationType.MULTI_VECTOR,
        embedding_space=descriptor(representation_type=RepresentationType.MULTI_VECTOR),
        dimensions=8,
        distance=DistanceMetric.COSINE,
    )


def chunk(chunk_id: str = "abc123-fixed-0000", doc_id: str = "doc.html") -> Chunk:
    return Chunk(
        id=chunk_id,
        doc_id=doc_id,
        text="hello world",
        index=0,
        count=1,
        length=11,
        length_unit="characters",
        block_refs=[BlockReference(0, 1, "text", 0, 11)],
        heading="Intro",
        metadata={"source": "doc.html", "pages": [1]},
    )


def record(
    *,
    space: VectorSpaceConfig | None = None,
    chunk_id: str = "abc123-fixed-0000",
    doc_id: str = "doc.html",
    dense: list[float] | None = None,
    metadata: dict | None = None,
) -> StoredRecord:
    space = space or dense_space()
    return StoredRecord(
        chunk=chunk(chunk_id, doc_id),
        vectors={
            space.name: SpaceVectors(
                space=space.name,
                embedding_space=space.embedding_space,
                dense=dense if dense is not None else [0.1] * 8,
            )
        },
        metadata=metadata or {},
    )


# ----------------------------------------------------------------------
# collection configuration
# ----------------------------------------------------------------------
def test_valid_collection_config_is_accepted():
    config = CollectionConfig(
        name="docs",
        vector_spaces={"dense": dense_space(), "sparse": sparse_space()},
        description="test collection",
        payload_indexes=(PayloadIndexSpec("tenant", PayloadFieldType.KEYWORD),),
    )
    validate_collection_config(config)
    assert config.space("dense").dimensions == 8
    assert "sparse" in config.vector_spaces


def test_collection_config_requires_a_name():
    config = CollectionConfig(name="  ", vector_spaces={"dense": dense_space()})
    with pytest.raises(VectorStoreConfigError):
        validate_collection_config(config)


def test_collection_config_requires_at_least_one_vector_space():
    config = CollectionConfig(name="docs", vector_spaces={})
    with pytest.raises(VectorStoreConfigError):
        validate_collection_config(config)


def test_dense_space_requires_dimensions():
    space = replace(dense_space(), dimensions=None)
    config = CollectionConfig(name="docs", vector_spaces={"dense": space})
    with pytest.raises(VectorStoreConfigError):
        validate_collection_config(config)


def test_dense_space_rejects_non_positive_dimensions():
    space = replace(dense_space(), dimensions=0)
    config = CollectionConfig(name="docs", vector_spaces={"dense": space})
    with pytest.raises(VectorStoreConfigError):
        validate_collection_config(config)


def test_sparse_space_must_not_declare_dimensions():
    space = replace(sparse_space(), dimensions=8)
    config = CollectionConfig(name="docs", vector_spaces={"sparse": space})
    with pytest.raises(VectorStoreConfigError):
        validate_collection_config(config)


def test_space_name_must_match_its_key():
    config = CollectionConfig(name="docs", vector_spaces={"dense": dense_space("other")})
    with pytest.raises(VectorStoreConfigError):
        validate_collection_config(config)


def test_space_requires_a_model_id():
    space = replace(dense_space(), embedding_space=descriptor(model_id=""))
    config = CollectionConfig(name="docs", vector_spaces={"dense": space})
    with pytest.raises(VectorStoreConfigError):
        validate_collection_config(config)


def test_multi_vector_space_rejects_manhattan_metric():
    space = replace(multi_space(), distance=DistanceMetric.MANHATTAN)
    config = CollectionConfig(name="docs", vector_spaces={"late": space})
    with pytest.raises(VectorStoreConfigError):
        validate_collection_config(config)


def test_duplicate_payload_index_is_rejected():
    config = CollectionConfig(
        name="docs",
        vector_spaces={"dense": dense_space()},
        payload_indexes=(
            PayloadIndexSpec("tenant", PayloadFieldType.KEYWORD),
            PayloadIndexSpec("tenant", PayloadFieldType.INTEGER),
        ),
    )
    with pytest.raises(VectorStoreConfigError):
        validate_collection_config(config)


def test_collection_config_round_trips_through_as_dict():
    config = CollectionConfig(
        name="docs",
        vector_spaces={"dense": dense_space(), "sparse": sparse_space()},
        description="round trip",
        payload_indexes=(PayloadIndexSpec("tenant", PayloadFieldType.KEYWORD),),
    )
    restored = CollectionConfig.from_dict(config.as_dict())
    assert restored.name == config.name
    assert restored.description == config.description
    assert set(restored.vector_spaces) == set(config.vector_spaces)
    assert restored.space("dense").embedding_space == config.space("dense").embedding_space
    assert restored.payload_indexes == config.payload_indexes


def test_distance_metric_declares_the_score_direction():
    assert DistanceMetric.COSINE.score_direction is ScoreDirection.HIGHER_IS_BETTER
    assert DistanceMetric.DOT.score_direction is ScoreDirection.HIGHER_IS_BETTER
    assert DistanceMetric.EUCLIDEAN.score_direction is ScoreDirection.LOWER_IS_BETTER
    assert DistanceMetric.MANHATTAN.score_direction is ScoreDirection.LOWER_IS_BETTER


# ----------------------------------------------------------------------
# embedding space compatibility
# ----------------------------------------------------------------------
def test_same_model_and_revision_is_compatible():
    space = dense_space()
    assert_embedding_space_compatible(space, space.embedding_space)


def test_different_model_is_rejected():
    space = dense_space()
    other = descriptor(model_id="other/model")
    with pytest.raises(VectorStoreError):
        assert_embedding_space_compatible(space, other)


def test_different_revision_is_rejected():
    space = dense_space()
    other = descriptor(model_revision="rev-2")
    with pytest.raises(VectorStoreError):
        assert_embedding_space_compatible(space, other)


def test_same_dimensions_but_different_model_is_rejected():
    """Matching dimensions alone never establish compatibility."""
    space = dense_space()
    other = descriptor(model_id="another/model")
    assert space.embedding_space.dimensions == other.dimensions
    with pytest.raises(VectorStoreError):
        assert_embedding_space_compatible(space, other)


def test_different_normalization_is_rejected():
    space = dense_space()
    other = descriptor(normalization=Normalization.NONE)
    with pytest.raises(VectorStoreError):
        assert_embedding_space_compatible(space, other)


def test_sparse_space_ignores_the_model_width():
    space = sparse_space()
    other = descriptor(dimensions=None, representation_type=RepresentationType.SPARSE)
    assert_embedding_space_compatible(space, other)


# ----------------------------------------------------------------------
# stored records
# ----------------------------------------------------------------------
def test_valid_record_is_accepted():
    config = CollectionConfig(name="docs", vector_spaces={"dense": dense_space()})
    validate_record(record(), config)


def test_record_with_wrong_dimension_is_rejected():
    config = CollectionConfig(name="docs", vector_spaces={"dense": dense_space()})
    with pytest.raises(VectorStoreError):
        validate_record(record(dense=[0.1] * 4), config)


def test_record_from_another_model_is_rejected():
    config = CollectionConfig(name="docs", vector_spaces={"dense": dense_space()})
    bad = record()
    bad.vectors["dense"].embedding_space = descriptor(model_id="other/model")
    with pytest.raises(VectorStoreError):
        validate_record(bad, config)


def test_record_targeting_an_unknown_space_is_rejected():
    config = CollectionConfig(name="docs", vector_spaces={"dense": dense_space()})
    bad = record()
    bad.vectors["sparse"] = SpaceVectors(
        space="sparse",
        embedding_space=sparse_space().embedding_space,
        sparse_indices=[1],
        sparse_values=[0.5],
    )
    with pytest.raises(VectorStoreError):
        validate_record(bad, config)


def test_record_without_vector_is_rejected():
    config = CollectionConfig(name="docs", vector_spaces={"dense": dense_space()})
    empty = StoredRecord(chunk=chunk())
    with pytest.raises(VectorStoreError):
        validate_record(empty, config)


def test_dense_space_rejects_a_sparse_only_record():
    space = dense_space()
    vectors = SpaceVectors(
        space="dense",
        embedding_space=space.embedding_space,
        sparse_indices=[1],
        sparse_values=[0.5],
    )
    with pytest.raises(VectorStoreError):
        validate_space_vectors(space, vectors)


def test_sparse_space_requires_matching_index_and_value_counts():
    space = sparse_space()
    vectors = SpaceVectors(
        space="sparse",
        embedding_space=space.embedding_space,
        sparse_indices=[1, 2],
        sparse_values=[0.5],
    )
    with pytest.raises(VectorStoreError):
        validate_space_vectors(space, vectors)


def test_multi_vector_space_rejects_rows_of_different_widths():
    space = multi_space()
    vectors = SpaceVectors(
        space="late",
        embedding_space=space.embedding_space,
        multi_vector=[[0.1] * 8, [0.1] * 4],
    )
    with pytest.raises(VectorStoreError):
        validate_space_vectors(space, vectors)


def test_multi_vector_space_keeps_the_full_matrix():
    space = multi_space()
    vectors = SpaceVectors(
        space="late",
        embedding_space=space.embedding_space,
        multi_vector=[[0.1] * 8, [0.2] * 8, [0.3] * 8],
    )
    validate_space_vectors(space, vectors)
    assert len(vectors.multi_vector) == 3


def test_record_from_embedding_result_keeps_only_the_selected_representation():
    result = EmbeddingResult(
        model_id=MODEL,
        dimensions=8,
        dense_vector=[0.1] * 8,
        sparse_indices=[1, 5],
        sparse_values=[0.5, 0.25],
        multi_vector=[[0.1] * 8],
        representation_type=RepresentationType.DENSE,
        normalization=Normalization.L2,
    )
    dense_record = StoredRecord.from_embedding_result(chunk(), result, space="dense")
    assert dense_record.vectors["dense"].dense is not None
    assert dense_record.vectors["dense"].sparse_indices is None

    sparse_record = StoredRecord.from_embedding_result(
        chunk(), result, space="sparse", vector_type=RepresentationType.SPARSE
    )
    assert sparse_record.vectors["sparse"].sparse_indices == [1, 5]
    assert sparse_record.vectors["sparse"].dense is None


# ----------------------------------------------------------------------
# filters
# ----------------------------------------------------------------------
def test_filter_builders_produce_the_expected_tree():
    node = and_(eq("a", 1), or_(ne("b", "x"), is_in("c", [1, 2])))
    assert node.op.value == "and"
    assert node.children[0].op.value == "eq"
    assert node.children[1].op.value == "or"


def test_valid_filters_are_accepted():
    validate_filter(eq("a", 1))
    validate_filter(ne("a", "x"))
    validate_filter(is_in("a", [1, 2]))
    validate_filter(not_in("a", [1, 2]))
    validate_filter(gt("a", 1))
    validate_filter(gte("a", 1))
    validate_filter(lt("a", 1))
    validate_filter(lte("a", 1))
    validate_filter(exists("a"))
    validate_filter(not_(eq("a", 1)))
    validate_filter(or_(eq("a", 1), eq("b", 2)))


def test_filter_without_a_field_is_rejected():
    with pytest.raises(VectorStoreConfigError):
        validate_filter(MetadataFilter(op=eq("a", 1).op))


def test_filter_with_an_empty_field_is_rejected():
    with pytest.raises(VectorStoreConfigError):
        validate_filter(eq("", 1))


def test_in_filter_requires_values():
    with pytest.raises(VectorStoreConfigError):
        validate_filter(is_in("a", []))


def test_in_filter_requires_a_sequence():
    with pytest.raises(VectorStoreConfigError):
        validate_filter(MetadataFilter(op=is_in("a", 1).op, field="a", value=1))


def test_exists_filter_requires_a_boolean():
    with pytest.raises(VectorStoreConfigError):
        validate_filter(MetadataFilter(op=exists("a").op, field="a", value="yes"))


def test_and_filter_requires_children():
    with pytest.raises(VectorStoreConfigError):
        validate_filter(and_())


def test_not_filter_requires_exactly_one_child():
    with pytest.raises(VectorStoreConfigError):
        validate_filter(MetadataFilter(op=not_(eq("a", 1)).op, children=()))


def test_logical_filter_must_not_carry_a_field():
    with pytest.raises(VectorStoreConfigError):
        validate_filter(
            MetadataFilter(op=and_(eq("a", 1)).op, field="a", children=(eq("a", 1),))
        )


def test_ordered_filter_rejects_booleans():
    with pytest.raises(VectorStoreConfigError):
        validate_filter(gt("a", True))


def test_filter_as_dict_is_serializable():
    data = and_(eq("a", 1), not_(exists("b"))).as_dict()
    assert data["op"] == "and"
    assert data["children"][0]["field"] == "a"
    assert data["children"][1]["op"] == "not"


# ----------------------------------------------------------------------
# search requests
# ----------------------------------------------------------------------
def test_valid_dense_request_is_accepted():
    request = SearchRequest(
        collection="docs", space="dense", dense=[0.1] * 8, limit=5
    )
    validate_search_request(request, dense_space())


def test_request_without_a_query_vector_is_rejected():
    request = SearchRequest(collection="docs", space="dense")
    with pytest.raises(VectorStoreConfigError):
        validate_search_request(request, dense_space())


def test_request_with_two_query_vectors_is_rejected():
    request = SearchRequest(
        collection="docs", space="dense", dense=[0.1] * 8, multi_vector=[[0.1] * 8]
    )
    with pytest.raises(VectorStoreConfigError):
        validate_search_request(request)


def test_request_with_a_non_positive_limit_is_rejected():
    request = SearchRequest(collection="docs", space="dense", dense=[0.1] * 8, limit=0)
    with pytest.raises(VectorStoreConfigError):
        validate_search_request(request)


def test_request_with_a_wrong_dimension_is_rejected():
    request = SearchRequest(collection="docs", space="dense", dense=[0.1] * 4)
    with pytest.raises(VectorStoreError):
        validate_search_request(request, dense_space())


def test_request_with_a_mismatching_representation_is_rejected():
    request = SearchRequest(collection="docs", space="dense", dense=[0.1] * 8)
    with pytest.raises(VectorStoreConfigError):
        validate_search_request(request, sparse_space())


def test_sparse_request_requires_indices_and_values():
    request = SearchRequest(collection="docs", space="sparse", sparse_indices=[1])
    with pytest.raises(VectorStoreConfigError):
        validate_search_request(request, sparse_space())


def test_request_with_a_broken_filter_is_rejected():
    request = SearchRequest(
        collection="docs",
        space="dense",
        dense=[0.1] * 8,
        filter=MetadataFilter(op=eq("a", 1).op),
    )
    with pytest.raises(VectorStoreConfigError):
        validate_search_request(request, dense_space())


# ----------------------------------------------------------------------
# results and capabilities
# ----------------------------------------------------------------------
def test_search_result_keeps_identity_score_and_metric():
    result = SearchResult(
        chunk_id="abc123-fixed-0000",
        doc_id="doc.html",
        score=0.87,
        space="dense",
        metric=DistanceMetric.COSINE,
        score_direction=ScoreDirection.HIGHER_IS_BETTER,
    )
    assert result.chunk_id == "abc123-fixed-0000"
    assert result.doc_id == "doc.html"
    assert result.score == 0.87
    assert result.metric is DistanceMetric.COSINE
    assert result.score_direction is ScoreDirection.HIGHER_IS_BETTER


def test_index_config_describes_its_target():
    index = IndexConfig(
        kind=IndexKind.PAYLOAD,
        target="tenant",
        field_type=PayloadFieldType.KEYWORD,
        build_params={"field_type": "keyword"},
    )
    assert index.index_name == "tenant"
    assert index.as_dict()["kind"] == "payload"


def test_hnsw_and_sparse_index_configs_are_self_describing():
    hnsw = HnswIndexConfig(m=32, ef_construct=200)
    assert hnsw.as_dict()["m"] == 32
    sparse = SparseIndexConfig(full_scan_threshold=1000)
    assert sparse.as_dict()["full_scan_threshold"] == 1000
