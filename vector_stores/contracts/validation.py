"""Validation helpers shared by the vector-store contracts.

The rules enforced here are the portable part of the contract:

- a collection declares at least one named vector space, and every space
  declares its dimensions and metric where they apply;
- a record's vectors must match the shape of the target space and belong to
  the *same* embedding space (same model, revision, dimensions and
  normalization) - matching dimensions alone are never enough;
- a search request carries exactly one query representation, shaped like the
  target space.

Backends call these helpers before touching their own client so that errors
are identical whichever backend is used.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from embeddings.contracts.models import EmbeddingSpaceDescriptor, RepresentationType
from vector_stores.contracts.exceptions import (
    DimensionMismatchError,
    EmbeddingSpaceMismatchError,
    InvalidRecordError,
    VectorStoreConfigError,
)
from vector_stores.contracts.filters import validate_filter
from vector_stores.contracts.models import (
    CollectionConfig,
    DistanceMetric,
    SearchRequest,
    SpaceVectors,
    StoredRecord,
    VectorSpaceConfig,
)

#: Metrics that can be used with a multi-vector (late interaction) space.
MULTI_VECTOR_METRICS = (
    DistanceMetric.COSINE,
    DistanceMetric.DOT,
    DistanceMetric.EUCLIDEAN,
)


def _non_empty_string(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VectorStoreConfigError(f"{what} must be a non-empty string")
    return value


def validate_vector_space(name: str, space: VectorSpaceConfig) -> None:
    """Validate a single named vector space."""
    _non_empty_string(space.name, "vector space name")
    if space.name != name:
        raise VectorStoreConfigError(
            f"vector space name mismatch: declared {space.name!r} under key {name!r}"
        )
    if not isinstance(space.vector_type, RepresentationType):
        raise VectorStoreConfigError(
            f"vector space {name!r} has an invalid vector type: {space.vector_type!r}"
        )
    if not isinstance(space.embedding_space, EmbeddingSpaceDescriptor):
        raise VectorStoreConfigError(
            f"vector space {name!r} must carry an EmbeddingSpaceDescriptor"
        )
    _non_empty_string(space.embedding_space.model_id, "embedding space model_id")

    if space.vector_type in (RepresentationType.DENSE, RepresentationType.MULTI_VECTOR):
        if not isinstance(space.dimensions, int) or isinstance(space.dimensions, bool):
            raise VectorStoreConfigError(
                f"vector space {name!r} must declare a positive integer dimension"
            )
        if space.dimensions <= 0:
            raise VectorStoreConfigError(
                f"vector space {name!r} must declare a positive dimension, got {space.dimensions}"
            )
    elif space.dimensions is not None:
        raise VectorStoreConfigError(
            f"sparse vector space {name!r} must not declare dimensions "
            f"(got {space.dimensions})"
        )

    if space.distance is not None and not isinstance(space.distance, DistanceMetric):
        raise VectorStoreConfigError(
            f"vector space {name!r} has an invalid distance metric: {space.distance!r}"
        )
    if (
        space.vector_type is RepresentationType.MULTI_VECTOR
        and space.distance is not None
        and space.distance not in MULTI_VECTOR_METRICS
    ):
        raise VectorStoreConfigError(
            f"multi-vector space {name!r} cannot use the {space.distance.value!r} metric"
        )
    if space.vector_type is RepresentationType.SPARSE and space.distance is not None:
        raise VectorStoreConfigError(
            f"sparse vector space {name!r} must not declare a distance metric"
        )


def validate_collection_config(config: CollectionConfig) -> None:
    """Validate a collection configuration.

    Raises:
        VectorStoreConfigError: if the configuration is malformed.
    """
    if not isinstance(config, CollectionConfig):
        raise VectorStoreConfigError(
            f"expected a CollectionConfig, got {type(config).__name__}"
        )
    _non_empty_string(config.name, "collection name")
    if not config.vector_spaces:
        raise VectorStoreConfigError(
            f"collection {config.name!r} must declare at least one vector space"
        )
    for name, space in config.vector_spaces.items():
        validate_vector_space(name, space)

    seen_fields: set[str] = set()
    for spec in config.payload_indexes:
        _non_empty_string(spec.field, "payload index field")
        if spec.field in seen_fields:
            raise VectorStoreConfigError(
                f"collection {config.name!r} declares the payload index {spec.field!r} twice"
            )
        seen_fields.add(spec.field)


def normalize_descriptor(
    descriptor: EmbeddingSpaceDescriptor, vector_type: RepresentationType
) -> EmbeddingSpaceDescriptor:
    """Return ``descriptor`` with the representation kind and width of ``vector_type``.

    Sparse spaces have no fixed width, and the representation kind of a space
    is authoritative - a model may produce several representations at once
    (Qwen3-Embedding-0.6B), so the kind of the *space* decides how the descriptor is read.
    """
    return replace(
        descriptor,
        representation_type=vector_type,
        dimensions=None if vector_type is RepresentationType.SPARSE else descriptor.dimensions,
    )


def embedding_space_mismatch_reasons(
    expected: EmbeddingSpaceDescriptor, actual: EmbeddingSpaceDescriptor
) -> list[str]:
    """Return the reasons why two embedding spaces are incompatible."""
    reasons: list[str] = []
    if expected.model_id != actual.model_id:
        reasons.append(f"model_id {actual.model_id!r} != {expected.model_id!r}")
    if expected.model_revision != actual.model_revision:
        reasons.append(
            f"model_revision {actual.model_revision!r} != {expected.model_revision!r}"
        )
    if expected.dimensions != actual.dimensions:
        reasons.append(f"dimensions {actual.dimensions!r} != {expected.dimensions!r}")
    if expected.normalization != actual.normalization:
        reasons.append(
            f"normalization {actual.normalization.value!r} != {expected.normalization.value!r}"
        )
    if (
        expected.supported_languages
        and actual.supported_languages
        and not set(expected.supported_languages).intersection(actual.supported_languages)
    ):
        reasons.append("supported languages do not overlap")
    if expected.representation_type != actual.representation_type:
        reasons.append(
            f"representation type {actual.representation_type.value!r} != "
            f"{expected.representation_type.value!r}"
        )
    return reasons


def assert_embedding_space_compatible(
    space: VectorSpaceConfig, descriptor: EmbeddingSpaceDescriptor
) -> None:
    """Check that ``descriptor`` belongs to the embedding space of ``space``.

    Raises:
        EmbeddingSpaceMismatchError: if the model identity, revision, width or
            normalization differ.
    """
    expected = normalize_descriptor(space.embedding_space, space.vector_type)
    actual = normalize_descriptor(descriptor, space.vector_type)
    reasons = embedding_space_mismatch_reasons(expected, actual)
    if reasons:
        raise EmbeddingSpaceMismatchError(
            f"vector space {space.name!r} expects the embedding space of "
            f"{expected.model_id!r} (revision {expected.model_revision!r}, "
            f"{expected.dimensions} dims, {expected.normalization.value} normalization) "
            f"but got a different one: {'; '.join(reasons)}"
        )
    if not expected.is_compatible_with(actual):  # pragma: no cover - defensive
        raise EmbeddingSpaceMismatchError(
            f"vector space {space.name!r} received an incompatible embedding space"
        )


def _check_finite(vector: Sequence[float], what: str) -> None:
    for value in vector:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise InvalidRecordError(f"{what} contains a non-numeric value: {value!r}")
        if not math.isfinite(float(value)):
            raise InvalidRecordError(f"{what} contains a non-finite value: {value!r}")


def validate_space_vectors(space: VectorSpaceConfig, vectors: SpaceVectors) -> None:
    """Validate the vectors of one record against one vector space.

    Raises:
        InvalidRecordError: if the payload shape does not match the space.
        DimensionMismatchError: if the width differs from the declared one.
        EmbeddingSpaceMismatchError: if the vectors come from another model.
    """
    if vectors.space != space.name:
        raise InvalidRecordError(
            f"vector payload declares space {vectors.space!r} but is stored under "
            f"{space.name!r}"
        )
    assert_embedding_space_compatible(space, vectors.embedding_space)

    if space.vector_type is RepresentationType.DENSE:
        if vectors.dense is None:
            raise InvalidRecordError(
                f"vector space {space.name!r} is dense but the record carries no dense vector"
            )
        if len(vectors.dense) != space.dimensions:
            raise DimensionMismatchError(
                f"dense vector for space {space.name!r} has {len(vectors.dense)} "
                f"dimensions, expected {space.dimensions}"
            )
        _check_finite(vectors.dense, f"dense vector for space {space.name!r}")
    elif space.vector_type is RepresentationType.SPARSE:
        if vectors.sparse_indices is None or vectors.sparse_values is None:
            raise InvalidRecordError(
                f"vector space {space.name!r} is sparse but the record carries no "
                "sparse indices/values"
            )
        if len(vectors.sparse_indices) != len(vectors.sparse_values):
            raise InvalidRecordError(
                f"sparse vector for space {space.name!r} has "
                f"{len(vectors.sparse_indices)} indices but "
                f"{len(vectors.sparse_values)} values"
            )
        if not vectors.sparse_indices:
            raise InvalidRecordError(
                f"sparse vector for space {space.name!r} is empty"
            )
        for index in vectors.sparse_indices:
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise InvalidRecordError(
                    f"sparse vector for space {space.name!r} has an invalid index: {index!r}"
                )
        _check_finite(vectors.sparse_values, f"sparse vector for space {space.name!r}")
    else:
        if not vectors.multi_vector:
            raise InvalidRecordError(
                f"vector space {space.name!r} is multi-vector but the record carries "
                "no multi-vector matrix"
            )
        for row in vectors.multi_vector:
            if len(row) != space.dimensions:
                raise DimensionMismatchError(
                    f"multi-vector row for space {space.name!r} has {len(row)} "
                    f"dimensions, expected {space.dimensions}"
                )
            _check_finite(row, f"multi-vector row for space {space.name!r}")


def validate_record(record: StoredRecord, config: CollectionConfig) -> None:
    """Validate a record against the collection it is stored in.

    Raises:
        InvalidRecordError: if the record is malformed or targets an unknown
            vector space.
        DimensionMismatchError: if a vector width differs from the space.
        EmbeddingSpaceMismatchError: if a vector belongs to another model.
    """
    if not isinstance(record, StoredRecord):
        raise InvalidRecordError(f"expected a StoredRecord, got {type(record).__name__}")
    chunk = record.chunk
    if not isinstance(chunk.id, str) or not chunk.id.strip():
        raise InvalidRecordError("chunk id must be a non-empty string")
    if not isinstance(chunk.doc_id, str) or not chunk.doc_id.strip():
        raise InvalidRecordError("chunk document id must be a non-empty string")
    if not record.vectors:
        raise InvalidRecordError(
            f"record for chunk {chunk.id!r} carries no vector"
        )
    for name, vectors in record.vectors.items():
        space = config.vector_spaces.get(name)
        if space is None:
            raise InvalidRecordError(
                f"record for chunk {chunk.id!r} targets the unknown vector space "
                f"{name!r}; known spaces: {sorted(config.vector_spaces)}"
            )
        validate_space_vectors(space, vectors)


def validate_search_request(
    request: SearchRequest, space: VectorSpaceConfig | None = None
) -> None:
    """Validate a search request.

    When ``space`` is given, the query representation must match the vector
    type and the width of that space.

    Raises:
        VectorStoreConfigError: if the request is malformed.
    """
    if not isinstance(request, SearchRequest):
        raise VectorStoreConfigError(
            f"expected a SearchRequest, got {type(request).__name__}"
        )
    _non_empty_string(request.collection, "search collection")
    _non_empty_string(request.space, "search vector space")
    if not isinstance(request.limit, int) or isinstance(request.limit, bool):
        raise VectorStoreConfigError("search limit must be an integer")
    if request.limit < 1:
        raise VectorStoreConfigError(
            f"search limit must be >= 1, got {request.limit}"
        )
    if request.offset is not None and request.offset < 0:
        raise VectorStoreConfigError("search offset must be >= 0")
    if request.score_threshold is not None and not isinstance(
        request.score_threshold, (int, float)
    ):
        raise VectorStoreConfigError("search score_threshold must be a number")
    if not all(isinstance(key, str) for key in request.params):
        raise VectorStoreConfigError("search params keys must be strings")

    kind = request.query_representation()

    if request.dense is not None:
        _check_finite(request.dense, "query dense vector")
    if request.sparse_indices is not None or request.sparse_values is not None:
        if request.sparse_indices is None or request.sparse_values is None:
            raise VectorStoreConfigError(
                "a sparse query needs both sparse_indices and sparse_values"
            )
        if len(request.sparse_indices) != len(request.sparse_values):
            raise VectorStoreConfigError(
                f"sparse query has {len(request.sparse_indices)} indices but "
                f"{len(request.sparse_values)} values"
            )
        if not request.sparse_indices:
            raise VectorStoreConfigError("sparse query is empty")
        _check_finite(request.sparse_values, "query sparse vector")
    if request.multi_vector is not None:
        if not request.multi_vector:
            raise VectorStoreConfigError("multi-vector query is empty")
        widths = {len(row) for row in request.multi_vector}
        if len(widths) != 1:
            raise VectorStoreConfigError(
                "multi-vector query rows have different widths"
            )
        for row in request.multi_vector:
            _check_finite(row, "query multi-vector row")

    if space is not None:
        if kind is not space.vector_type:
            raise VectorStoreConfigError(
                f"vector space {space.name!r} is {space.vector_type.value} but the "
                f"query carries a {kind.value} representation"
            )
        if (
            kind is RepresentationType.DENSE
            and request.dense is not None
            and len(request.dense) != space.dimensions
        ):
            raise DimensionMismatchError(
                f"query vector has {len(request.dense)} dimensions, expected "
                f"{space.dimensions} for space {space.name!r}"
            )
        if kind is RepresentationType.MULTI_VECTOR and request.multi_vector is not None:
            for row in request.multi_vector:
                if len(row) != space.dimensions:
                    raise DimensionMismatchError(
                        f"query multi-vector row has {len(row)} dimensions, expected "
                        f"{space.dimensions} for space {space.name!r}"
                    )

    if request.filter is not None:
        validate_filter(request.filter)
