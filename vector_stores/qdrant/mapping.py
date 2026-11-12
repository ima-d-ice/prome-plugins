"""Translation between the portable contracts and the Qdrant data model.

Three things are translated here:

1. **Point ids** - a Qdrant point id is an unsigned integer or a UUID, while
   a Prome chunk id is a free-form string (``{digest}-{strategy}-{index}``).
   The chunk id is mapped to a deterministic UUIDv5, so the same chunk always
   lands on the same point, and the original id is preserved in the payload.
2. **Payloads** - user metadata stays at the top level of the payload (so it
   can be filtered and indexed directly), while the contract fields live
   under the reserved ``_prome`` key.
3. **Filters** - the portable filter tree is mapped onto Qdrant conditions.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping
from typing import Any

from embeddings.contracts.models import RepresentationType
from vector_stores.contracts.exceptions import (
    InvalidRecordError,
    UnsupportedFilterOperationError,
)
from vector_stores.contracts.filters import FilterOp, MetadataFilter
from vector_stores.contracts.indexes import PayloadFieldType, PayloadIndexSpec
from vector_stores.contracts.models import (
    DistanceMetric,
    SpaceVectors,
    StoredRecord,
    VectorSpaceConfig,
    descriptor_from_dict,
)
from vector_stores.contracts.payload import (
    CONTRACT_PAYLOAD_KEY,
    RESERVED_PAYLOAD_KEYS,
    decode_contract_payload,
    encode_contract_payload,
)
from vector_stores.qdrant._client import models

#: Namespace used to derive deterministic point ids from chunk ids.
POINT_ID_NAMESPACE = uuid.UUID("6f1c2b3a-9d4e-4f2a-8b7c-1e2d3c4b5a60")

__all__ = [
    "CONTRACT_PAYLOAD_KEY",
    "RESERVED_PAYLOAD_KEYS",
    "decode_payload",
    "encode_payload",
    "from_qdrant_payload_type",
    "point_id",
    "to_qdrant_filter",
    "to_qdrant_payload_type",
    "to_qdrant_search_params",
    "to_qdrant_vectors_config",
]


def point_id(chunk_id: str) -> str:
    """Return the deterministic Qdrant point id of ``chunk_id``.

    The mapping is a UUIDv5 (SHA-1 based) of the chunk id in a fixed
    namespace: deterministic across processes and machines, and collision
    resistant for the number of chunks a store realistically holds.
    """
    if not isinstance(chunk_id, str) or not chunk_id:
        raise InvalidRecordError("chunk id must be a non-empty string")
    return str(uuid.uuid5(POINT_ID_NAMESPACE, chunk_id))


def encode_payload(record: StoredRecord) -> dict[str, Any]:
    """Build the Qdrant payload of ``record``."""
    payload: dict[str, Any] = {}
    for key, value in record.metadata.items():
        if key in RESERVED_PAYLOAD_KEYS:
            raise InvalidRecordError(
                f"metadata key {key!r} is reserved by the vector store contract"
            )
        payload[key] = value

    chunk = record.chunk
    payload["chunk_id"] = chunk.id
    payload["doc_id"] = chunk.doc_id
    payload[CONTRACT_PAYLOAD_KEY] = encode_contract_payload(record)
    return payload


def _sparse_parts(value: Any) -> tuple[Any, Any]:
    """Return the ``(indices, values)`` of a sparse vector.

    Qdrant returns a ``SparseVector`` model (or a plain mapping, depending on
    the transport); both are accepted.
    """
    if isinstance(value, Mapping):
        return value.get("indices"), value.get("values")
    indices = getattr(value, "indices", None)
    values = getattr(value, "values", None)
    if indices is None and values is None:
        return None, None
    return indices, values


def _decode_vectors(
    payload: Mapping[str, Any], raw_vectors: Any
) -> dict[str, SpaceVectors]:
    contract = payload.get(CONTRACT_PAYLOAD_KEY) or {}
    descriptors = contract.get("embedding_spaces") or {}
    vectors: dict[str, SpaceVectors] = {}
    raw_vectors = raw_vectors or {}
    if not isinstance(raw_vectors, Mapping):
        raw_vectors = {}
    for name, raw_descriptor in descriptors.items():
        descriptor = descriptor_from_dict(raw_descriptor)
        value = raw_vectors.get(name)
        dense: list[float] | None = None
        sparse_indices: list[int] | None = None
        sparse_values: list[float] | None = None
        multi_vector: list[list[float]] | None = None
        if value is not None:
            if descriptor.representation_type is RepresentationType.SPARSE:
                indices, values = _sparse_parts(value)
                sparse_indices = [int(i) for i in (indices or [])]
                sparse_values = [float(v) for v in (values or [])]
            elif descriptor.representation_type is RepresentationType.MULTI_VECTOR:
                multi_vector = [[float(v) for v in row] for row in value]
            else:
                dense = [float(v) for v in value]
        vectors[name] = SpaceVectors(
            space=name,
            embedding_space=descriptor,
            dense=dense,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            multi_vector=multi_vector,
        )
    return vectors


def decode_payload(payload: Mapping[str, Any], raw_vectors: Any = None) -> StoredRecord:
    """Rebuild a :class:`StoredRecord` from a Qdrant payload.

    A payload fetched without the contract fields (``_prome``) still yields
    the chunk id and the document id, so a result never loses its identity.
    """
    record = decode_contract_payload(payload)
    record.vectors = _decode_vectors(payload, raw_vectors)
    return record


def _distance(metric: DistanceMetric | None):
    models_ = models()
    if metric is None:
        return models_.Distance.COSINE
    return {
        DistanceMetric.COSINE: models_.Distance.COSINE,
        DistanceMetric.DOT: models_.Distance.DOT,
        DistanceMetric.EUCLIDEAN: models_.Distance.EUCLID,
        DistanceMetric.MANHATTAN: models_.Distance.MANHATTAN,
    }[metric]


def to_qdrant_vectors_config(
    spaces: Mapping[str, VectorSpaceConfig],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(vectors_config, sparse_vectors_config)`` for the spaces."""
    models_ = models()
    vectors_config: dict[str, Any] = {}
    sparse_config: dict[str, Any] = {}
    for name, space in spaces.items():
        if space.vector_type is RepresentationType.SPARSE:
            index = None
            if space.sparse is not None:
                index = models_.SparseIndexParams(
                    full_scan_threshold=space.sparse.full_scan_threshold,
                    **dict(space.sparse.backend_options),
                )
            sparse_config[name] = models_.SparseVectorParams(index=index)
            continue
        params: dict[str, Any] = {
            "size": space.dimensions,
            "distance": _distance(space.distance),
        }
        if space.vector_type is RepresentationType.MULTI_VECTOR:
            params["multivector_config"] = models_.MultiVectorConfig(
                comparator=models_.MultiVectorComparator.MAX_SIM
            )
        elif space.hnsw is not None:
            params["hnsw_config"] = models_.HnswConfigDiff(
                m=space.hnsw.m,
                ef_construct=space.hnsw.ef_construct,
                full_scan_threshold=space.hnsw.full_scan_threshold,
                max_indexing_threads=space.hnsw.max_indexing_threads,
                **dict(space.hnsw.backend_options),
            )
        vectors_config[name] = models_.VectorParams(**params)
    return vectors_config, sparse_config


def to_qdrant_payload_type(field_type: PayloadFieldType):
    """Return the Qdrant payload schema type of a portable field type."""
    models_ = models()
    return {
        PayloadFieldType.KEYWORD: models_.PayloadSchemaType.KEYWORD,
        PayloadFieldType.INTEGER: models_.PayloadSchemaType.INTEGER,
        PayloadFieldType.FLOAT: models_.PayloadSchemaType.FLOAT,
        PayloadFieldType.BOOLEAN: models_.PayloadSchemaType.BOOL,
        PayloadFieldType.DATETIME: models_.PayloadSchemaType.DATETIME,
    }[field_type]


def from_qdrant_payload_type(schema: Any) -> PayloadFieldType:
    """Return the portable field type of a Qdrant payload schema."""
    value = getattr(schema, "data_type", schema)
    raw = getattr(value, "value", value)
    mapping = {
        "keyword": PayloadFieldType.KEYWORD,
        "integer": PayloadFieldType.INTEGER,
        "float": PayloadFieldType.FLOAT,
        "bool": PayloadFieldType.BOOLEAN,
        "boolean": PayloadFieldType.BOOLEAN,
        "datetime": PayloadFieldType.DATETIME,
    }
    return mapping.get(str(raw), PayloadFieldType.KEYWORD)


def _value_condition(key: str, value: Any) -> Any:
    """Return a Qdrant condition matching ``value`` exactly.

    Qdrant's ``MatchValue`` only accepts strings, integers and booleans;
    floats are matched with an exact range instead.
    """
    models_ = models()
    if isinstance(value, float):
        return models_.FieldCondition(key=key, range=models_.Range(gte=value, lte=value))
    return models_.FieldCondition(key=key, match=models_.MatchValue(value=value))


def _membership_condition(key: str, values: Iterable[Any]) -> Any:
    """Return a Qdrant condition matching any of ``values``."""
    models_ = models()
    values = list(values)
    simple: list[Any] = []
    others: list[Any] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            others.append(value)
        else:
            simple.append(value)
    if not others:
        return models_.FieldCondition(key=key, match=models_.MatchAny(any=simple))
    conditions = []
    if simple:
        conditions.append(models_.FieldCondition(key=key, match=models_.MatchAny(any=simple)))
    for value in others:
        conditions.append(_value_condition(key, value))
    return models_.Filter(should=conditions)


def to_qdrant_filter(node: MetadataFilter) -> Any:
    """Translate a portable filter into a Qdrant filter.

    The result is always a ``Filter`` object (Qdrant expects a filter, not a
    bare condition).

    Raises:
        UnsupportedFilterOperationError: if Qdrant cannot express the node.
    """
    models_ = models()
    op = node.op
    if op is FilterOp.AND:
        return models_.Filter(must=[to_qdrant_filter(child) for child in node.children])
    if op is FilterOp.OR:
        return models_.Filter(should=[to_qdrant_filter(child) for child in node.children])
    if op is FilterOp.NOT:
        return models_.Filter(must_not=[to_qdrant_filter(node.children[0])])
    if op is FilterOp.EQ:
        return models_.Filter(must=[_value_condition(node.field, node.value)])
    if op is FilterOp.NE:
        return models_.Filter(must_not=[_value_condition(node.field, node.value)])
    if op is FilterOp.IN:
        return models_.Filter(must=[_membership_condition(node.field, node.value)])
    if op is FilterOp.NOT_IN:
        return models_.Filter(must_not=[_membership_condition(node.field, node.value)])
    if op is FilterOp.GT:
        return models_.Filter(
            must=[models_.FieldCondition(key=node.field, range=models_.Range(gt=node.value))]
        )
    if op is FilterOp.GTE:
        return models_.Filter(
            must=[models_.FieldCondition(key=node.field, range=models_.Range(gte=node.value))]
        )
    if op is FilterOp.LT:
        return models_.Filter(
            must=[models_.FieldCondition(key=node.field, range=models_.Range(lt=node.value))]
        )
    if op is FilterOp.LTE:
        return models_.Filter(
            must=[models_.FieldCondition(key=node.field, range=models_.Range(lte=node.value))]
        )
    if op is FilterOp.EXISTS:
        return models_.Filter(
            must=[
                models_.FieldCondition(key=node.field, is_empty=not bool(node.value))
            ]
        )
    raise UnsupportedFilterOperationError(
        f"Qdrant does not support the filter operation {op.value!r}"
    )


def to_qdrant_search_params(params: Mapping[str, Any], space: VectorSpaceConfig) -> Any:
    """Build the Qdrant search parameters of a request.

    Portable keys understood here:

    - ``exact``: run a brute-force search (exact results);
    - ``hnsw_ef``: beam width of the HNSW search.
    """
    models_ = models()
    exact = bool(params.get("exact", False))
    hnsw_ef = params.get("hnsw_ef")
    if hnsw_ef is None and space.hnsw is not None:
        hnsw_ef = space.hnsw.query_ef
    if not exact and hnsw_ef is None:
        return None
    return models_.SearchParams(exact=exact, hnsw_ef=hnsw_ef)


def payload_index_spec(field: str, field_type: PayloadFieldType) -> PayloadIndexSpec:
    """Return the payload index specification of a field."""
    return PayloadIndexSpec(field=field, field_type=field_type)
