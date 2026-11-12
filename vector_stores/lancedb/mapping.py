"""Translation between the portable contracts and the LanceDB data model.

A LanceDB collection is one **table**.  The table schema is derived from the
collection configuration:

===========================  ==========================================
column                       content
===========================  ==========================================
``chunk_id``                 the public chunk id (the LanceDB key)
``doc_id``                   the public document id
``v_<space>``                dense vectors (fixed size list of float32)
``s_<space>_indices``        sparse indices (storage only)
``s_<space>_values``         sparse weights (storage only)
``m_<space>``                multi-vector matrices (token level vectors)
``metadata``                 struct with the declared filterable fields
``metadata_json``            the full metadata, lossless
``payload``                  the contract fields (JSON)
===========================  ==========================================

Only the fields the user declares in ``payload_indexes`` become columns of
the ``metadata`` struct: LanceDB tables have a fixed schema, so a filter can
only be pushed down to a field that exists in it.  A filter on an
undeclared field is rejected explicitly instead of being ignored.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

from embeddings.contracts.models import RepresentationType
from vector_stores.contracts.exceptions import (
    InvalidRecordError,
    UnsupportedFilterOperationError,
    VectorStoreConfigError,
)
from vector_stores.contracts.filters import FilterOp, MetadataFilter
from vector_stores.contracts.indexes import PayloadFieldType
from vector_stores.contracts.models import (
    CollectionConfig,
    DistanceMetric,
    SpaceVectors,
    StoredRecord,
    VectorSpaceConfig,
)
from vector_stores.contracts.payload import (
    CONTRACT_PAYLOAD_KEY,
    decode_contract_payload,
    encode_contract_payload,
)
from vector_stores.lancedb._client import pyarrow

#: Name of the table that holds the collection configurations.
CONFIG_TABLE = "_prome_collections"

#: Name of the struct column holding the declared filterable metadata.
METADATA_COLUMN = "metadata"

#: Name of the column holding the full metadata as JSON.
METADATA_JSON_COLUMN = "metadata_json"

#: Name of the column holding the contract fields as JSON.
PAYLOAD_COLUMN = "payload"

#: Column holding the public chunk id.
CHUNK_ID_COLUMN = "chunk_id"

#: Column holding the public document id.
DOC_ID_COLUMN = "doc_id"

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def dense_column(space: str) -> str:
    """Return the column name of a dense vector space."""
    return f"v_{space}"


def sparse_indices_column(space: str) -> str:
    """Return the column name of the sparse indices of a space."""
    return f"s_{space}_indices"


def sparse_values_column(space: str) -> str:
    """Return the column name of the sparse weights of a space."""
    return f"s_{space}_values"


def multi_vector_column(space: str) -> str:
    """Return the column name of a multi-vector space."""
    return f"m_{space}"


def vector_column(space: VectorSpaceConfig) -> str:
    """Return the column that holds the searchable vector of a space."""
    if space.vector_type is RepresentationType.DENSE:
        return dense_column(space.name)
    if space.vector_type is RepresentationType.MULTI_VECTOR:
        return multi_vector_column(space.name)
    raise VectorStoreConfigError(
        f"vector space {space.name!r} is {space.vector_type.value}: LanceDB has no "
        "searchable vector column for it"
    )


def arrow_type(field_type: PayloadFieldType) -> Any:
    """Return the Arrow type of a portable payload field type."""
    pa = pyarrow()
    return {
        PayloadFieldType.KEYWORD: pa.string(),
        PayloadFieldType.INTEGER: pa.int64(),
        PayloadFieldType.FLOAT: pa.float64(),
        PayloadFieldType.BOOLEAN: pa.bool_(),
        PayloadFieldType.DATETIME: pa.timestamp("us"),
    }[field_type]


def validate_field_name(field: str) -> str:
    """Validate that ``field`` is a plain identifier.

    LanceDB builds SQL expressions from the field names; only simple
    identifiers are accepted so that a field name can never inject SQL.
    """
    if not isinstance(field, str) or not _IDENTIFIER.match(field):
        raise VectorStoreConfigError(
            f"LanceDB metadata field names must be simple identifiers, got {field!r}"
        )
    return field


def build_schema(config: CollectionConfig) -> Any:
    """Build the Arrow schema of the table of ``config``.

    The ``metadata`` struct column only exists when the collection declares
    at least one filterable field: LanceDB mis-handles an *empty* struct
    column when a row is cast to the schema.
    """
    pa = pyarrow()
    fields = [
        pa.field(CHUNK_ID_COLUMN, pa.string()),
        pa.field(DOC_ID_COLUMN, pa.string()),
    ]
    for space in config.vector_spaces.values():
        if space.vector_type is RepresentationType.DENSE:
            fields.append(
                pa.field(
                    dense_column(space.name),
                    pa.list_(pa.float32(), space.dimensions),
                )
            )
        elif space.vector_type is RepresentationType.SPARSE:
            fields.append(
                pa.field(sparse_indices_column(space.name), pa.list_(pa.int32()))
            )
            fields.append(
                pa.field(sparse_values_column(space.name), pa.list_(pa.float32()))
            )
        else:
            fields.append(
                pa.field(
                    multi_vector_column(space.name),
                    pa.list_(pa.list_(pa.float32(), space.dimensions)),
                )
            )
    struct_fields = [
        pa.field(validate_field_name(spec.field), arrow_type(spec.field_type))
        for spec in config.payload_indexes
    ]
    if struct_fields:
        fields.append(pa.field(METADATA_COLUMN, pa.struct(struct_fields)))
    fields.append(pa.field(METADATA_JSON_COLUMN, pa.string()))
    fields.append(pa.field(PAYLOAD_COLUMN, pa.string()))
    return pa.schema(fields)


def encode_record(record: StoredRecord, config: CollectionConfig) -> dict[str, Any]:
    """Build the LanceDB row of ``record``.

    The keys are written in the order of the table schema: LanceDB casts a
    row to the schema of the table and an empty struct column breaks that
    cast.
    """
    chunk = record.chunk
    row: dict[str, Any] = {
        CHUNK_ID_COLUMN: chunk.id,
        DOC_ID_COLUMN: chunk.doc_id,
    }
    for name, vectors in record.vectors.items():
        space = config.vector_spaces[name]
        if space.vector_type is RepresentationType.DENSE:
            row[dense_column(name)] = [float(value) for value in (vectors.dense or [])]
        elif space.vector_type is RepresentationType.SPARSE:
            row[sparse_indices_column(name)] = [
                int(index) for index in (vectors.sparse_indices or [])
            ]
            row[sparse_values_column(name)] = [
                float(value) for value in (vectors.sparse_values or [])
            ]
        else:
            row[multi_vector_column(name)] = [
                [float(value) for value in vector] for vector in (vectors.multi_vector or [])
            ]
    if config.payload_indexes:
        row[METADATA_COLUMN] = {
            spec.field: record.metadata.get(spec.field) for spec in config.payload_indexes
        }
    row[METADATA_JSON_COLUMN] = json.dumps(record.metadata, ensure_ascii=False, default=str)
    row[PAYLOAD_COLUMN] = json.dumps(
        {
            CHUNK_ID_COLUMN: chunk.id,
            DOC_ID_COLUMN: chunk.doc_id,
            CONTRACT_PAYLOAD_KEY: encode_contract_payload(record),
        },
        ensure_ascii=False,
        default=str,
    )
    return row


def decode_row(row: Mapping[str, Any], config: CollectionConfig) -> StoredRecord:
    """Rebuild a :class:`StoredRecord` from a LanceDB row."""
    payload = row.get(PAYLOAD_COLUMN)
    record = decode_contract_payload(
        json.loads(payload) if isinstance(payload, str) and payload else {}
    )
    vectors: dict[str, SpaceVectors] = {}
    for name, space in config.vector_spaces.items():
        vectors[name] = SpaceVectors(
            space=name,
            embedding_space=space.embedding_space,
        )
    for name, space in config.vector_spaces.items():
        if space.vector_type is RepresentationType.DENSE:
            raw = row.get(dense_column(name))
            vectors[name].dense = [float(value) for value in (raw or [])]
        elif space.vector_type is RepresentationType.SPARSE:
            indices = row.get(sparse_indices_column(name))
            values = row.get(sparse_values_column(name))
            vectors[name].sparse_indices = [int(index) for index in (indices or [])]
            vectors[name].sparse_values = [float(value) for value in (values or [])]
        else:
            raw = row.get(multi_vector_column(name))
            vectors[name].multi_vector = [
                [float(value) for value in vector] for vector in (raw or [])
            ]
    record.vectors = vectors
    raw_metadata = row.get(METADATA_JSON_COLUMN)
    if isinstance(raw_metadata, str) and raw_metadata:
        try:
            decoded = json.loads(raw_metadata)
        except ValueError:  # pragma: no cover - defensive
            decoded = {}
        if isinstance(decoded, dict):
            record.metadata = decoded
    return record


# ----------------------------------------------------------------------
# filters
# ----------------------------------------------------------------------
def _literal(value: Any) -> str:
    """Return the SQL literal of ``value``."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value)
    if "\x00" in text:
        raise VectorStoreConfigError("filter values must not contain null bytes")
    escaped = text.replace("'", "''")
    return f"'{escaped}'"


def _column(field: str, config: CollectionConfig) -> str:
    """Return the SQL column of a portable filter field."""
    if field in (CHUNK_ID_COLUMN, DOC_ID_COLUMN):
        return field
    declared = {spec.field for spec in config.payload_indexes}
    if field not in declared:
        raise UnsupportedFilterOperationError(
            f"LanceDB can only filter on the fields declared in the collection "
            f"configuration; {field!r} is not declared (declared: {sorted(declared)})"
        )
    return f"{METADATA_COLUMN}.{field}"


def to_sql_filter(node: MetadataFilter, config: CollectionConfig) -> str:
    """Translate a portable filter into a LanceDB SQL expression.

    Raises:
        UnsupportedFilterOperationError: if the filter uses a field that is
            not declared in the collection configuration.
    """
    op = node.op
    if op is FilterOp.AND:
        return "(" + " AND ".join(
            to_sql_filter(child, config) for child in node.children
        ) + ")"
    if op is FilterOp.OR:
        return "(" + " OR ".join(
            to_sql_filter(child, config) for child in node.children
        ) + ")"
    if op is FilterOp.NOT:
        return "NOT (" + to_sql_filter(node.children[0], config) + ")"
    column = _column(node.field, config)
    if op is FilterOp.EQ:
        if node.value is None:
            return f"{column} IS NULL"
        return f"{column} = {_literal(node.value)}"
    if op is FilterOp.NE:
        return f"{column} <> {_literal(node.value)}"
    if op is FilterOp.IN:
        values = list(node.value)
        return f"{column} IN ({', '.join(_literal(value) for value in values)})"
    if op is FilterOp.NOT_IN:
        values = list(node.value)
        return f"{column} NOT IN ({', '.join(_literal(value) for value in values)})"
    if op is FilterOp.GT:
        return f"{column} > {_literal(node.value)}"
    if op is FilterOp.GTE:
        return f"{column} >= {_literal(node.value)}"
    if op is FilterOp.LT:
        return f"{column} < {_literal(node.value)}"
    if op is FilterOp.LTE:
        return f"{column} <= {_literal(node.value)}"
    if op is FilterOp.EXISTS:
        return f"{column} IS NOT NULL" if node.value else f"{column} IS NULL"
    raise UnsupportedFilterOperationError(
        f"LanceDB does not support the filter operation {op.value!r}"
    )


def lance_distance_type(metric: DistanceMetric | None) -> str:
    """Return the LanceDB distance type of a portable metric.

    Raises:
        VectorStoreConfigError: if LanceDB has no equivalent for the metric.
    """
    if metric is None:
        return "cosine"
    try:
        return {
            DistanceMetric.COSINE: "cosine",
            DistanceMetric.DOT: "dot",
            DistanceMetric.EUCLIDEAN: "l2",
        }[metric]
    except KeyError as exc:
        raise VectorStoreConfigError(
            f"LanceDB does not support the {metric.value!r} metric "
            "(supported: cosine, dot, l2)"
        ) from exc


def canonical_score(
    distance: float,
    metric: DistanceMetric | None,
    *,
    query_vectors: int = 1,
) -> float:
    """Convert a LanceDB distance into the canonical score of the metric.

    LanceDB reports distances: ``1 - similarity`` for cosine and dot, and the
    *squared* euclidean distance for ``l2``.  The portable contract reports
    the score in the convention of the metric (a similarity for cosine/dot, a
    distance for euclidean), so the value is converted here.
    """
    if metric is None:
        metric = DistanceMetric.COSINE
    if metric in (DistanceMetric.COSINE, DistanceMetric.DOT):
        return query_vectors - distance
    if metric is DistanceMetric.EUCLIDEAN:
        return distance**0.5 if query_vectors == 1 else distance
    return distance


def score_direction(metric: DistanceMetric | None) -> Any:
    """Return the score direction of a metric."""
    from vector_stores.contracts.models import ScoreDirection

    if metric is None:
        return ScoreDirection.HIGHER_IS_BETTER
    return metric.score_direction


def chunk_id_list(chunk_ids: Iterable[str]) -> str:
    """Return the SQL expression matching a list of chunk ids."""
    values = ", ".join(_literal(chunk_id) for chunk_id in chunk_ids)
    return f"{CHUNK_ID_COLUMN} IN ({values})"


def check_metadata_keys(record: StoredRecord, config: CollectionConfig) -> None:
    """Reject metadata keys that would collide with a reserved column."""
    reserved = {CHUNK_ID_COLUMN, DOC_ID_COLUMN, METADATA_COLUMN, PAYLOAD_COLUMN}
    for key in record.metadata:
        if key in reserved:
            raise InvalidRecordError(
                f"metadata key {key!r} is reserved by the LanceDB adapter"
            )
