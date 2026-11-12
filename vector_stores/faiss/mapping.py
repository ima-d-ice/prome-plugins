"""Translation between the portable contracts and the FAISS / SQLite model.

FAISS is a similarity search library, not a document database: it stores
vectors and answers nearest neighbour queries.  Everything else - the
records, the metadata, the provenance, the collection configuration - lives
in SQLite, which is the **authoritative** store of a collection.

That split has one consequence: the FAISS index is a *derived* artifact.  It
can always be rebuilt from the vectors stored in SQLite, which is what the
adapter does when the two disagree (a missing, stale or corrupt index).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from embeddings.contracts.models import RepresentationType
from vector_stores.contracts.exceptions import (
    UnsupportedFilterOperationError,
    VectorStoreConfigError,
)
from vector_stores.contracts.filters import FilterOp, MetadataFilter
from vector_stores.contracts.models import DistanceMetric, VectorSpaceConfig

#: Name of the SQLite database of a collection.
DATABASE_NAME = "records.sqlite3"

#: Name of the FAISS index file of a vector space.
INDEX_PREFIX = "index-"
INDEX_SUFFIX = ".faiss"

#: Columns of the ``records`` table.
CHUNK_ID_COLUMN = "chunk_id"
DOC_ID_COLUMN = "doc_id"
FAISS_ID_COLUMN = "faiss_id"
PAYLOAD_COLUMN = "payload"
METADATA_COLUMN = "metadata"
VECTOR_COLUMN = "vector"

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_field_name(field: str) -> str:
    """Validate that ``field`` is a plain identifier.

    Field names end up in SQL expressions; only simple identifiers are
    accepted so that a field name can never inject SQL.
    """
    if not isinstance(field, str) or not _IDENTIFIER.match(field):
        raise VectorStoreConfigError(
            f"FAISS metadata field names must be simple identifiers, got {field!r}"
        )
    return field


def metadata_expression(field: str) -> str:
    """Return the SQL expression reading a metadata field."""
    if field == CHUNK_ID_COLUMN:
        return CHUNK_ID_COLUMN
    if field == DOC_ID_COLUMN:
        return DOC_ID_COLUMN
    validate_field_name(field)
    return f"json_extract({METADATA_COLUMN}, '$.{field}')"


def to_sql_filter(node: MetadataFilter) -> tuple[str, list[Any]]:
    """Compile a portable filter into parameterized SQL.

    Values are always bound as parameters; only validated identifiers and
    the fixed ``json_extract`` calls end up in the SQL text.

    Returns:
        The SQL expression and the list of parameters.
    """
    op = node.op
    if op is FilterOp.AND:
        parts, params = [], []
        for child in node.children:
            sql, child_params = to_sql_filter(child)
            parts.append(sql)
            params.extend(child_params)
        return "(" + " AND ".join(parts) + ")", params
    if op is FilterOp.OR:
        parts, params = [], []
        for child in node.children:
            sql, child_params = to_sql_filter(child)
            parts.append(sql)
            params.extend(child_params)
        return "(" + " OR ".join(parts) + ")", params
    if op is FilterOp.NOT:
        sql, params = to_sql_filter(node.children[0])
        return f"NOT ({sql})", params

    column = metadata_expression(node.field)
    if op is FilterOp.EQ:
        if node.value is None:
            return f"{column} IS NULL", []
        return f"{column} = ?", [node.value]
    if op is FilterOp.NE:
        return f"{column} <> ?", [node.value]
    if op is FilterOp.IN:
        values = list(node.value)
        placeholders = ", ".join("?" for _ in values)
        return f"{column} IN ({placeholders})", values
    if op is FilterOp.NOT_IN:
        values = list(node.value)
        placeholders = ", ".join("?" for _ in values)
        return f"{column} NOT IN ({placeholders})", values
    if op is FilterOp.GT:
        return f"{column} > ?", [node.value]
    if op is FilterOp.GTE:
        return f"{column} >= ?", [node.value]
    if op is FilterOp.LT:
        return f"{column} < ?", [node.value]
    if op is FilterOp.LTE:
        return f"{column} <= ?", [node.value]
    if op is FilterOp.EXISTS:
        return (
            f"{column} IS NOT NULL" if node.value else f"{column} IS NULL"
        ), []
    raise UnsupportedFilterOperationError(
        f"the FAISS adapter does not support the filter operation {op.value!r}"
    )


def payload_index_name(field: str) -> str:
    """Return the name of the SQLite index of a metadata field."""
    return f"idx_meta_{validate_field_name(field)}"


def is_cosine(space: VectorSpaceConfig) -> bool:
    """Return True when the space uses the cosine metric."""
    return space.distance in (None, DistanceMetric.COSINE)


def metric_is_distance(space: VectorSpaceConfig) -> bool:
    """Return True when the metric reports a distance (lower is better)."""
    return space.distance is DistanceMetric.EUCLIDEAN


def normalize(vector: list[float]) -> list[float]:
    """Return the L2-normalized vector (used for the cosine metric)."""
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0.0:
        return list(vector)
    return [value / norm for value in vector]


def encode_vector(vector: list[float], space: VectorSpaceConfig) -> bytes:
    """Encode a dense vector as float32 bytes (the SQLite representation)."""
    import numpy as np

    values = np.asarray(vector, dtype="float32")
    if is_cosine(space):
        norm = float(np.linalg.norm(values))
        if norm > 0:
            values = values / norm
    return values.tobytes()


def decode_vector(blob: bytes, space: VectorSpaceConfig) -> list[float]:
    """Decode a dense vector from its float32 bytes."""
    import numpy as np

    values = np.frombuffer(blob, dtype="float32")
    if metric_is_distance(space):
        return [float(value) for value in values]
    return [float(value) for value in values]


def contract_payload(
    chunk_id: str, doc_id: str, record_payload: Mapping[str, Any]
) -> str:
    """Serialize the contract fields of a record."""
    return json.dumps(
        {
            CHUNK_ID_COLUMN: chunk_id,
            DOC_ID_COLUMN: doc_id,
            "_prome": dict(record_payload),
        },
        ensure_ascii=False,
        default=str,
    )


def metadata_json(metadata: Mapping[str, Any]) -> str:
    """Serialize the metadata of a record."""
    return json.dumps(dict(metadata), ensure_ascii=False, default=str)


def supports_dense_only(space: VectorSpaceConfig) -> bool:
    """Return True when the space holds plain dense vectors."""
    return space.vector_type is RepresentationType.DENSE
