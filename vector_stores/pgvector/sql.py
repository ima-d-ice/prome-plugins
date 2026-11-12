"""Translation between the portable contracts and the PostgreSQL data model.

A collection is a table.  Every identifier is quoted with
``psycopg.sql.Identifier`` and every value is bound as a parameter, so a
field name or a filter value can never change the meaning of a statement.

Vector columns:

- a dense space becomes a ``vector(dim)`` column;
- a sparse space becomes a ``sparsevec`` column;
- a multi-vector space becomes a ``jsonb`` column (storage only: PostgreSQL
  has no late interaction operator).

The metadata is stored in a ``jsonb`` column; the declared payload fields get
an expression index on ``metadata->>'field'``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from vector_stores.contracts.exceptions import (
    InvalidRecordError,
    UnsupportedFilterOperationError,
    VectorStoreConfigError,
)
from vector_stores.contracts.filters import FilterOp, MetadataFilter
from vector_stores.contracts.models import DistanceMetric

#: Column holding the public chunk id.
CHUNK_ID_COLUMN = "chunk_id"

#: Column holding the public document id.
DOC_ID_COLUMN = "doc_id"

#: Column holding the contract fields.
PAYLOAD_COLUMN = "payload"

#: Column holding the user metadata.
METADATA_COLUMN = "metadata"

#: Column holding the embedding-space descriptor.
EMBEDDING_SPACE_COLUMN = "embedding_space"

#: Name of the table holding the collection configurations.
CONFIG_TABLE = "prome_collection_config"

#: Name of the table holding the dimension count of the sparse vector spaces.
SPARSE_DIMENSIONS_TABLE = "prome_sparse_dimensions"

#: Maximum number of dimensions of a pgvector ``vector`` column.
MAX_VECTOR_DIMENSIONS = 2000

#: Maximum number of dimensions of a pgvector ``sparsevec`` value.
MAX_SPARSE_DIMENSIONS = 1_000_000

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_identifier(name: str, what: str = "identifier") -> str:
    """Validate that ``name`` is a plain SQL identifier."""
    if not isinstance(name, str) or not _IDENTIFIER.match(name):
        raise VectorStoreConfigError(
            f"{what} must be a simple SQL identifier, got {name!r}"
        )
    return name


def dense_column(space: str) -> str:
    """Return the column name of a dense vector space."""
    return f"{validate_identifier(space, 'vector space name')}_vector"


def sparse_column(space: str) -> str:
    """Return the column name of a sparse vector space."""
    return f"{validate_identifier(space, 'vector space name')}_sparse"


def multi_vector_column(space: str) -> str:
    """Return the column name of a multi-vector space."""
    return f"{validate_identifier(space, 'vector space name')}_multivector"


def vector_column(space: str, vector_type: Any) -> str:
    """Return the column that holds the vector of a space."""
    from embeddings.contracts.models import RepresentationType

    if vector_type is RepresentationType.DENSE:
        return dense_column(space)
    if vector_type is RepresentationType.SPARSE:
        return sparse_column(space)
    return multi_vector_column(space)


def metadata_expression(field: str) -> str:
    """Return the SQL expression reading a metadata field."""
    if field == CHUNK_ID_COLUMN:
        return CHUNK_ID_COLUMN
    if field == DOC_ID_COLUMN:
        return DOC_ID_COLUMN
    validate_identifier(field, "metadata field name")
    return f"{METADATA_COLUMN}->'{field}'"


def to_sql_filter(node: MetadataFilter) -> tuple[str, list[Any]]:
    """Compile a portable filter into parameterized SQL.

    The metadata is read as ``jsonb`` so that the comparison works for every
    JSON type without casting the text form.

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
    # The chunk id and the document id are real columns; the other fields live
    # in the JSONB metadata column and are compared as JSONB.
    is_column = node.field in (CHUNK_ID_COLUMN, DOC_ID_COLUMN)
    suffix = "" if is_column else "::jsonb"

    def bind(value: Any) -> Any:
        return value if is_column else json.dumps(value, default=str)

    if op is FilterOp.EQ:
        if node.value is None:
            return f"{column} IS NULL", []
        return f"{column} = %s{suffix}", [bind(node.value)]
    if op is FilterOp.NE:
        return f"{column} <> %s{suffix}", [bind(node.value)]
    if op is FilterOp.IN:
        values = list(node.value)
        placeholders = ", ".join(f"%s{suffix}" for _ in values)
        return f"{column} IN ({placeholders})", [bind(value) for value in values]
    if op is FilterOp.NOT_IN:
        values = list(node.value)
        placeholders = ", ".join(f"%s{suffix}" for _ in values)
        return f"{column} NOT IN ({placeholders})", [bind(value) for value in values]
    if op is FilterOp.GT:
        return f"{column} > %s{suffix}", [bind(node.value)]
    if op is FilterOp.GTE:
        return f"{column} >= %s{suffix}", [bind(node.value)]
    if op is FilterOp.LT:
        return f"{column} < %s{suffix}", [bind(node.value)]
    if op is FilterOp.LTE:
        return f"{column} <= %s{suffix}", [bind(node.value)]
    if op is FilterOp.EXISTS:
        return (
            f"{column} IS NOT NULL" if node.value else f"{column} IS NULL"
        ), []
    raise UnsupportedFilterOperationError(
        f"the PostgreSQL adapter does not support the filter operation {op.value!r}"
    )


def payload_index_expression(field: str) -> str:
    """Return the expression of a metadata payload index."""
    validate_identifier(field, "metadata field name")
    return f"({METADATA_COLUMN}->>'{field}')"


def distance_operator(metric: DistanceMetric | None) -> str:
    """Return the pgvector distance operator of a metric.

    Raises:
        VectorStoreConfigError: if pgvector has no operator for the metric.
    """
    if metric is None:
        return "<=>"
    try:
        return {
            DistanceMetric.COSINE: "<=>",
            DistanceMetric.EUCLIDEAN: "<->",
            DistanceMetric.DOT: "<#>",
        }[metric]
    except KeyError as exc:
        raise VectorStoreConfigError(
            f"pgvector does not support the {metric.value!r} metric "
            "(supported: cosine, euclidean, dot)"
        ) from exc


def operator_class(metric: DistanceMetric | None) -> str:
    """Return the pgvector operator class of a metric."""
    if metric is None:
        return "vector_cosine_ops"
    try:
        return {
            DistanceMetric.COSINE: "vector_cosine_ops",
            DistanceMetric.EUCLIDEAN: "vector_l2_ops",
            DistanceMetric.DOT: "vector_ip_ops",
        }[metric]
    except KeyError as exc:
        raise VectorStoreConfigError(
            f"pgvector does not support the {metric.value!r} metric"
        ) from exc


def canonical_score(distance: float, metric: DistanceMetric | None) -> float:
    """Convert a pgvector distance into the canonical score of the metric.

    pgvector reports distances: the cosine distance (``<=>``), the euclidean
    distance (``<->``) and the *negative* inner product (``<#>``).
    """
    if metric is None:
        metric = DistanceMetric.COSINE
    if metric is DistanceMetric.COSINE:
        return 1.0 - distance
    if metric is DistanceMetric.DOT:
        return -distance
    return distance


def sparse_dimensions(indices: list[int]) -> int:
    """Return the number of dimensions of a sparse vector.

    pgvector stores the dimensions of a ``sparsevec`` from 0, like the
    embedding contract does: the number of dimensions is the highest index
    plus one.
    """
    if not indices:
        raise InvalidRecordError("a sparse vector must have at least one element")
    highest = max(indices) + 1
    if highest > MAX_SPARSE_DIMENSIONS:
        raise InvalidRecordError(
            f"a sparse vector may not exceed {MAX_SPARSE_DIMENSIONS} dimensions "
            f"(got {highest})"
        )
    if len(indices) > MAX_SPARSE_DIMENSIONS:
        raise InvalidRecordError(
            f"a sparse vector may not have more than {MAX_SPARSE_DIMENSIONS} "
            f"non-zero elements (got {len(indices)})"
        )
    return highest


def sparse_parameters(
    indices: list[int], values: list[float]
) -> tuple[dict[int, float], int]:
    """Return the pgvector representation of a sparse vector.

    The embedding contract and pgvector both count the dimensions of a
    sparse vector from 0, so the indices are used as they are and the values
    are never densified.  (Only the *text* format of ``sparsevec`` is
    1-based; the binary format the driver sends is not.)
    """
    if len(indices) != len(values):
        raise InvalidRecordError(
            f"a sparse vector has {len(indices)} indices but {len(values)} values"
        )
    dimensions = sparse_dimensions(indices)
    elements = {int(index): float(value) for index, value in zip(indices, values)}
    return elements, dimensions


def json_dumps(data: Mapping[str, Any]) -> str:
    """Serialize a mapping to JSON."""
    return json.dumps(dict(data), ensure_ascii=False, default=str)
