"""PostgreSQL (pgvector) adapter for the Prome vector-store contracts.

The adapter stores precomputed embeddings in a PostgreSQL table and searches
them with the pgvector operators.  It needs a PostgreSQL server that has the
``vector`` extension installed; the adapter never creates a database and
never talks to a hosted service.

The connection is explicit (a DSN), the transactions are explicit, and the
installed pgvector version is detected at connection time so that the
capability descriptor is accurate (sparse vectors need pgvector 0.7 or
newer).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from embeddings.contracts.models import RepresentationType
from vector_stores.contracts.capabilities import StoreCapabilities
from vector_stores.contracts.exceptions import (
    BackendConnectionError,
    CollectionExistsError,
    CollectionNotFoundError,
    InvalidRecordError,
    UnsupportedIndexOperationError,
    UnsupportedVectorTypeError,
    VectorStoreConfigError,
    VectorStoreError,
)
from vector_stores.contracts.filters import FIELD_OPS, MetadataFilter
from vector_stores.contracts.indexes import (
    IndexConfig,
    IndexDescription,
    IndexKind,
    IndexKindCapability,
    PayloadFieldType,
)
from vector_stores.contracts.models import (
    ChunkProvenance,
    CollectionConfig,
    CollectionInfo,
    DistanceMetric,
    ScrollPage,
    SearchRequest,
    SearchResult,
    SpaceVectors,
    StoredRecord,
    VectorSpaceConfig,
)
from vector_stores.contracts.payload import (
    decode_contract_payload,
    encode_contract_payload,
)
from vector_stores.contracts.validation import (
    validate_collection_config,
    validate_record,
    validate_search_request,
)
from vector_stores.pgvector._client import import_psycopg, register_vector
from vector_stores.pgvector.sql import (
    CHUNK_ID_COLUMN,
    CONFIG_TABLE,
    DOC_ID_COLUMN,
    EMBEDDING_SPACE_COLUMN,
    MAX_VECTOR_DIMENSIONS,
    METADATA_COLUMN,
    PAYLOAD_COLUMN,
    SPARSE_DIMENSIONS_TABLE,
    canonical_score,
    dense_column,
    distance_operator,
    json_dumps,
    multi_vector_column,
    operator_class,
    payload_index_expression,
    sparse_column,
    sparse_parameters,
    to_sql_filter,
    validate_identifier,
    vector_column,
)

logger = logging.getLogger(__name__)

#: The pgvector version that introduced the ``sparsevec`` type.
SPARSEVEC_MIN_VERSION = (0, 7)


@dataclass(frozen=True)
class PgVectorStoreConfig:
    """Connection settings of the PostgreSQL adapter."""

    dsn: str
    """Connection string of the PostgreSQL server."""

    schema: str = "public"
    """Schema holding the collection tables."""

    table_prefix: str = ""
    """Prefix added to the table name of every collection."""

    autocommit: bool = False
    """Run the statements outside a transaction (the default is transactional)."""


class PgVectorVectorStore:
    """Vector store backed by PostgreSQL and pgvector."""

    name = "pgvector"

    def __init__(
        self,
        config: PgVectorStoreConfig | None = None,
        *,
        connection: Any | None = None,
    ) -> None:
        """Create a store.

        Args:
            config: Connection settings; ``dsn`` is required.
            connection: An already opened ``psycopg`` connection. When given,
                the adapter uses it as is.
        """
        if config is None and connection is None:
            raise VectorStoreConfigError(
                "the PostgreSQL adapter needs a DSN (PgVectorStoreConfig) or an "
                "open connection"
            )
        self._config = config
        self._collections: dict[str, CollectionConfig] = {}
        self._closed = False
        self._extension_version: str | None = None
        self._current_collection: str = ""
        if connection is not None:
            self._connection = connection
            self._owns_connection = False
        else:
            self._connection = self._connect(config)
            self._owns_connection = True
        self._extension_version = self._detect_extension_version()

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _connect(config: PgVectorStoreConfig | None) -> Any:
        psycopg = import_psycopg()
        if config is None:  # pragma: no cover - guarded by the caller
            raise VectorStoreConfigError("a DSN is required")
        try:
            connection = psycopg.connect(config.dsn, autocommit=config.autocommit)
        except Exception as exc:
            raise BackendConnectionError(
                f"could not connect to the PostgreSQL server: {exc}"
            ) from exc
        try:
            register_vector(connection)
        except Exception as exc:
            connection.close()
            raise BackendConnectionError(
                f"could not register the pgvector types: {exc}"
            ) from exc
        return connection

    @classmethod
    def server(cls, dsn: str, **kwargs: Any) -> PgVectorVectorStore:
        """Create a store connected to a PostgreSQL server."""
        return cls(PgVectorStoreConfig(dsn=dsn, **kwargs))

    # ------------------------------------------------------------------
    # capabilities
    # ------------------------------------------------------------------
    @property
    def extension_version(self) -> str | None:
        """Return the installed pgvector version (``None`` when unknown)."""
        return self._extension_version

    @property
    def capabilities(self) -> StoreCapabilities:
        """Return the capabilities of the PostgreSQL backend."""
        sparse = self._supports_sparsevec()
        return StoreCapabilities(
            backend=self.name,
            dense_search=True,
            sparse_search=sparse,
            multi_vector_search=False,
            exact_search=True,
            filter_ops=frozenset(FIELD_OPS),
            payload_field_types=frozenset(PayloadFieldType),
            index_kinds={
                IndexKind.EXACT: IndexKindCapability(
                    kind=IndexKind.EXACT,
                    supported=True,
                    query_parameters=("exact",),
                    notes="Exact search: the planner is prevented from using the index.",
                ),
                IndexKind.HNSW: IndexKindCapability(
                    kind=IndexKind.HNSW,
                    supported=True,
                    build_parameters=("m", "ef_construction"),
                    query_parameters=("hnsw_ef_search", "hnsw_ef_search"),
                    notes="pgvector HNSW index (hnsw.m, hnsw.ef_construction).",
                ),
                IndexKind.IVF: IndexKindCapability(
                    kind=IndexKind.IVF,
                    supported=True,
                    build_parameters=("lists",),
                    query_parameters=("ivfflat_probes",),
                    notes="pgvector IVFFlat index (lists).",
                ),
                IndexKind.SPARSE_INVERTED: IndexKindCapability(
                    kind=IndexKind.SPARSE_INVERTED,
                    supported=False,
                    notes=(
                        "pgvector has no inverted index for sparsevec: the sparse "
                        "search is always an exact scan."
                    ),
                ),
                IndexKind.PAYLOAD: IndexKindCapability(
                    kind=IndexKind.PAYLOAD,
                    supported=True,
                    build_parameters=("field", "field_type"),
                    notes=(
                        "PostgreSQL index on the collection table: B-tree on the "
                        "document id, GIN on the metadata and expression indexes "
                        "on the declared metadata fields."
                    ),
                ),
                IndexKind.MULTI_VECTOR: IndexKindCapability(
                    kind=IndexKind.MULTI_VECTOR,
                    supported=False,
                    notes=(
                        "PostgreSQL has no late interaction operator; "
                        "multi-vectors are stored as JSONB but cannot be searched."
                    ),
                ),
            },
            persistence_modes=("server",),
            delete_by_filter=True,
            delete_by_document=True,
            create_index=True,
            delete_index=True,
            notes=(
                f"pgvector version {self._extension_version or 'unknown'}. "
                "Sparse vector search needs pgvector 0.7 or newer."
            ),
        )

    # ------------------------------------------------------------------
    # collection lifecycle
    # ------------------------------------------------------------------
    def create_collection(
        self, config: CollectionConfig, *, exist_ok: bool = False
    ) -> CollectionInfo:
        """Create a collection (a table)."""
        self._ensure_open()
        validate_collection_config(config)
        validate_identifier(config.name, "collection name")
        for space in config.vector_spaces.values():
            if space.vector_type is RepresentationType.DENSE:
                if space.dimensions and space.dimensions > MAX_VECTOR_DIMENSIONS:
                    raise VectorStoreConfigError(
                        f"pgvector supports at most {MAX_VECTOR_DIMENSIONS} "
                        f"dimensions per vector column, got {space.dimensions}"
                    )
            elif space.vector_type is RepresentationType.SPARSE and not self._supports_sparsevec():
                raise UnsupportedVectorTypeError(
                    "the installed pgvector version does not support the "
                    "sparsevec type (0.7 or newer is required)"
                )
        table = self._table_name(config.name)
        if self._table_exists(table):
            if not exist_ok:
                raise CollectionExistsError(
                    f"collection {config.name!r} already exists"
                )
            stored = self._read_collection_config(config.name)
            if stored is not None and stored != config:
                raise VectorStoreConfigError(
                    f"collection {config.name!r} already exists with a different "
                    "configuration; drop it first or use a different name"
                )
        else:
            self._create_table(config)
        self._store_collection_config(config)
        self._create_payload_indexes(config)
        self._collections[config.name] = config
        return self.describe_collection(config.name)

    def describe_collection(self, name: str) -> CollectionInfo:
        """Describe an existing collection."""
        self._ensure_open()
        table = self._table_name(name)
        if not self._table_exists(table):
            raise CollectionNotFoundError(f"collection {name!r} does not exist")
        config = self._collection_config(name)
        return CollectionInfo(
            name=name,
            description=config.description,
            vector_spaces=config.vector_spaces,
            point_count=self.count(name),
            indexes=tuple(self._describe_indexes(name)),
            backend=self.name,
            metadata=dict(config.metadata),
        )

    def list_collections(self) -> list[CollectionInfo]:
        """List the collections of the backend."""
        self._ensure_open()
        infos: list[CollectionInfo] = []
        for name in self._known_collections():
            try:
                infos.append(self.describe_collection(name))
            except (CollectionNotFoundError, VectorStoreConfigError):
                continue
        return infos

    def delete_collection(self, name: str) -> bool:
        """Delete a collection. Returns True when a collection was deleted."""
        self._ensure_open()
        table = self._table_name(name)
        if not self._table_exists(table):
            return False
        with self._connection.cursor() as cursor:
            cursor.execute(
                self._sql("DROP TABLE IF EXISTS {table} CASCADE").format(
                    table=self._identifier(table)
                )
            )
            cursor.execute(
                self._sql("DELETE FROM {schema}.{config} WHERE name = %s").format(
                    schema=self._identifier(self._config.schema),
                    config=self._identifier(CONFIG_TABLE),
                ),
                (name,),
            )
        try:
            self._ensure_sparse_dimensions_table()
            with self._connection.cursor() as cursor:
                cursor.execute(
                    self._sql(
                        "DELETE FROM {schema}.{table} WHERE collection = %s"
                    ).format(
                        schema=self._identifier(self._config.schema),
                        table=self._identifier(SPARSE_DIMENSIONS_TABLE),
                    ),
                    (name,),
                )
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("could not delete the sparse dimensions of %s: %s", name, exc)
        self._connection.commit()
        self._collections.pop(name, None)
        return True

    # ------------------------------------------------------------------
    # records
    # ------------------------------------------------------------------
    def upsert(self, collection: str, records: Sequence[StoredRecord]) -> int:
        """Insert or replace ``records``. Returns the number of records written."""
        self._ensure_open()
        records = list(records)
        if not records:
            return 0
        config = self._collection_config(collection)
        table = self._table_name(collection)
        rows = []
        self._current_collection = collection
        for record in records:
            validate_record(record, config)
        # The dimension count of a sparse space is fixed by the first batch:
        # pgvector only compares sparse vectors of the same width.
        for name, space in config.vector_spaces.items():
            if space.vector_type is not RepresentationType.SPARSE:
                continue
            required = 0
            for record in records:
                vectors = record.vectors.get(name)
                if vectors is None or not vectors.sparse_indices:
                    continue
                required = max(required, max(vectors.sparse_indices) + 1)
            if required:
                self._sparse_dimension(collection, name, required)
        for record in records:
            rows.append(self._record_values(record, config))
        columns = list(rows[0]["columns"])
        values: list[Any] = []
        placeholders: list[str] = []
        for row in rows:
            values.extend(row["values"])
            placeholders.append(
                "(" + ", ".join("%s" for _ in row["columns"]) + ")"
            )
        statement = self._sql(
            "INSERT INTO {table} ({columns}) VALUES {values} "
            "ON CONFLICT ({chunk_id}) DO UPDATE SET {assignments}"
        ).format(
            table=self._identifier(table),
            columns=self._column_list(columns),
            values=self._sql(", ").join(self._sql(value) for value in placeholders),
            chunk_id=self._identifier(CHUNK_ID_COLUMN),
            assignments=self._sql(", ").join(
                self._sql("{name} = EXCLUDED.{name}").format(
                    name=self._identifier(name)
                )
                for name in columns
                if name != CHUNK_ID_COLUMN
            ),
        )
        with self._connection.cursor() as cursor:
            cursor.execute(statement, values)
        self._connection.commit()
        return len(records)

    def upsert_one(self, collection: str, record: StoredRecord) -> int:
        """Insert or replace a single record."""
        return self.upsert(collection, [record])

    def get(self, collection: str, ids: Sequence[str]) -> list[StoredRecord]:
        """Fetch records by their public chunk ids."""
        self._ensure_open()
        ids = list(ids)
        if not ids:
            return []
        config = self._collection_config(collection)
        table = self._table_name(collection)
        statement = self._sql(
            "SELECT {columns} FROM {table} WHERE {chunk_id} = ANY(%s)"
        ).format(
            columns=self._column_list(self._selected_columns(config)),
            table=self._identifier(table),
            chunk_id=self._identifier(CHUNK_ID_COLUMN),
        )
        with self._connection.cursor() as cursor:
            cursor.execute(statement, (ids,))
            rows = cursor.fetchall()
        self._connection.commit()
        return [self._row_to_record(row, config) for row in rows]

    def delete(self, collection: str, ids: Sequence[str]) -> int:
        """Delete records by their public chunk ids."""
        self._ensure_open()
        ids = list(ids)
        if not ids:
            return 0
        table = self._table_name(collection)
        statement = self._sql(
            "DELETE FROM {table} WHERE {chunk_id} = ANY(%s)"
        ).format(
            table=self._identifier(table), chunk_id=self._identifier(CHUNK_ID_COLUMN)
        )
        with self._connection.cursor() as cursor:
            cursor.execute(statement, (ids,))
            deleted = cursor.rowcount
        self._connection.commit()
        return max(deleted, 0)

    def delete_by_document(self, collection: str, doc_id: str) -> int:
        """Delete every record of a document."""
        from vector_stores.contracts.filters import eq

        return self.delete_by_filter(collection, eq(DOC_ID_COLUMN, doc_id))

    def delete_by_filter(self, collection: str, filter: MetadataFilter) -> int:
        """Delete every record matching ``filter``."""
        self._ensure_open()
        self._collection_config(collection)
        table = self._table_name(collection)
        expression, params = to_sql_filter(filter)
        statement = self._sql("DELETE FROM {table} WHERE {expression}").format(
            table=self._identifier(table), expression=self._sql(expression)
        )
        with self._connection.cursor() as cursor:
            cursor.execute(statement, params)
            deleted = cursor.rowcount
        self._connection.commit()
        return max(deleted, 0)

    def count(self, collection: str, filter: MetadataFilter | None = None) -> int:
        """Count the records of a collection, optionally filtered."""
        self._ensure_open()
        self._collection_config(collection)
        table = self._table_name(collection)
        if filter is None:
            statement = self._sql("SELECT COUNT(*) FROM {table}").format(
                table=self._identifier(table)
            )
            params: list[Any] = []
        else:
            expression, params = to_sql_filter(filter)
            statement = self._sql(
                "SELECT COUNT(*) FROM {table} WHERE {expression}"
            ).format(table=self._identifier(table), expression=self._sql(expression))
        with self._connection.cursor() as cursor:
            cursor.execute(statement, params)
            row = cursor.fetchone()
        self._connection.commit()
        return int(row[0]) if row else 0

    def scroll(
        self,
        collection: str,
        *,
        limit: int = 100,
        offset: int | str | None = None,
        filter: MetadataFilter | None = None,
        with_vectors: bool = False,
    ) -> ScrollPage:
        """Return one page of records."""
        self._ensure_open()
        if limit < 1:
            raise VectorStoreConfigError(f"scroll limit must be >= 1, got {limit}")
        config = self._collection_config(collection)
        table = self._table_name(collection)
        start = int(offset) if offset is not None else 0
        if filter is None:
            statement = self._sql(
                "SELECT {columns} FROM {table} ORDER BY {chunk_id} LIMIT %s OFFSET %s"
            ).format(
                columns=self._column_list(self._selected_columns(config)),
                table=self._identifier(table),
                chunk_id=self._identifier(CHUNK_ID_COLUMN),
            )
            params: list[Any] = [limit, start]
        else:
            expression, filter_params = to_sql_filter(filter)
            statement = self._sql(
                "SELECT {columns} FROM {table} WHERE {expression} "
                "ORDER BY {chunk_id} LIMIT %s OFFSET %s"
            ).format(
                columns=self._column_list(self._selected_columns(config)),
                table=self._identifier(table),
                expression=self._sql(expression),
                chunk_id=self._identifier(CHUNK_ID_COLUMN),
            )
            params = [*filter_params, limit, start]
        with self._connection.cursor() as cursor:
            cursor.execute(statement, params)
            rows = cursor.fetchall()
        self._connection.commit()
        records = [self._row_to_record(row, config) for row in rows]
        next_offset = start + len(rows) if len(rows) == limit else None
        return ScrollPage(records=records, next_offset=next_offset)

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------
    def search(self, request: SearchRequest) -> list[SearchResult]:
        """Run a vector search and return the hits, best first."""
        self._ensure_open()
        config = self._collection_config(request.collection)
        space = config.vector_spaces.get(request.space)
        if space is None:
            raise VectorStoreConfigError(
                f"collection {request.collection!r} has no vector space "
                f"{request.space!r}; known spaces: {sorted(config.vector_spaces)}"
            )
        validate_search_request(request, space)
        if space.vector_type is RepresentationType.MULTI_VECTOR:
            raise UnsupportedVectorTypeError(
                "PostgreSQL has no late interaction operator: multi-vectors are "
                "stored but cannot be searched"
            )
        table = self._table_name(request.collection)
        column = self._identifier(vector_column(space.name, space.vector_type))
        operator = distance_operator(space.distance)
        metric = space.distance or DistanceMetric.COSINE

        if space.vector_type is RepresentationType.SPARSE:
            query = self._sparse_query(request)
        else:
            query = list(request.dense or [])

        exact = bool(request.params.get("exact"))
        cast = self._vector_cast(space)
        order = self._sql("{column} {operator} %s{cast}").format(
            column=column, operator=self._sql(operator), cast=self._sql(cast)
        )
        if exact:
            # The planner cannot use the index when the ordering expression is
            # wrapped: this is the exact-search baseline.
            order = self._sql("({column} {operator} %s{cast}) + 0").format(
                column=column,
                operator=self._sql(operator),
                cast=self._sql(cast),
            )
        statement = self._sql(
            "SELECT {chunk_id}, {doc_id}, {payload}, {metadata}, "
            "{column} {operator} %s{cast} AS distance FROM {table} {where} "
            "ORDER BY {order} LIMIT %s"
        ).format(
            chunk_id=self._identifier(CHUNK_ID_COLUMN),
            doc_id=self._identifier(DOC_ID_COLUMN),
            payload=self._identifier(PAYLOAD_COLUMN),
            metadata=self._identifier(METADATA_COLUMN),
            column=column,
            operator=self._sql(operator),
            cast=self._sql(cast),
            table=self._identifier(table),
            where=self._sql("WHERE {expression}").format(
                expression=self._sql(to_sql_filter(request.filter)[0])
            )
            if request.filter is not None
            else self._sql(""),
            order=order,
        )
        params: list[Any] = [query]
        if request.filter is not None:
            params.extend(to_sql_filter(request.filter)[1])
        params.append(query)
        params.append(request.limit)
        with self._connection.cursor() as cursor:
            cursor.execute(statement, params)
            rows = cursor.fetchall()
        self._connection.commit()
        results = []
        for row in rows:
            record = self._row_to_record(row, config, raw=True)
            distance = row[4]
            results.append(
                SearchResult(
                    chunk_id=record.chunk.id,
                    doc_id=record.chunk.doc_id,
                    score=canonical_score(float(distance), metric),
                    space=request.space,
                    metric=metric,
                    score_direction=metric.score_direction,
                    chunk=record.chunk,
                    text=record.chunk.text,
                    metadata=record.metadata,
                    provenance=ChunkProvenance.from_chunk(record.chunk),
                )
            )
        return results

    @staticmethod
    def _vector_cast(space: VectorSpaceConfig) -> str:
        """Return the SQL cast of a query vector parameter."""
        if space.vector_type is RepresentationType.SPARSE:
            return "::sparsevec"
        return f"::vector({int(space.dimensions or 0)})"

    def _sparse_query(self, request: SearchRequest) -> Any:
        from pgvector import sparsevec

        elements, required = sparse_parameters(
            list(request.sparse_indices or []), list(request.sparse_values or [])
        )
        dimensions = self._sparse_dimension(request.collection, request.space)
        if dimensions is None:
            dimensions = required
        elif required > dimensions:
            raise InvalidRecordError(
                f"the sparse vector space {request.space!r} of "
                f"{request.collection!r} holds {dimensions} dimensions but the "
                f"query needs {required}"
            )
        return sparsevec.SparseVector(elements, dimensions)

    # ------------------------------------------------------------------
    # indexes
    # ------------------------------------------------------------------
    def create_index(self, collection: str, index: IndexConfig) -> IndexDescription:
        """Create an index on an existing collection."""
        self._ensure_open()
        config = self._collection_config(collection)
        table = self._table_name(collection)
        if index.kind is IndexKind.PAYLOAD:
            field = index.target
            validate_identifier(field, "metadata field name")
            self._create_expression_index(table, field)
            return IndexDescription(
                kind=IndexKind.PAYLOAD,
                target=field,
                field_type=index.field_type or PayloadFieldType.KEYWORD,
                build_params=dict(index.build_params),
            )
        if index.kind is IndexKind.EXACT:
            return IndexDescription(
                kind=IndexKind.EXACT,
                target=index.target,
                notes="Exact search is a query mode, not an index.",
            )
        if index.kind is IndexKind.SPARSE_INVERTED:
            raise UnsupportedIndexOperationError(
                "pgvector has no inverted index for sparsevec"
            )
        if index.kind is IndexKind.MULTI_VECTOR:
            raise UnsupportedIndexOperationError(
                "PostgreSQL has no late interaction index"
            )
        space = config.vector_spaces.get(index.target)
        if space is None:
            raise VectorStoreConfigError(
                f"collection {collection!r} has no vector space {index.target!r}"
            )
        if space.vector_type is not RepresentationType.DENSE:
            raise UnsupportedIndexOperationError(
                "only dense vector columns can be indexed in PostgreSQL"
            )
        column = dense_column(space.name)
        build = dict(index.build_params)
        if index.kind is IndexKind.HNSW:
            statement = self._sql(
                "CREATE INDEX IF NOT EXISTS {name} ON {table} USING hnsw "
                "({column} {ops}) WITH (m = {m}, ef_construction = {ef})"
            ).format(
                name=self._identifier(f"{table}_hnsw_{column}"),
                table=self._identifier(table),
                column=self._identifier(column),
                ops=self._sql(operator_class(space.distance)),
                m=int(build.get("m", 16)),
                ef=int(build.get("ef_construction", 64)),
            )
        else:
            statement = self._sql(
                "CREATE INDEX IF NOT EXISTS {name} ON {table} USING ivfflat "
                "({column} {ops}) WITH (lists = {lists})"
            ).format(
                name=self._identifier(f"{table}_ivfflat_{column}"),
                table=self._identifier(table),
                column=self._identifier(column),
                ops=self._sql(operator_class(space.distance)),
                lists=int(build.get("lists", 100)),
            )
        with self._connection.cursor() as cursor:
            cursor.execute(statement)
        self._connection.commit()
        return IndexDescription(
            kind=index.kind,
            target=space.name,
            build_params=build,
        )

    def delete_index(self, collection: str, name: str) -> bool:
        """Delete an index by name. Returns True when an index was deleted."""
        self._ensure_open()
        table = self._table_name(collection)
        with self._connection.cursor() as cursor:
            cursor.execute(
                self._sql("SELECT indexname FROM pg_indexes WHERE schemaname = %s "
                          "AND tablename = %s AND indexname = %s"),
                (self._config.schema, table, name),
            )
            row = cursor.fetchone()
        self._connection.commit()
        if row is None:
            return False
        with self._connection.cursor() as cursor:
            cursor.execute(
                self._sql("DROP INDEX IF EXISTS {schema}.{index}").format(
                    schema=self._identifier(self._config.schema),
                    index=self._identifier(name),
                )
            )
        self._connection.commit()
        return True

    def list_indexes(self, collection: str) -> list[IndexDescription]:
        """List the indexes of a collection."""
        self._ensure_open()
        return self._describe_indexes(collection)

    def _describe_indexes(self, name: str) -> list[IndexDescription]:
        """Describe the indexes of a collection."""
        table = self._table_name(name)
        with self._connection.cursor() as cursor:
            cursor.execute(
                self._sql(
                    "SELECT indexname, indexdef FROM pg_indexes "
                    "WHERE schemaname = %s AND tablename = %s ORDER BY indexname"
                ),
                (self._config.schema, table),
            )
            rows = cursor.fetchall()
        self._connection.commit()
        descriptions = []
        for index_name, definition in rows:
            description = str(definition)
            kind = IndexKind.PAYLOAD
            target = ""
            if "USING hnsw" in description:
                kind = IndexKind.HNSW
                target = self._target_from_index(description, "hnsw")
            elif "USING ivfflat" in description:
                kind = IndexKind.IVF
                target = self._target_from_index(description, "ivfflat")
            else:
                target = index_name
            descriptions.append(
                IndexDescription(
                    kind=kind,
                    target=target,
                    name=index_name,
                    build_params={"definition": description},
                )
            )
        return descriptions

    @staticmethod
    def _target_from_index(definition: str, method: str) -> str:
        marker = f"USING {method} ("
        start = definition.find(marker)
        if start < 0:
            return ""
        start += len(marker)
        end = definition.find(")", start)
        return definition[start:end].strip().strip('"')

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close the connection."""
        if self._closed:
            return
        self._closed = True
        if self._owns_connection:
            try:
                self._connection.close()
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("error while closing the PostgreSQL connection: %s", exc)

    def __enter__(self) -> PgVectorVectorStore:  # noqa: PYI034 - typing.Self needs 3.11
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _ensure_open(self) -> None:
        if self._closed:
            raise VectorStoreError("this vector store is closed")

    def _sql(self, text: str) -> Any:
        psycopg = import_psycopg()
        return psycopg.sql.SQL(text)

    def _identifier(self, name: str) -> Any:
        psycopg = import_psycopg()
        return psycopg.sql.Identifier(name)

    def _column_list(self, columns: Sequence[str]) -> Any:
        psycopg = import_psycopg()
        return psycopg.sql.SQL(", ").join(
            self._identifier(column) for column in columns
        )

    def _table_name(self, collection: str) -> str:
        return f"{self._config.table_prefix}{collection}"

    def _schema(self) -> str:
        return self._config.schema

    def _detect_extension_version(self) -> str | None:
        try:
            with self._connection.cursor() as cursor:
                cursor.execute(
                    "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
                )
                row = cursor.fetchone()
        except Exception as exc:
            raise BackendConnectionError(
                f"could not read the installed pgvector version: {exc}"
            ) from exc
        if row is None:
            raise BackendConnectionError(
                "the 'vector' extension is not installed in the database: run "
                "'CREATE EXTENSION vector;' as a superuser"
            )
        return str(row[0])

    def _supports_sparsevec(self) -> bool:
        version = self._extension_version
        if not version:
            return False
        try:
            parts = tuple(int(part) for part in version.split(".")[:2])
        except ValueError:  # pragma: no cover - defensive
            return False
        return parts >= SPARSEVEC_MIN_VERSION

    def _table_exists(self, table: str) -> bool:
        with self._connection.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM information_schema.tables WHERE table_schema = %s "
                "AND table_name = %s",
                (self._schema(), table),
            )
            return cursor.fetchone() is not None

    def _known_collections(self) -> list[str]:
        with self._connection.cursor() as cursor:
            cursor.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = %s "
                "AND tablename NOT LIKE 'prome_%%' ORDER BY tablename",
                (self._schema(),),
            )
            rows = cursor.fetchall()
        self._connection.commit()
        prefix = self._config.table_prefix
        names = []
        for (table,) in rows:
            name = table[len(prefix) :] if prefix and table.startswith(prefix) else table
            names.append(name)
        return names

    def _create_table(self, config: CollectionConfig) -> None:
        columns = [
            self._sql("{name} TEXT PRIMARY KEY").format(
                name=self._identifier(CHUNK_ID_COLUMN)
            ),
            self._sql("{name} TEXT NOT NULL").format(
                name=self._identifier(DOC_ID_COLUMN)
            ),
        ]
        for space in config.vector_spaces.values():
            if space.vector_type is RepresentationType.DENSE:
                columns.append(
                    self._sql("{name} vector({dimensions})").format(
                        name=self._identifier(dense_column(space.name)),
                        dimensions=int(space.dimensions or 0),
                    )
                )
            elif space.vector_type is RepresentationType.SPARSE:
                columns.append(
                    self._sql("{name} sparsevec").format(
                        name=self._identifier(sparse_column(space.name))
                    )
                )
            else:
                columns.append(
                    self._sql("{name} JSONB").format(
                        name=self._identifier(multi_vector_column(space.name))
                    )
                )
        columns.append(
            self._sql("{name} JSONB NOT NULL").format(
                name=self._identifier(PAYLOAD_COLUMN)
            )
        )
        columns.append(
            self._sql("{name} JSONB NOT NULL").format(
                name=self._identifier(METADATA_COLUMN)
            )
        )
        columns.append(
            self._sql("{name} JSONB NOT NULL").format(
                name=self._identifier(EMBEDDING_SPACE_COLUMN)
            )
        )
        statement = self._sql("CREATE TABLE {table} ({columns})").format(
            table=self._identifier(self._table_name(config.name)),
            columns=self._sql(", ").join(columns),
        )
        with self._connection.cursor() as cursor:
            cursor.execute(statement)
            cursor.execute(
                self._sql(
                    "CREATE INDEX IF NOT EXISTS {name} ON {table} ({column})"
                ).format(
                    name=self._identifier(f"{self._table_name(config.name)}_doc_id_idx"),
                    table=self._identifier(self._table_name(config.name)),
                    column=self._identifier(DOC_ID_COLUMN),
                )
            )
            cursor.execute(
                self._sql(
                    "CREATE INDEX IF NOT EXISTS {name} ON {table} "
                    "USING gin ({column})"
                ).format(
                    name=self._identifier(f"{self._table_name(config.name)}_metadata_gin"),
                    table=self._identifier(self._table_name(config.name)),
                    column=self._identifier(METADATA_COLUMN),
                )
            )
        self._connection.commit()

    def _create_payload_indexes(self, config: CollectionConfig) -> None:
        for spec in config.payload_indexes:
            self._create_expression_index(self._table_name(config.name), spec.field)

    def _create_expression_index(self, table: str, field: str) -> None:
        validate_identifier(field, "metadata field name")
        statement = self._sql(
            "CREATE INDEX IF NOT EXISTS {name} ON {table} ({expression})"
        ).format(
            name=self._identifier(f"{table}_meta_{field}_idx"),
            table=self._identifier(table),
            expression=self._sql(payload_index_expression(field)),
        )
        with self._connection.cursor() as cursor:
            cursor.execute(statement)
        self._connection.commit()

    def _ensure_sparse_dimensions_table(self) -> None:
        """Create the table that holds the sparse dimension counts."""
        with self._connection.cursor() as cursor:
            cursor.execute(
                self._sql(
                    "CREATE TABLE IF NOT EXISTS {schema}.{table} "
                    "(collection TEXT, space TEXT, dimensions INTEGER, "
                    "PRIMARY KEY (collection, space))"
                ).format(
                    schema=self._identifier(self._schema()),
                    table=self._identifier(SPARSE_DIMENSIONS_TABLE),
                )
            )
        self._connection.commit()

    def _sparse_dimension(
        self, collection: str, space: str, required: int | None = None
    ) -> int | None:
        """Return (and fix) the dimension count of a sparse vector space.

        pgvector compares two ``sparsevec`` values only when they have the
        same number of dimensions.  The embedding contract does not carry
        that count, so the adapter fixes it with the first batch of a
        collection and pads the vectors that follow.
        """
        self._ensure_sparse_dimensions_table()
        with self._connection.cursor() as cursor:
            cursor.execute(
                self._sql(
                    "SELECT dimensions FROM {schema}.{table} "
                    "WHERE collection = %s AND space = %s"
                ).format(
                    schema=self._identifier(self._schema()),
                    table=self._identifier(SPARSE_DIMENSIONS_TABLE),
                ),
                (collection, space),
            )
            row = cursor.fetchone()
        if row is not None:
            dimensions = int(row[0])
            if required is not None and required > dimensions:
                raise InvalidRecordError(
                    f"the sparse vector space {space!r} of {collection!r} holds "
                    f"{dimensions} dimensions but the record needs {required}"
                )
            self._connection.commit()
            return dimensions
        if required is None:
            self._connection.commit()
            return None
        with self._connection.cursor() as cursor:
            cursor.execute(
                self._sql(
                    "INSERT INTO {schema}.{table} (collection, space, dimensions) "
                    "VALUES (%s, %s, %s) ON CONFLICT (collection, space) DO NOTHING"
                ).format(
                    schema=self._identifier(self._schema()),
                    table=self._identifier(SPARSE_DIMENSIONS_TABLE),
                ),
                (collection, space, required),
            )
        self._connection.commit()
        return required

    def _store_collection_config(self, config: CollectionConfig) -> None:
        statement = self._sql(
            "CREATE TABLE IF NOT EXISTS {schema}.{config} "
            "(name TEXT PRIMARY KEY, config JSONB NOT NULL)"
        ).format(
            schema=self._identifier(self._schema()),
            config=self._identifier(CONFIG_TABLE),
        )
        with self._connection.cursor() as cursor:
            cursor.execute(statement)
            cursor.execute(
                self._sql(
                    "INSERT INTO {schema}.{config} (name, config) VALUES (%s, %s::jsonb) "
                    "ON CONFLICT (name) DO UPDATE SET config = EXCLUDED.config"
                ).format(
                    schema=self._identifier(self._schema()),
                    config=self._identifier(CONFIG_TABLE),
                ),
                (config.name, json_dumps(config.as_dict())),
            )
        self._connection.commit()

    def _read_collection_config(self, name: str) -> CollectionConfig | None:
        try:
            with self._connection.cursor() as cursor:
                cursor.execute(
                    self._sql(
                        "SELECT config FROM {schema}.{config} WHERE name = %s"
                    ).format(
                        schema=self._identifier(self._schema()),
                        config=self._identifier(CONFIG_TABLE),
                    ),
                    (name,),
                )
                row = cursor.fetchone()
        except Exception as exc:
            logger.debug("could not read the configuration of %s: %s", name, exc)
            return None
        if row is None:
            return None
        data = row[0]
        if isinstance(data, str):
            data = json_dumps(data)
        try:
            return CollectionConfig.from_dict(data)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None

    def _collection_config(self, name: str) -> CollectionConfig:
        if name in self._collections:
            return self._collections[name]
        table = self._table_name(name)
        if not self._table_exists(table):
            raise CollectionNotFoundError(f"collection {name!r} does not exist")
        stored = self._read_collection_config(name)
        if stored is None:
            raise VectorStoreConfigError(
                f"collection {name!r} carries no Prome configuration: it was not "
                "created by this adapter. Recreate it to use the vector store "
                "contract."
            )
        self._collections[name] = stored
        return stored

    def _record_values(
        self, record: StoredRecord, config: CollectionConfig
    ) -> dict[str, Any]:
        columns = [CHUNK_ID_COLUMN, DOC_ID_COLUMN]
        values: list[Any] = [record.chunk.id, record.chunk.doc_id]
        for name, vectors in record.vectors.items():
            space = config.vector_spaces[name]
            if space.vector_type is RepresentationType.DENSE:
                columns.append(dense_column(name))
                values.append(list(vectors.dense or []))
            elif space.vector_type is RepresentationType.SPARSE:
                elements, required = sparse_parameters(
                    list(vectors.sparse_indices or []),
                    list(vectors.sparse_values or []),
                )
                dimensions = self._sparse_dimension(
                    self._current_collection, name, required
                )
                columns.append(sparse_column(name))
                values.append(self._sparse_value(elements, dimensions or required))
            else:
                columns.append(multi_vector_column(name))
                values.append(json_dumps({"vectors": vectors.multi_vector or []}))
        columns.append(PAYLOAD_COLUMN)
        values.append(
            json_dumps(
                {
                    CHUNK_ID_COLUMN: record.chunk.id,
                    DOC_ID_COLUMN: record.chunk.doc_id,
                    "_prome": encode_contract_payload(record),
                }
            )
        )
        columns.append(METADATA_COLUMN)
        values.append(json_dumps(record.metadata))
        columns.append(EMBEDDING_SPACE_COLUMN)
        values.append(
            json_dumps(
                {
                    name: {
                        "model_id": vectors.embedding_space.model_id,
                        "model_revision": vectors.embedding_space.model_revision,
                        "dimensions": vectors.embedding_space.dimensions,
                        "representation_type": vectors.embedding_space.representation_type.value,
                        "normalization": vectors.embedding_space.normalization.value,
                    }
                    for name, vectors in record.vectors.items()
                }
            )
        )
        return {"columns": columns, "values": values}

    @staticmethod
    def _sparse_value(shifted: dict[int, float], dimensions: int) -> Any:
        from pgvector import sparsevec

        return sparsevec.SparseVector(shifted, dimensions)

    def _selected_columns(self, config: CollectionConfig) -> list[str]:
        return [CHUNK_ID_COLUMN, DOC_ID_COLUMN, PAYLOAD_COLUMN, METADATA_COLUMN]

    def _row_to_record(
        self, row: Sequence[Any], config: CollectionConfig, raw: bool = False
    ) -> StoredRecord:
        payload, metadata = row[2], row[3]
        if isinstance(payload, str):
            payload = json_dumps(payload)
        record = decode_contract_payload(payload)
        record.metadata = metadata if isinstance(metadata, dict) else {}
        space = self._default_space(config)
        record.vectors = {
            space.name: SpaceVectors(
                space=space.name,
                embedding_space=space.embedding_space,
            )
        }
        return record

    @staticmethod
    def _default_space(config: CollectionConfig) -> VectorSpaceConfig:
        for space in config.vector_spaces.values():
            return space
        raise VectorStoreConfigError(  # pragma: no cover - validated earlier
            f"collection {config.name!r} declares no vector space"
        )
