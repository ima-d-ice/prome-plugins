"""LanceDB adapter for the Prome vector-store contracts.

LanceDB is an embedded, local vector database: the data lives in a
directory, no server is needed and no hosted service is ever used.  The
adapter stores precomputed embeddings only - it never embeds text and never
loads a model.

What LanceDB gives us natively:

- **dense vector search** with IVF based indexes (``IVF_FLAT``, ``IVF_PQ``,
  ``IVF_HNSW_FLAT``, ``IVF_HNSW_PQ``, ``IVF_HNSW_SQ``, ``IVF_SQ``,
  ``IVF_RQ``);
- **exact dense search** (bypassing the vector index);
- **metadata filtering** pushed down into the scan;
- **scalar indexes** on the declared metadata fields;
- **multi-vector (late interaction) search**: the distance is the sum over
  the query vectors of the distance to the best matching stored vector, i.e.
  a max-sim style score.

What LanceDB does **not** give us: sparse vector search.  Sparse vectors are
stored (indices and weights, never densified) but cannot be searched; the
capability descriptor says so.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from embeddings.contracts.models import RepresentationType
from vector_stores.contracts.capabilities import StoreCapabilities
from vector_stores.contracts.exceptions import (
    BackendConnectionError,
    CollectionExistsError,
    CollectionNotFoundError,
    UnsupportedFilterOperationError,
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
    StoredRecord,
    VectorSpaceConfig,
)
from vector_stores.contracts.validation import (
    validate_collection_config,
    validate_record,
    validate_search_request,
)
from vector_stores.lancedb._client import import_lancedb, index_module
from vector_stores.lancedb.mapping import (
    CHUNK_ID_COLUMN,
    CONFIG_TABLE,
    DOC_ID_COLUMN,
    METADATA_COLUMN,
    PAYLOAD_COLUMN,
    build_schema,
    canonical_score,
    chunk_id_list,
    decode_row,
    encode_record,
    lance_distance_type,
    score_direction,
    to_sql_filter,
    vector_column,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LanceDBStoreConfig:
    """Connection settings of the LanceDB adapter.

    ``path`` is a local directory.  ``None`` selects the ephemeral mode: a
    temporary directory that is removed when the store is closed.
    """

    path: str | None = None

    @classmethod
    def local(cls, path: str) -> LanceDBStoreConfig:
        """Return the configuration of a persistent local directory."""
        return cls(path=path)

    @classmethod
    def memory(cls) -> LanceDBStoreConfig:
        """Return the configuration of the ephemeral (temporary) mode."""
        return cls(path=None)


class LanceDBVectorStore:
    """Vector store backed by LanceDB.

    A collection is a LanceDB table; the chunk id is the primary key, so the
    public chunk id is never replaced by a backend generated identifier.
    """

    name = "lancedb"

    def __init__(
        self,
        config: LanceDBStoreConfig | None = None,
        *,
        connection: Any | None = None,
    ) -> None:
        """Create a store.

        Args:
            config: Connection settings; defaults to the ephemeral mode.
            connection: An already opened ``lancedb.DBConnection``. When
                given, the adapter uses it as is.
        """
        self._config = config or LanceDBStoreConfig()
        self._collections: dict[str, CollectionConfig] = {}
        self._closed = False
        self._temporary_path: str | None = None
        if connection is not None:
            self._db = connection
            self._temporary_path = None
        else:
            self._db = self._connect(self._config)

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _connect(config: LanceDBStoreConfig) -> Any:
        lancedb = import_lancedb()
        path = config.path
        temporary: str | None = None
        if path is None:
            temporary = tempfile.mkdtemp(prefix="prome-lancedb-")
            path = temporary
        try:
            return lancedb.connect(path)
        except Exception as exc:
            if temporary:
                shutil.rmtree(temporary, ignore_errors=True)
            raise BackendConnectionError(
                f"could not open the LanceDB backend at {path!r}: {exc}"
            ) from exc

    @classmethod
    def local(cls, path: str) -> LanceDBVectorStore:
        """Create a store using a persistent local directory."""
        return cls(LanceDBStoreConfig.local(path))

    @classmethod
    def memory(cls) -> LanceDBVectorStore:
        """Create a store using an ephemeral temporary directory."""
        return cls(LanceDBStoreConfig.memory())

    # ------------------------------------------------------------------
    # capabilities
    # ------------------------------------------------------------------
    @property
    def capabilities(self) -> StoreCapabilities:
        """Return the capabilities of the LanceDB backend."""
        return StoreCapabilities(
            backend=self.name,
            dense_search=True,
            sparse_search=False,
            multi_vector_search=True,
            exact_search=True,
            filter_ops=frozenset(FIELD_OPS),
            payload_field_types=frozenset(PayloadFieldType),
            index_kinds={
                IndexKind.EXACT: IndexKindCapability(
                    kind=IndexKind.EXACT,
                    supported=True,
                    query_parameters=("exact",),
                    notes="Exact search bypasses the vector index (brute force).",
                ),
                IndexKind.HNSW: IndexKindCapability(
                    kind=IndexKind.HNSW,
                    supported=True,
                    build_parameters=("m", "ef_construction", "num_partitions"),
                    query_parameters=("nprobes", "refine_factor"),
                    notes=(
                        "Mapped to the LanceDB IVF_HNSW_FLAT index type "
                        "(IVF partitions + an HNSW graph inside each partition)."
                    ),
                ),
                IndexKind.SPARSE_INVERTED: IndexKindCapability(
                    kind=IndexKind.SPARSE_INVERTED,
                    supported=False,
                    notes=(
                        "LanceDB has no sparse vector index: sparse vectors are "
                        "stored (indices and weights) but cannot be searched."
                    ),
                ),
                IndexKind.PAYLOAD: IndexKindCapability(
                    kind=IndexKind.PAYLOAD,
                    supported=True,
                    build_parameters=("field", "field_type"),
                    notes=(
                        "LanceDB scalar index (BTREE, BITMAP or LABEL_LIST) on "
                        "a declared metadata field."
                    ),
                ),
                IndexKind.MULTI_VECTOR: IndexKindCapability(
                    kind=IndexKind.MULTI_VECTOR,
                    supported=True,
                    notes=(
                        "Native late interaction search: the distance is the sum "
                        "over the query vectors of the distance to the best "
                        "matching stored vector (max-sim style)."
                    ),
                ),
            },
            persistence_modes=("memory", "local"),
            delete_by_filter=True,
            delete_by_document=True,
            filter_requires_declared_fields=True,
            create_index=True,
            delete_index=True,
            notes=(
                "LanceDB has no sparse vector search: sparse vectors are stored "
                "but never searched. Filters can only be pushed down to the "
                "metadata fields declared in the collection configuration; a "
                "filter on another field raises UnsupportedFilterOperationError."
            ),
        )

    # ------------------------------------------------------------------
    # collection lifecycle
    # ------------------------------------------------------------------
    def create_collection(
        self, config: CollectionConfig, *, exist_ok: bool = False
    ) -> CollectionInfo:
        """Create a collection (a LanceDB table)."""
        self._ensure_open()
        validate_collection_config(config)
        if config.name == CONFIG_TABLE:
            raise VectorStoreConfigError(
                f"{CONFIG_TABLE!r} is reserved by the LanceDB adapter"
            )
        exists = config.name in self._table_names()
        if exists and not exist_ok:
            raise CollectionExistsError(
                f"collection {config.name!r} already exists"
            )
        if exists:
            self._check_matching_configuration(config)
        else:
            schema = build_schema(config)
            try:
                self._db.create_table(config.name, schema=schema)
            except Exception as exc:
                raise self._translate(
                    exc, f"create the collection {config.name!r}"
                ) from exc
        self._store_collection_config(config)
        for spec in config.payload_indexes:
            self._create_scalar_index(config.name, spec.field, spec.field_type)
        self._collections[config.name] = config
        return self.describe_collection(config.name)

    def describe_collection(self, name: str) -> CollectionInfo:
        """Describe an existing collection."""
        self._ensure_open()
        table = self._open_table(name)
        config = self._collection_config(name)
        try:
            point_count = table.count_rows()
        except Exception as exc:  # pragma: no cover - count is best effort
            logger.debug("could not count the rows of %s: %s", name, exc)
            point_count = None
        return CollectionInfo(
            name=name,
            description=config.description,
            vector_spaces=config.vector_spaces,
            point_count=point_count,
            indexes=tuple(self._describe_indexes(table, config)),
            backend=self.name,
            metadata=dict(config.metadata),
        )

    def list_collections(self) -> list[CollectionInfo]:
        """List the collections of the backend."""
        self._ensure_open()
        infos: list[CollectionInfo] = []
        for name in sorted(self._table_names()):
            if name == CONFIG_TABLE:
                continue
            try:
                infos.append(self.describe_collection(name))
            except (CollectionNotFoundError, VectorStoreConfigError):
                continue
        return infos

    def delete_collection(self, name: str) -> bool:
        """Delete a collection. Returns True when a collection was deleted."""
        self._ensure_open()
        if name not in self._table_names():
            return False
        try:
            self._db.drop_table(name)
        except Exception as exc:
            raise self._translate(exc, f"delete the collection {name!r}") from exc
        self._collections.pop(name, None)
        self._delete_collection_config(name)
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
        table = self._open_table(collection)
        rows = []
        for record in records:
            validate_record(record, config)
            rows.append(encode_record(record, config))
        try:
            merge = table.merge_insert(CHUNK_ID_COLUMN)
            merge.when_matched_update_all().when_not_matched_insert_all().execute(rows)
        except Exception as exc:
            raise self._translate(exc, f"upsert {len(rows)} record(s)") from exc
        return len(rows)

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
        table = self._open_table(collection)
        try:
            rows = (
                table.search(None)
                .where(chunk_id_list(ids))
                .limit(len(ids))
                .to_list()
            )
        except Exception as exc:
            raise self._translate(exc, "fetch records") from exc
        return [decode_row(row, config) for row in rows]

    def delete(self, collection: str, ids: Sequence[str]) -> int:
        """Delete records by their public chunk ids."""
        self._ensure_open()
        ids = list(ids)
        if not ids:
            return 0
        table = self._open_table(collection)
        before = self.count(collection)
        try:
            table.delete(chunk_id_list(ids))
        except Exception as exc:
            raise self._translate(exc, "delete records") from exc
        return max(before - self.count(collection), 0)

    def delete_by_document(self, collection: str, doc_id: str) -> int:
        """Delete every record of a document."""
        from vector_stores.contracts.filters import eq

        return self.delete_by_filter(collection, eq(DOC_ID_COLUMN, doc_id))

    def delete_by_filter(self, collection: str, filter: MetadataFilter) -> int:
        """Delete every record matching ``filter``."""
        self._ensure_open()
        config = self._collection_config(collection)
        table = self._open_table(collection)
        expression = to_sql_filter(filter, config)
        before = self.count(collection)
        try:
            table.delete(expression)
        except Exception as exc:
            raise self._translate(exc, "delete filtered records") from exc
        return max(before - self.count(collection), 0)

    def count(self, collection: str, filter: MetadataFilter | None = None) -> int:
        """Count the records of a collection, optionally filtered."""
        self._ensure_open()
        config = self._collection_config(collection)
        table = self._open_table(collection)
        expression = to_sql_filter(filter, config) if filter is not None else None
        try:
            return table.count_rows(expression)
        except Exception as exc:
            raise self._translate(exc, "count the records") from exc

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
        table = self._open_table(collection)
        expression = to_sql_filter(filter, config) if filter is not None else None
        start = int(offset) if offset is not None else 0
        query = table.search(None).limit(limit).offset(start)
        if expression:
            query = query.where(expression)
        try:
            rows = query.to_list()
        except Exception as exc:
            raise self._translate(exc, "scroll the collection") from exc
        records = [decode_row(row, config) for row in rows]
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
        kind = request.query_representation()
        if kind is RepresentationType.SPARSE:
            raise UnsupportedVectorTypeError(
                "LanceDB cannot search sparse vectors: sparse vectors are stored "
                "but LanceDB has no sparse index"
            )
        table = self._open_table(request.collection)
        column = vector_column(space)
        if kind is RepresentationType.DENSE:
            query: Any = [float(value) for value in (request.dense or [])]
            query_vectors = 1
        else:
            query = [
                [float(value) for value in row] for row in (request.multi_vector or [])
            ]
            query_vectors = len(query)

        builder = table.search(query, vector_column_name=column)
        builder = builder.distance_type(lance_distance_type(space.distance))
        if request.filter is not None:
            builder = builder.where(
                to_sql_filter(request.filter, config), prefilter=True
            )
        if request.params.get("exact"):
            builder = builder.bypass_vector_index()
        nprobes = request.params.get("nprobes")
        if nprobes is not None:
            builder = builder.nprobes(int(nprobes))
        refine_factor = request.params.get("refine_factor")
        if refine_factor is not None:
            builder = builder.refine_factor(int(refine_factor))
        builder = builder.limit(request.limit)
        if request.offset:
            builder = builder.offset(request.offset)
        columns = self._select_columns(request, space, config)
        if columns is not None:
            builder = builder.select(columns)
        try:
            rows = builder.to_list()
        except Exception as exc:
            raise self._translate(exc, "search the collection") from exc

        metric = space.distance or DistanceMetric.COSINE
        results: list[SearchResult] = []
        for row in rows:
            distance = float(row.get("_distance", 0.0))
            record = decode_row(row, config)
            has_contract_payload = PAYLOAD_COLUMN in row
            results.append(
                SearchResult(
                    chunk_id=record.chunk.id,
                    doc_id=record.chunk.doc_id,
                    score=canonical_score(
                        distance, metric, query_vectors=query_vectors
                    ),
                    space=request.space,
                    metric=metric,
                    score_direction=score_direction(metric),
                    chunk=record.chunk if has_contract_payload else None,
                    text=record.chunk.text if has_contract_payload else None,
                    metadata=record.metadata,
                    provenance=(
                        ChunkProvenance.from_chunk(record.chunk)
                        if has_contract_payload
                        else None
                    ),
                )
            )
        return results

    # ------------------------------------------------------------------
    # indexes
    # ------------------------------------------------------------------
    def create_index(self, collection: str, index: IndexConfig) -> IndexDescription:
        """Create an index on an existing collection."""
        self._ensure_open()
        config = self._collection_config(collection)
        if index.kind is IndexKind.PAYLOAD:
            field_type = index.field_type or PayloadFieldType.KEYWORD
            self._create_scalar_index(collection, index.target, field_type)
            return IndexDescription(
                kind=IndexKind.PAYLOAD,
                target=index.target,
                field_type=field_type,
                build_params=dict(index.build_params),
                backend_options=dict(index.backend_options),
            )
        if index.kind is IndexKind.EXACT:
            # Exact search is a query mode, not an index: it needs no artifact.
            return IndexDescription(
                kind=IndexKind.EXACT,
                target=index.target,
                notes="Exact search bypasses the vector index; nothing to create.",
            )
        if index.kind is IndexKind.SPARSE_INVERTED:
            raise UnsupportedIndexOperationError(
                "LanceDB has no sparse vector index; sparse vectors are stored but "
                "cannot be searched"
            )
        space = config.vector_spaces.get(index.target)
        if space is None:
            raise VectorStoreConfigError(
                f"collection {collection!r} has no vector space {index.target!r}"
            )
        description = self._create_vector_index(collection, space, index)
        return description

    def delete_index(self, collection: str, name: str) -> bool:
        """Delete an index by name. Returns True when an index was deleted."""
        self._ensure_open()
        table = self._open_table(collection)
        for description in self._describe_indexes(table, None):
            if description.index_name != name:
                continue
            try:
                table.drop_index(name)
            except Exception as exc:
                raise self._translate(exc, f"delete the index {name!r}") from exc
            return True
        return False

    def list_indexes(self, collection: str) -> list[IndexDescription]:
        """List the indexes of a collection."""
        self._ensure_open()
        table = self._open_table(collection)
        config = self._collection_config(collection)
        return self._describe_indexes(table, config)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release the backend resources held by this store."""
        if self._closed:
            return
        self._closed = True
        close = getattr(self._db, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("error while closing the LanceDB connection: %s", exc)
        if self._temporary_path:
            shutil.rmtree(self._temporary_path, ignore_errors=True)
            self._temporary_path = None

    def __enter__(self) -> LanceDBVectorStore:  # noqa: PYI034 - typing.Self needs 3.11
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _ensure_open(self) -> None:
        if self._closed:
            raise VectorStoreError("this vector store is closed")

    def _table_names(self) -> list[str]:
        lister = getattr(self._db, "list_tables", None) or self._db.table_names
        try:
            return list(lister())
        except Exception as exc:
            raise self._translate(exc, "list the tables") from exc

    def _open_table(self, name: str) -> Any:
        try:
            return self._db.open_table(name)
        except Exception as exc:
            message = str(exc).lower()
            if "not found" in message or "does not exist" in message:
                raise CollectionNotFoundError(
                    f"collection {name!r} does not exist"
                ) from exc
            raise self._translate(exc, f"open the collection {name!r}") from exc

    def _collection_config(self, name: str) -> CollectionConfig:
        """Return the declared configuration of a collection."""
        if name in self._collections:
            return self._collections[name]
        if name not in self._table_names():
            raise CollectionNotFoundError(f"collection {name!r} does not exist")
        stored = self._read_collection_config(name)
        if stored is None:
            raise VectorStoreConfigError(
                f"collection {name!r} carries no Prome configuration: it was not "
                "created by this adapter. Recreate it to use the vector store "
                "contract."
            )
        config = CollectionConfig.from_dict(stored)
        self._collections[name] = config
        return config

    def _check_matching_configuration(self, config: CollectionConfig) -> None:
        stored = self._collection_config(config.name)
        expected = {
            name: space.as_dict() for name, space in config.vector_spaces.items()
        }
        actual = {
            name: space.as_dict() for name, space in stored.vector_spaces.items()
        }
        if expected != actual:
            raise VectorStoreConfigError(
                f"collection {config.name!r} already exists with a different vector "
                "configuration; drop it first or use a different name"
            )

    def _config_table(self) -> Any:
        if CONFIG_TABLE not in self._table_names():
            pa = __import__("pyarrow")
            schema = pa.schema(
                [
                    pa.field("name", pa.string()),
                    pa.field("config", pa.string()),
                ]
            )
            try:
                self._db.create_table(CONFIG_TABLE, schema=schema)
            except Exception as exc:  # pragma: no cover - race condition
                logger.debug("could not create the config table: %s", exc)
        return self._open_table(CONFIG_TABLE)

    def _store_collection_config(self, config: CollectionConfig) -> None:
        import json

        table = self._config_table()
        row = {"name": config.name, "config": json.dumps(config.as_dict(), default=str)}
        try:
            merge = table.merge_insert("name")
            merge.when_matched_update_all().when_not_matched_insert_all().execute([row])
        except Exception as exc:
            raise self._translate(
                exc, f"store the configuration of {config.name!r}"
            ) from exc

    def _read_collection_config(self, name: str) -> dict[str, Any] | None:
        import json

        if CONFIG_TABLE not in self._table_names():
            return None
        table = self._open_table(CONFIG_TABLE)
        try:
            rows = (
                table.search(None).where(f"name = '{name}'").limit(1).to_list()
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("could not read the configuration of %s: %s", name, exc)
            return None
        if not rows:
            return None
        raw = rows[0].get("config")
        try:
            return json.loads(raw)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None

    def _delete_collection_config(self, name: str) -> None:
        if CONFIG_TABLE not in self._table_names():
            return
        table = self._open_table(CONFIG_TABLE)
        try:
            table.delete(f"name = '{name}'")
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("could not delete the configuration of %s: %s", name, exc)

    def _create_scalar_index(
        self, collection: str, field: str, field_type: PayloadFieldType
    ) -> None:
        index = index_module()
        config = self._collection_config(collection)
        declared = {spec.field for spec in config.payload_indexes}
        if field not in declared:
            raise UnsupportedFilterOperationError(
                f"LanceDB can only index the metadata fields declared in the "
                f"collection configuration; {field!r} is not declared "
                f"(declared: {sorted(declared)})"
            )
        options = {
            PayloadFieldType.KEYWORD: index.BTree,
            PayloadFieldType.INTEGER: index.BTree,
            PayloadFieldType.FLOAT: index.BTree,
            PayloadFieldType.BOOLEAN: index.BTree,
            PayloadFieldType.DATETIME: index.BTree,
        }
        table = self._open_table(collection)
        try:
            table.create_index(
                f"{METADATA_COLUMN}.{field}", config=options[field_type]()
            )
        except Exception as exc:
            raise self._translate(
                exc, f"create the scalar index {field!r} on {collection!r}"
            ) from exc

    def _create_vector_index(
        self, collection: str, space: VectorSpaceConfig, index: IndexConfig
    ) -> IndexDescription:
        index_module_ = index_module()
        table = self._open_table(collection)
        column = vector_column(space)
        options = dict(index.backend_options)
        index_type = str(options.pop("index_type", "IVF_HNSW_FLAT"))
        distance_type = lance_distance_type(space.distance)
        build = dict(index.build_params)
        num_partitions = build.get("num_partitions", options.pop("num_partitions", None))
        if num_partitions is not None:
            num_partitions = int(num_partitions)
        if index_type == "IVF_FLAT":
            config_obj = index_module_.IvfFlat(
                distance_type=distance_type, num_partitions=num_partitions
            )
        elif index_type == "IVF_PQ":
            config_obj = index_module_.IvfPq(
                distance_type=distance_type,
                num_partitions=num_partitions,
                num_sub_vectors=build.get("num_sub_vectors"),
            )
        elif index_type == "IVF_HNSW_FLAT":
            config_obj = index_module_.IvfHnswFlat(
                distance_type=distance_type,
                num_partitions=num_partitions,
                m=int(build.get("m", 20)),
                ef_construction=int(build.get("ef_construction", 300)),
            )
        elif index_type == "IVF_HNSW_PQ":
            config_obj = index_module_.IvfHnswPq(
                distance_type=distance_type,
                num_partitions=num_partitions,
                num_sub_vectors=build.get("num_sub_vectors"),
                m=int(build.get("m", 20)),
                ef_construction=int(build.get("ef_construction", 300)),
            )
        else:
            raise UnsupportedIndexOperationError(
                f"LanceDB does not support the vector index type {index_type!r} "
                "(supported: IVF_FLAT, IVF_PQ, IVF_HNSW_FLAT, IVF_HNSW_PQ)"
            )
        try:
            table.create_index(column, config=config_obj)
        except Exception as exc:
            raise self._translate(
                exc, f"create the {index_type} index on {collection!r}"
            ) from exc
        return IndexDescription(
            kind=IndexKind.HNSW
            if "HNSW" in index_type
            else IndexKind.MULTI_VECTOR
            if space.vector_type is RepresentationType.MULTI_VECTOR
            else IndexKind.EXACT,
            target=space.name,
            build_params={**dict(build), "index_type": index_type},
            backend_options=options,
        )

    def _describe_indexes(
        self, table: Any, config: CollectionConfig | None
    ) -> list[IndexDescription]:
        descriptions: list[IndexDescription] = []
        try:
            indices = list(table.list_indices())
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("could not list the indices: %s", exc)
            return descriptions
        for index in indices:
            columns = list(getattr(index, "columns", []) or [])
            index_type = str(getattr(index, "index_type", ""))
            target = columns[0] if columns else getattr(index, "name", "")
            kind = IndexKind.PAYLOAD
            if index_type.startswith("Ivf"):
                kind = IndexKind.HNSW if "Hnsw" in index_type else IndexKind.EXACT
            if target.startswith(f"{METADATA_COLUMN}."):
                target = target[len(METADATA_COLUMN) + 1 :]
            elif target.startswith(("v_", "m_")):
                target = target[2:]
            descriptions.append(
                IndexDescription(
                    kind=kind,
                    target=target,
                    name=getattr(index, "name", None),
                    build_params={"index_type": index_type},
                    backend_options={
                        "columns": columns,
                        "num_indexed_rows": str(
                            getattr(index, "num_indexed_rows", "")
                        ),
                    },
                )
            )
        return descriptions

    @staticmethod
    def _select_columns(
        request: SearchRequest, space: VectorSpaceConfig, config: CollectionConfig
    ) -> list[str] | None:
        if request.with_payload is True:
            columns = None
        elif request.with_payload is False:
            columns = [CHUNK_ID_COLUMN, DOC_ID_COLUMN]
        else:
            columns = list(request.with_payload)
            for column in (CHUNK_ID_COLUMN, DOC_ID_COLUMN):
                if column not in columns:
                    columns.append(column)
        if request.with_vectors:
            column = vector_column(space)
            if columns is not None and column not in columns:
                columns.append(column)
        return columns

    @staticmethod
    def _translate(exc: Exception, action: str) -> VectorStoreError:
        if isinstance(exc, VectorStoreError):
            return exc
        message = str(exc)
        lowered = message.lower()
        if "not found" in lowered or "does not exist" in lowered:
            return CollectionNotFoundError(message)
        if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
            return BackendConnectionError(f"could not {action}: {message}")
        name = type(exc).__name__
        if "Connect" in name or "Timeout" in name:
            return BackendConnectionError(f"could not {action}: {message}")
        return VectorStoreError(f"could not {action}: {message}")
