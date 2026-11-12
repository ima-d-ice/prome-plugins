"""Qdrant adapter for the Prome vector-store contracts.

The adapter stores precomputed embeddings only: it never embeds text and
never loads a model.  It supports the three deployment shapes of Qdrant that
matter for local development and self-hosted deployments:

- **memory**: the in-process, non persistent mode of the Qdrant client;
- **local**: the embedded, on-disk mode (a directory, no server needed);
- **server**: a separately running Qdrant server (``url=`` or ``host=``).

Docker, Qdrant Cloud and any other hosted service are never used.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from embeddings.contracts.models import RepresentationType
from vector_stores.contracts.capabilities import StoreCapabilities
from vector_stores.contracts.exceptions import (
    BackendConnectionError,
    CollectionExistsError,
    CollectionNotFoundError,
    InvalidRecordError,
    UnsupportedIndexOperationError,
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
    ScoreDirection,
    ScrollPage,
    SearchRequest,
    SearchResult,
    StoredRecord,
)
from vector_stores.contracts.validation import (
    validate_collection_config,
    validate_record,
    validate_search_request,
)
from vector_stores.qdrant._client import import_qdrant, models
from vector_stores.qdrant.mapping import (
    CONTRACT_PAYLOAD_KEY,
    decode_payload,
    encode_payload,
    from_qdrant_payload_type,
    point_id,
    to_qdrant_filter,
    to_qdrant_payload_type,
    to_qdrant_search_params,
    to_qdrant_vectors_config,
)

logger = logging.getLogger(__name__)


class QdrantMode(str, Enum):
    """How the adapter talks to Qdrant."""

    MEMORY = "memory"
    """In-process, non persistent (tests, throwaway data)."""

    LOCAL = "local"
    """Embedded on-disk mode (a directory, no server)."""

    SERVER = "server"
    """A separately running Qdrant server."""


@dataclass(frozen=True)
class QdrantStoreConfig:
    """Connection settings of the Qdrant adapter."""

    mode: QdrantMode = QdrantMode.MEMORY
    path: str | None = None
    """Directory of the embedded mode (``":memory:"`` for the in-memory mode)."""

    url: str | None = None
    """Base URL of a running Qdrant server."""

    host: str | None = None
    port: int | None = None
    grpc_port: int | None = None
    api_key: str | None = None
    https: bool | None = None
    timeout: int | None = None
    prefer_grpc: bool = False

    @classmethod
    def memory(cls) -> QdrantStoreConfig:
        """Return the configuration of the in-memory mode."""
        return cls(mode=QdrantMode.MEMORY, path=":memory:")

    @classmethod
    def local(cls, path: str) -> QdrantStoreConfig:
        """Return the configuration of the embedded on-disk mode."""
        return cls(mode=QdrantMode.LOCAL, path=path)

    @classmethod
    def server(cls, url: str | None = None, **kwargs: Any) -> QdrantStoreConfig:
        """Return the configuration of a running Qdrant server."""
        return cls(mode=QdrantMode.SERVER, url=url, **kwargs)


class QdrantVectorStore:
    """Vector store backed by Qdrant.

    The adapter keeps the public chunk id and document id of every record:
    Qdrant point ids are UUIDs, so the chunk id is mapped to a deterministic
    UUIDv5 and the original id is stored in the payload.
    """

    name = "qdrant"

    def __init__(
        self,
        config: QdrantStoreConfig | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        """Create a store.

        Args:
            config: Connection settings; defaults to the in-memory mode.
            client: An already built ``qdrant_client.QdrantClient``. When
                given, the adapter uses it as is (useful to share a server
                connection or to run against a test double).
        """
        self._config = config or QdrantStoreConfig()
        self._collections: dict[str, CollectionConfig] = {}
        self._closed = False
        if client is not None:
            self._client = client
        else:
            self._client = self._create_client(self._config)

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _create_client(config: QdrantStoreConfig) -> Any:
        qdrant_client, _ = import_qdrant()
        try:
            if config.mode is QdrantMode.SERVER:
                if config.url:
                    return qdrant_client.QdrantClient(
                        url=config.url,
                        api_key=config.api_key,
                        https=config.https,
                        timeout=config.timeout,
                        prefer_grpc=config.prefer_grpc,
                    )
                return qdrant_client.QdrantClient(
                    host=config.host,
                    port=config.port,
                    grpc_port=config.grpc_port,
                    api_key=config.api_key,
                    https=config.https,
                    timeout=config.timeout,
                    prefer_grpc=config.prefer_grpc,
                )
            path = config.path or ":memory:"
            return qdrant_client.QdrantClient(path=path)
        except ImportError:  # pragma: no cover - re-raised as a contract error
            raise
        except Exception as exc:
            raise BackendConnectionError(
                f"could not open the Qdrant backend: {exc}"
            ) from exc

    @classmethod
    def memory(cls) -> QdrantVectorStore:
        """Create a store using the in-memory mode."""
        return cls(QdrantStoreConfig.memory())

    @classmethod
    def local(cls, path: str) -> QdrantVectorStore:
        """Create a store using the embedded on-disk mode."""
        return cls(QdrantStoreConfig.local(path))

    @classmethod
    def server(cls, url: str | None = None, **kwargs: Any) -> QdrantVectorStore:
        """Create a store connected to a running Qdrant server."""
        return cls(QdrantStoreConfig.server(url, **kwargs))

    # ------------------------------------------------------------------
    # capabilities
    # ------------------------------------------------------------------
    @property
    def capabilities(self) -> StoreCapabilities:
        """Return the capabilities of the Qdrant backend."""
        return StoreCapabilities(
            backend=self.name,
            dense_search=True,
            sparse_search=True,
            multi_vector_search=True,
            exact_search=True,
            filter_ops=frozenset(FIELD_OPS),
            payload_field_types=frozenset(PayloadFieldType),
            index_kinds={
                IndexKind.EXACT: IndexKindCapability(
                    kind=IndexKind.EXACT,
                    supported=True,
                    query_parameters=("exact",),
                    notes="Brute-force scan; the local mode is always exact.",
                ),
                IndexKind.HNSW: IndexKindCapability(
                    kind=IndexKind.HNSW,
                    supported=True,
                    build_parameters=("m", "ef_construct", "full_scan_threshold"),
                    query_parameters=("hnsw_ef",),
                    notes=(
                        "Configured when the collection is created; the local "
                        "mode always answers exactly and ignores the index."
                    ),
                ),
                IndexKind.SPARSE_INVERTED: IndexKindCapability(
                    kind=IndexKind.SPARSE_INVERTED,
                    supported=True,
                    build_parameters=("full_scan_threshold",),
                    notes="Sparse vectors are indexed in inverted form by Qdrant.",
                ),
                IndexKind.PAYLOAD: IndexKindCapability(
                    kind=IndexKind.PAYLOAD,
                    supported=True,
                    build_parameters=("field", "field_type"),
                    notes=(
                        "Only the fields the user declares are indexed; the "
                        "local mode records the index but does not use it to "
                        "speed up filtering."
                    ),
                ),
                IndexKind.MULTI_VECTOR: IndexKindCapability(
                    kind=IndexKind.MULTI_VECTOR,
                    supported=True,
                    notes="Late interaction (max_sim) over the token matrix.",
                ),
            },
            persistence_modes=("memory", "local", "server"),
            delete_by_filter=True,
            delete_by_document=True,
            create_index=True,
            delete_index=True,
            notes=(
                "The in-process local mode performs an exact scan for every "
                "query: search parameters (exact, hnsw_ef) have no effect "
                "there, and payload indexes do not speed up filtering."
            ),
        )

    # ------------------------------------------------------------------
    # collection lifecycle
    # ------------------------------------------------------------------
    def create_collection(
        self, config: CollectionConfig, *, exist_ok: bool = False
    ) -> CollectionInfo:
        """Create a collection.

        The declared configuration is stored in the collection metadata so it
        survives a restart and so that vectors of a foreign model are
        rejected even after the process was restarted.
        """
        self._ensure_open()
        validate_collection_config(config)
        exists = self._collection_exists(config.name)
        if exists and not exist_ok:
            raise CollectionExistsError(
                f"collection {config.name!r} already exists"
            )
        if exists:
            self._check_matching_configuration(config)
        else:
            vectors_config, sparse_config = to_qdrant_vectors_config(
                config.vector_spaces
            )
            try:
                self._client.create_collection(
                    collection_name=config.name,
                    vectors_config=vectors_config or None,
                    sparse_vectors_config=sparse_config or None,
                    metadata={"prome_collection": config.as_dict()},
                )
            except Exception as exc:
                raise self._translate(
                    exc, f"create the collection {config.name!r}"
                ) from exc

        for spec in config.payload_indexes:
            self._create_payload_index(config.name, spec.field, spec.field_type)

        self._collections[config.name] = config
        return self.describe_collection(config.name)

    def _check_matching_configuration(self, config: CollectionConfig) -> None:
        """Make sure an existing collection matches the declared configuration."""
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

    def describe_collection(self, name: str) -> CollectionInfo:
        """Describe an existing collection."""
        self._ensure_open()
        raw = self._backend_collection(name)
        config = self._collection_config(name, raw=raw)
        point_count = None
        try:
            point_count = self._client.count(
                collection_name=name, exact=True
            ).count
        except Exception as exc:  # pragma: no cover - count is best effort
            logger.debug("could not count the points of %s: %s", name, exc)
        return CollectionInfo(
            name=name,
            description=config.description,
            vector_spaces=config.vector_spaces,
            point_count=point_count,
            indexes=tuple(self._describe_indexes(name, raw)),
            backend=self.name,
            metadata=dict(config.metadata),
        )

    def list_collections(self) -> list[CollectionInfo]:
        """List the collections of the backend."""
        self._ensure_open()
        try:
            response = self._client.get_collections()
        except Exception as exc:
            raise self._translate(exc, "list the collections") from exc
        infos: list[CollectionInfo] = []
        for description in response.collections:
            try:
                infos.append(self.describe_collection(description.name))
            except CollectionNotFoundError:  # pragma: no cover - race condition
                continue
            except VectorStoreConfigError:
                # A collection that was not created by this adapter: it is
                # listed, but its vector spaces cannot be described.
                infos.append(
                    CollectionInfo(
                        name=description.name,
                        backend=self.name,
                        metadata={"prome_configured": False},
                    )
                )
        return infos

    def delete_collection(self, name: str) -> bool:
        """Delete a collection. Returns True when a collection was deleted."""
        self._ensure_open()
        if not self._collection_exists(name):
            return False
        try:
            self._client.delete_collection(collection_name=name)
        except Exception as exc:
            raise self._translate(exc, f"delete the collection {name!r}") from exc
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
        points = []
        seen: dict[str, str] = {}
        for record in records:
            validate_record(record, config)
            pid = point_id(record.chunk.id)
            previous = seen.get(pid)
            if previous is not None and previous != record.chunk.id:
                raise InvalidRecordError(
                    f"chunk ids {previous!r} and {record.chunk.id!r} map to the same "
                    "Qdrant point id"
                )
            seen[pid] = record.chunk.id
            points.append(self._build_point(record, config))
        try:
            self._client.upsert(collection_name=collection, points=points)
        except Exception as exc:
            raise self._translate(exc, f"upsert {len(points)} record(s)") from exc
        return len(points)

    def upsert_one(self, collection: str, record: StoredRecord) -> int:
        """Insert or replace a single record."""
        return self.upsert(collection, [record])

    def get(self, collection: str, ids: Sequence[str]) -> list[StoredRecord]:
        """Fetch records by their public chunk ids.

        Ids that do not exist are skipped.
        """
        self._ensure_open()
        ids = list(ids)
        if not ids:
            return []
        self._collection_config(collection)
        point_ids = [point_id(chunk_id) for chunk_id in ids]
        try:
            records = self._client.retrieve(
                collection_name=collection,
                ids=point_ids,
                with_payload=True,
                with_vectors=True,
            )
        except Exception as exc:
            raise self._translate(exc, "fetch records") from exc
        return [
            decode_payload(record.payload or {}, record.vector)
            for record in records
        ]

    def delete(self, collection: str, ids: Sequence[str]) -> int:
        """Delete records by their public chunk ids. Returns the number deleted."""
        self._ensure_open()
        ids = list(ids)
        if not ids:
            return 0
        self._collection_config(collection)
        before = self.count(collection)
        point_ids = [point_id(chunk_id) for chunk_id in ids]
        models_ = models()
        try:
            self._client.delete(
                collection_name=collection,
                points_selector=models_.PointIdsList(points=point_ids),
            )
        except Exception as exc:
            raise self._translate(exc, "delete records") from exc
        return max(before - self.count(collection), 0)

    def delete_by_document(self, collection: str, doc_id: str) -> int:
        """Delete every record of a document. Returns the number deleted."""
        from vector_stores.contracts.filters import eq

        return self.delete_by_filter(collection, eq("doc_id", doc_id))

    def delete_by_filter(self, collection: str, filter: MetadataFilter) -> int:
        """Delete every record matching ``filter``. Returns the number deleted."""
        self._ensure_open()
        self._collection_config(collection)
        models_ = models()
        qdrant_filter = to_qdrant_filter(filter)
        before = self.count(collection)
        try:
            self._client.delete(
                collection_name=collection,
                points_selector=models_.FilterSelector(filter=qdrant_filter),
            )
        except Exception as exc:
            raise self._translate(exc, "delete filtered records") from exc
        return max(before - self.count(collection), 0)

    def count(self, collection: str, filter: MetadataFilter | None = None) -> int:
        """Count the records of a collection, optionally filtered."""
        self._ensure_open()
        self._collection_config(collection)
        qdrant_filter = to_qdrant_filter(filter) if filter is not None else None
        try:
            return self._client.count(
                collection_name=collection,
                count_filter=qdrant_filter,
                exact=True,
            ).count
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
        self._collection_config(collection)
        qdrant_filter = to_qdrant_filter(filter) if filter is not None else None
        try:
            records, next_offset = self._client.scroll(
                collection_name=collection,
                limit=limit,
                offset=offset,
                scroll_filter=qdrant_filter,
                with_payload=True,
                with_vectors=with_vectors,
            )
        except Exception as exc:
            raise self._translate(exc, "scroll the collection") from exc
        return ScrollPage(
            records=[
                decode_payload(record.payload or {}, record.vector) for record in records
            ],
            next_offset=next_offset,
        )

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

        models_ = models()
        kind = request.query_representation()
        if kind is RepresentationType.DENSE:
            query: Any = list(request.dense or [])
        elif kind is RepresentationType.SPARSE:
            query = models_.SparseVector(
                indices=list(request.sparse_indices or []),
                values=list(request.sparse_values or []),
            )
        else:
            query = [list(row) for row in (request.multi_vector or [])]

        qdrant_filter = (
            to_qdrant_filter(request.filter) if request.filter is not None else None
        )
        search_params = to_qdrant_search_params(request.params, space)
        try:
            response = self._client.query_points(
                collection_name=request.collection,
                query=query,
                using=request.space,
                query_filter=qdrant_filter,
                search_params=search_params,
                limit=request.limit,
                offset=request.offset,
                with_payload=self._payload_selector(request.with_payload),
                with_vectors=request.with_vectors,
                score_threshold=request.score_threshold,
            )
        except Exception as exc:
            raise self._translate(exc, "search the collection") from exc

        metric = space.distance
        if metric is None and kind is RepresentationType.SPARSE:
            score_direction = ScoreDirection.HIGHER_IS_BETTER
        else:
            score_direction = metric.score_direction if metric else None
        results: list[SearchResult] = []
        for point in response.points:
            payload = point.payload or {}
            record = decode_payload(payload, point.vector)
            has_contract_payload = CONTRACT_PAYLOAD_KEY in payload
            results.append(
                SearchResult(
                    chunk_id=record.chunk.id,
                    doc_id=record.chunk.doc_id,
                    score=float(point.score),
                    space=request.space,
                    metric=metric,
                    score_direction=score_direction,
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
        """Create an index on an existing collection.

        Only metadata (payload) indexes can be created after the collection
        exists: Qdrant configures its vector indexes when the collection is
        created, so a vector index must be declared in the collection config.
        """
        self._ensure_open()
        self._collection_config(collection)
        if index.kind is not IndexKind.PAYLOAD:
            raise UnsupportedIndexOperationError(
                f"Qdrant configures {index.kind.value} indexes when the collection "
                "is created; recreate the collection with the index configuration "
                "instead of creating the index afterwards"
            )
        field_type = index.field_type or PayloadFieldType.KEYWORD
        self._create_payload_index(collection, index.target, field_type)
        return IndexDescription(
            kind=IndexKind.PAYLOAD,
            target=index.target,
            field_type=field_type,
            build_params=dict(index.build_params),
            backend_options=dict(index.backend_options),
        )

    def delete_index(self, collection: str, name: str) -> bool:
        """Delete a metadata index. Returns True when an index was deleted."""
        self._ensure_open()
        self._collection_config(collection)
        existing = {
            description.target for description in self._describe_indexes(collection)
        }
        if name not in existing:
            return False
        try:
            self._client.delete_payload_index(
                collection_name=collection, field_name=name
            )
        except Exception as exc:
            raise self._translate(exc, f"delete the index {name!r}") from exc
        return True

    def list_indexes(self, collection: str) -> list[IndexDescription]:
        """List the indexes of a collection."""
        self._ensure_open()
        raw = self._backend_collection(collection)
        return self._describe_indexes(collection, raw)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release the backend resources held by this store."""
        if self._closed:
            return
        self._closed = True
        close = getattr(self._client, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("error while closing the Qdrant client: %s", exc)

    def __enter__(self) -> QdrantVectorStore:  # noqa: PYI034 - typing.Self needs 3.11
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _ensure_open(self) -> None:
        if self._closed:
            raise VectorStoreError("this vector store is closed")

    def _collection_exists(self, name: str) -> bool:
        try:
            response = self._client.get_collections()
        except Exception as exc:
            raise self._translate(exc, "list the collections") from exc
        return any(item.name == name for item in response.collections)

    def _backend_collection(self, name: str) -> Any:
        try:
            return self._client.get_collection(collection_name=name)
        except Exception as exc:
            raise self._translate(exc, f"describe the collection {name!r}") from exc

    def _collection_config(
        self, name: str, raw: Any | None = None
    ) -> CollectionConfig:
        """Return the declared configuration of a collection.

        The configuration is read from the collection metadata written at
        creation time, so it is available after a restart.
        """
        if name in self._collections:
            return self._collections[name]
        if raw is None:
            raw = self._backend_collection(name)
        metadata = getattr(getattr(raw, "config", None), "metadata", None) or {}
        stored = metadata.get("prome_collection")
        if not stored:
            raise VectorStoreConfigError(
                f"collection {name!r} carries no Prome configuration: it was not "
                "created by this adapter (or by a version that stored one). "
                "Recreate it to use the vector store contract."
            )
        config = CollectionConfig.from_dict(stored)
        self._collections[name] = config
        return config

    def _build_point(self, record: StoredRecord, config: CollectionConfig) -> Any:
        models_ = models()
        vector_payload: dict[str, Any] = {}
        for name, vectors in record.vectors.items():
            space = config.vector_spaces[name]
            if space.vector_type is RepresentationType.DENSE:
                vector_payload[name] = [float(value) for value in (vectors.dense or [])]
            elif space.vector_type is RepresentationType.SPARSE:
                vector_payload[name] = models_.SparseVector(
                    indices=[int(index) for index in (vectors.sparse_indices or [])],
                    values=[float(value) for value in (vectors.sparse_values or [])],
                )
            else:
                vector_payload[name] = [
                    [float(value) for value in row] for row in (vectors.multi_vector or [])
                ]
        return models_.PointStruct(
            id=point_id(record.chunk.id),
            vector=vector_payload,
            payload=encode_payload(record),
        )

    def _create_payload_index(
        self, collection: str, field: str, field_type: PayloadFieldType
    ) -> None:
        try:
            self._client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=to_qdrant_payload_type(field_type),
            )
        except Exception as exc:
            raise self._translate(
                exc, f"create the payload index {field!r} on {collection!r}"
            ) from exc

    def _describe_indexes(
        self, collection: str, raw: Any | None = None
    ) -> list[IndexDescription]:
        if raw is None:
            raw = self._backend_collection(collection)
        descriptions: list[IndexDescription] = []
        schema = getattr(raw, "payload_schema", None) or {}
        for field, info in schema.items():
            descriptions.append(
                IndexDescription(
                    kind=IndexKind.PAYLOAD,
                    target=field,
                    field_type=from_qdrant_payload_type(info),
                )
            )
        try:
            config = self._collection_config(collection, raw=raw)
        except VectorStoreConfigError:
            config = None
        if config is not None:
            for name, space in config.vector_spaces.items():
                if space.vector_type is RepresentationType.SPARSE:
                    descriptions.append(
                        IndexDescription(
                            kind=IndexKind.SPARSE_INVERTED,
                            target=name,
                            build_params=(
                                space.sparse.as_dict() if space.sparse else {}
                            ),
                        )
                    )
                elif space.vector_type is RepresentationType.MULTI_VECTOR:
                    descriptions.append(
                        IndexDescription(
                            kind=IndexKind.MULTI_VECTOR,
                            target=name,
                            build_params={"comparator": "max_sim"},
                        )
                    )
                else:
                    descriptions.append(
                        IndexDescription(
                            kind=IndexKind.HNSW,
                            target=name,
                            build_params=(
                                space.hnsw.as_dict() if space.hnsw else {}
                            ),
                        )
                    )
        return descriptions

    @staticmethod
    def _payload_selector(with_payload: bool | Sequence[str]) -> Any:
        if with_payload is True:
            return True
        if with_payload is False:
            return ["chunk_id", "doc_id"]
        return list(with_payload)

    @staticmethod
    def _translate(exc: Exception, action: str) -> VectorStoreError:
        """Map a Qdrant exception onto the contract's error types."""
        if isinstance(exc, VectorStoreError):
            return exc
        message = str(exc)
        lowered = message.lower()
        if ("not found" in lowered or "404" in lowered) and "collection" in lowered:
            return CollectionNotFoundError(message)
        if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
            return BackendConnectionError(f"could not {action}: {message}")
        name = type(exc).__name__
        if "Connect" in name or "Timeout" in name or "Rpc" in name:
            return BackendConnectionError(f"could not {action}: {message}")
        if isinstance(exc, ValueError) and "not found" in lowered:
            return CollectionNotFoundError(message)
        return VectorStoreError(f"could not {action}: {message}")
