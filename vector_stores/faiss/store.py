"""FAISS adapter for the Prome vector-store contracts.

FAISS is a similarity search library, not a document database.  The adapter
therefore combines two stores:

- **FAISS** holds the dense vectors and answers the nearest neighbour
  queries (``IndexFlat`` for exact search, ``IndexHNSWFlat`` and
  ``IndexIVFFlat`` for approximate search);
- **SQLite** holds the records: the public chunk id, the document id, the
  text, the metadata, the provenance, the embedding-space descriptor and the
  collection configuration.

SQLite is the authoritative store: the FAISS index is a derived artifact and
is rebuilt automatically when the two disagree (missing, stale or corrupt
index file).  The rebuild is also available explicitly through
:meth:`FaissVectorStore.rebuild_index`.

Only dense vectors are supported.  Sparse and multi-vector representations
are rejected instead of being densified or averaged.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embeddings.contracts.models import RepresentationType
from vector_stores.contracts.capabilities import StoreCapabilities
from vector_stores.contracts.exceptions import (
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
from vector_stores.faiss._client import import_faiss, numpy
from vector_stores.faiss.mapping import (
    DATABASE_NAME,
    DOC_ID_COLUMN,
    INDEX_PREFIX,
    INDEX_SUFFIX,
    contract_payload,
    decode_vector,
    encode_vector,
    is_cosine,
    metadata_json,
)
from vector_stores.faiss.sqlite_store import SQLiteRecordStore

logger = logging.getLogger(__name__)

#: The index kinds the adapter can build.
INDEX_TYPES = ("flat", "hnsw", "ivf")


@dataclass(frozen=True)
class FaissStoreConfig:
    """Settings of the FAISS adapter."""

    path: str | None = None
    """Directory holding the collections.  ``None`` selects the ephemeral mode."""

    index_type: str = "flat"
    """Default index kind: ``flat`` (exact), ``hnsw`` or ``ivf``."""

    hnsw_m: int = 16
    hnsw_ef_construction: int = 100
    hnsw_ef_search: int = 64
    ivf_nlist: int = 64

    @classmethod
    def local(cls, path: str, **kwargs: Any) -> FaissStoreConfig:
        """Return the configuration of a persistent local directory."""
        return cls(path=path, **kwargs)

    @classmethod
    def memory(cls, **kwargs: Any) -> FaissStoreConfig:
        """Return the configuration of the ephemeral mode."""
        return cls(path=None, **kwargs)


class _Collection:
    """The runtime state of one collection (SQLite store + FAISS index)."""

    def __init__(self, name: str, directory: Path, config: CollectionConfig) -> None:
        self.name = name
        self.directory = directory
        self.config = config
        self.store = SQLiteRecordStore(directory / DATABASE_NAME)
        self.index: Any | None = None
        self.index_type = "flat"
        self.dimensions = 0
        self.metric = DistanceMetric.COSINE


class FaissVectorStore:
    """Vector store backed by FAISS (vectors) and SQLite (records)."""

    name = "faiss"

    def __init__(
        self,
        config: FaissStoreConfig | None = None,
        *,
        root: str | Path | None = None,
    ) -> None:
        """Create a store.

        Args:
            config: Settings of the adapter.
            root: Directory holding the collections; overrides ``config.path``.
        """
        self._config = config or FaissStoreConfig()
        # Import FAISS right away: the adapter needs it, and the import is
        # where the "torch and FAISS cannot share a process" guard fires.
        import_faiss()
        path = root if root is not None else self._config.path
        self._temporary_path: str | None = None
        if path is None:
            import tempfile

            self._temporary_path = tempfile.mkdtemp(prefix="prome-faiss-")
            path = self._temporary_path
        self._root = Path(path)
        self._root.mkdir(parents=True, exist_ok=True)
        self._collections: dict[str, _Collection] = {}
        self._closed = False

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def local(cls, path: str, **kwargs: Any) -> FaissVectorStore:
        """Create a store using a persistent local directory."""
        return cls(FaissStoreConfig.local(path, **kwargs))

    @classmethod
    def memory(cls, **kwargs: Any) -> FaissVectorStore:
        """Create a store using an ephemeral temporary directory."""
        return cls(FaissStoreConfig.memory(**kwargs))

    # ------------------------------------------------------------------
    # capabilities
    # ------------------------------------------------------------------
    @property
    def capabilities(self) -> StoreCapabilities:
        """Return the capabilities of the FAISS backend."""
        return StoreCapabilities(
            backend=self.name,
            dense_search=True,
            sparse_search=False,
            multi_vector_search=False,
            exact_search=True,
            filter_ops=frozenset(FIELD_OPS),
            payload_field_types=frozenset(PayloadFieldType),
            index_kinds={
                IndexKind.EXACT: IndexKindCapability(
                    kind=IndexKind.EXACT,
                    supported=True,
                    query_parameters=("exact",),
                    notes="IndexFlat: a brute-force scan, exact results.",
                ),
                IndexKind.HNSW: IndexKindCapability(
                    kind=IndexKind.HNSW,
                    supported=True,
                    build_parameters=("m", "ef_construct"),
                    query_parameters=("hnsw_ef",),
                    notes=(
                        "IndexHNSWFlat. FAISS has no stable ids after a "
                        "removal: the index is rebuilt from SQLite instead."
                    ),
                ),
                IndexKind.SPARSE_INVERTED: IndexKindCapability(
                    kind=IndexKind.SPARSE_INVERTED,
                    supported=False,
                    notes="FAISS only indexes dense vectors.",
                ),
                IndexKind.PAYLOAD: IndexKindCapability(
                    kind=IndexKind.PAYLOAD,
                    supported=True,
                    build_parameters=("field", "field_type"),
                    notes="SQLite expression index on a declared metadata field.",
                ),
                IndexKind.MULTI_VECTOR: IndexKindCapability(
                    kind=IndexKind.MULTI_VECTOR,
                    supported=False,
                    notes=(
                        "FAISS has no late interaction search; multi-vectors "
                        "are rejected instead of being averaged."
                    ),
                ),
            },
            persistence_modes=("memory", "local"),
            delete_by_filter=True,
            delete_by_document=True,
            create_index=True,
            delete_index=True,
            notes=(
                "FAISS is a similarity search library: the records live in "
                "SQLite, which is the authoritative store. Sparse and "
                "multi-vector spaces are not supported."
            ),
        )

    # ------------------------------------------------------------------
    # collection lifecycle
    # ------------------------------------------------------------------
    def create_collection(
        self, config: CollectionConfig, *, exist_ok: bool = False
    ) -> CollectionInfo:
        """Create a collection (a directory with an index and a database)."""
        self._ensure_open()
        validate_collection_config(config)
        for space in config.vector_spaces.values():
            if space.vector_type is not RepresentationType.DENSE:
                raise UnsupportedVectorTypeError(
                    f"the FAISS adapter only supports dense vector spaces; "
                    f"{space.name!r} is {space.vector_type.value}"
                )
        directory = self._root / config.name
        if directory.exists():
            if not exist_ok:
                raise CollectionExistsError(
                    f"collection {config.name!r} already exists"
                )
            existing = self._load_collection(config.name)
            if existing is not None and existing.config != config:
                raise VectorStoreConfigError(
                    f"collection {config.name!r} already exists with a different "
                    "configuration; drop it first or use a different name"
                )
        else:
            directory.mkdir(parents=True, exist_ok=True)

        collection = self._open_collection(config.name, directory, config)
        collection.store.set_config(config)
        collection.index_type = self._config.index_type
        self._rebuild_index(collection)
        for spec in config.payload_indexes:
            collection.store.create_payload_index(spec.field)
        self._collections[config.name] = collection
        return self.describe_collection(config.name)

    def describe_collection(self, name: str) -> CollectionInfo:
        """Describe an existing collection."""
        self._ensure_open()
        collection = self._open_collection(name)
        return CollectionInfo(
            name=name,
            description=collection.config.description,
            vector_spaces=collection.config.vector_spaces,
            point_count=collection.store.count(),
            indexes=tuple(self._describe_indexes(collection)),
            backend=self.name,
            metadata=dict(collection.config.metadata),
        )

    def list_collections(self) -> list[CollectionInfo]:
        """List the collections of the backend."""
        self._ensure_open()
        infos: list[CollectionInfo] = []
        for directory in sorted(self._root.iterdir()):
            if not directory.is_dir():
                continue
            if not (directory / DATABASE_NAME).exists():
                continue
            try:
                infos.append(self.describe_collection(directory.name))
            except (CollectionNotFoundError, VectorStoreConfigError):
                continue
        return infos

    def delete_collection(self, name: str) -> bool:
        """Delete a collection. Returns True when a collection was deleted."""
        self._ensure_open()
        directory = self._root / name
        if not (directory / DATABASE_NAME).exists():
            return False
        collection = self._collections.pop(name, None)
        if collection is not None:
            collection.store.close()
        shutil.rmtree(directory, ignore_errors=True)
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
        state = self._open_collection(collection)
        config = state.config
        space = self._default_space(config)
        written = 0
        for record in records:
            validate_record(record, config)
            vectors = record.vectors.get(space.name)
            if vectors is None or vectors.dense is None:
                raise InvalidRecordError(
                    f"record for chunk {record.chunk.id!r} carries no dense vector "
                    f"for the vector space {space.name!r}"
                )
            chunk_id = record.chunk.id
            doc_id = record.chunk.doc_id
            faiss_id = self._next_faiss_id(state)
            payload = contract_payload(
                chunk_id,
                doc_id,
                encode_contract_payload(record),
            )
            state.store.upsert_record(
                chunk_id=chunk_id,
                doc_id=doc_id,
                faiss_id=faiss_id,
                payload=payload,
                metadata=metadata_json(record.metadata),
                vector=encode_vector(vectors.dense, space),
            )
            # The vector is appended to the index; the previous vector of a
            # replaced record stays in the index but its id is not live any
            # more, so the search filters it out.
            self._append_vector(state, chunk_id, vectors.dense, space)
            written += 1
        self._save_index(state)
        return written

    def upsert_one(self, collection: str, record: StoredRecord) -> int:
        """Insert or replace a single record."""
        return self.upsert(collection, [record])

    def get(self, collection: str, ids: Sequence[str]) -> list[StoredRecord]:
        """Fetch records by their public chunk ids."""
        self._ensure_open()
        ids = list(ids)
        if not ids:
            return []
        state = self._open_collection(collection)
        rows = state.store.get_records(ids)
        return [self._row_to_record(state, row) for row in rows]

    def delete(self, collection: str, ids: Sequence[str]) -> int:
        """Delete records by their public chunk ids."""
        self._ensure_open()
        ids = list(ids)
        if not ids:
            return 0
        state = self._open_collection(collection)
        return state.store.delete_records(ids)

    def delete_by_document(self, collection: str, doc_id: str) -> int:
        """Delete every record of a document."""
        from vector_stores.contracts.filters import eq

        return self.delete_by_filter(collection, eq(DOC_ID_COLUMN, doc_id))

    def delete_by_filter(self, collection: str, filter: MetadataFilter) -> int:
        """Delete every record matching ``filter``."""
        self._ensure_open()
        state = self._open_collection(collection)
        return state.store.delete_by_filter(filter)

    def count(self, collection: str, filter: MetadataFilter | None = None) -> int:
        """Count the records of a collection, optionally filtered."""
        self._ensure_open()
        state = self._open_collection(collection)
        return state.store.count(filter)

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
        state = self._open_collection(collection)
        start = int(offset) if offset is not None else 0
        rows = state.store.scroll(limit=limit, offset=start, node=filter)
        records = [self._row_to_record(state, row) for row in rows]
        next_offset = start + len(rows) if len(rows) == limit else None
        return ScrollPage(records=records, next_offset=next_offset)

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------
    def search(self, request: SearchRequest) -> list[SearchResult]:
        """Run a vector search and return the hits, best first."""
        self._ensure_open()
        state = self._open_collection(request.collection)
        config = state.config
        space = config.vector_spaces.get(request.space)
        if space is None:
            raise VectorStoreConfigError(
                f"collection {request.collection!r} has no vector space "
                f"{request.space!r}; known spaces: {sorted(config.vector_spaces)}"
            )
        validate_search_request(request, space)
        if space.vector_type is not RepresentationType.DENSE:
            raise UnsupportedVectorTypeError(
                f"the FAISS adapter only supports dense vector spaces; "
                f"{space.name!r} is {space.vector_type.value}"
            )
        query = [float(value) for value in (request.dense or [])]
        if is_cosine(space):
            query = self._normalize(query)

        if request.filter is not None:
            hits = self._filtered_search(state, space, query, request)
        elif request.params.get("exact"):
            hits = self._exact_search(state, space, query, request)
        else:
            hits = self._approximate_search(state, space, query, request)
        return hits

    def _exact_search(
        self, state: _Collection, space: VectorSpaceConfig, query: list[float], request: SearchRequest
    ) -> list[SearchResult]:
        """Exact search over the live records (brute force)."""
        np = numpy()
        rows = list(state.store.iter_records())
        if not rows:
            return []
        vectors = np.vstack(
            [np.frombuffer(row["vector"], dtype="float32") for row in rows]
        )
        scores = self._scores(vectors, np.asarray(query, dtype="float32"), space)
        order = np.argsort(-scores if self._higher_is_better(space) else scores)
        return self._build_results(state, rows, order, scores, space, request)

    def _filtered_search(
        self,
        state: _Collection,
        space: VectorSpaceConfig,
        query: list[float],
        request: SearchRequest,
    ) -> list[SearchResult]:
        """Exact search restricted to the records matching the filter.

        The filter is evaluated by SQLite (the authoritative store) and the
        scores are computed over exactly that subset: the result is the true
        top-k of the matching records, never a post-filtered approximation.
        """
        np = numpy()
        rows = list(state.store.iter_records(request.filter))
        if not rows:
            return []
        vectors = np.vstack(
            [np.frombuffer(row["vector"], dtype="float32") for row in rows]
        )
        scores = self._scores(vectors, np.asarray(query, dtype="float32"), space)
        order = np.argsort(-scores if self._higher_is_better(space) else scores)
        return self._build_results(state, rows, order, scores, space, request)

    def _approximate_search(
        self,
        state: _Collection,
        space: VectorSpaceConfig,
        query: list[float],
        request: SearchRequest,
    ) -> list[SearchResult]:
        """Search the FAISS index (approximate for HNSW and IVF)."""
        np = numpy()
        if state.index is None:
            return self._exact_search(state, space, query, request)
        self._apply_query_parameters(state, request)
        live = state.store.live_faiss_ids()
        live_set = set(live)
        dead = max(int(state.index.ntotal) - len(live), 0)
        wanted = min(int(state.index.ntotal), request.limit + dead)
        if wanted <= 0:
            return []
        scores, ids = state.index.search(
            np.asarray([query], dtype="float32"), wanted
        )
        rows = []
        result_scores = []
        for score, faiss_id in zip(scores[0], ids[0]):
            if faiss_id < 0 or int(faiss_id) not in live_set:
                continue
            row = state.store.row_by_faiss_id(int(faiss_id))
            if row is None:
                continue
            rows.append(row)
            result_scores.append(float(score))
            if len(rows) >= request.limit:
                break
        return self._results_from_rows(state, rows, result_scores, space, request)

    def _apply_query_parameters(
        self, state: _Collection, request: SearchRequest
    ) -> None:
        params = request.params
        if state.index_type == "hnsw" and params.get("hnsw_ef") is not None:
            state.index.hnsw.efSearch = int(params["hnsw_ef"])
        if state.index_type == "ivf" and params.get("nprobe") is not None:
            state.index.nprobe = int(params["nprobe"])

    def _scores(self, vectors: Any, query: Any, space: VectorSpaceConfig) -> Any:
        """Return the scores of a query against a matrix of vectors."""
        if space.distance is DistanceMetric.EUCLIDEAN:
            difference = vectors - query
            return (difference * difference).sum(axis=1)
        return vectors @ query

    def _higher_is_better(self, space: VectorSpaceConfig) -> bool:
        return space.distance in (None, DistanceMetric.COSINE, DistanceMetric.DOT)

    def _build_results(
        self,
        state: _Collection,
        rows: list[Any],
        order: Any,
        scores: Any,
        space: VectorSpaceConfig,
        request: SearchRequest,
    ) -> list[SearchResult]:
        selected = []
        for position in order[: request.limit]:
            selected.append((rows[int(position)], float(scores[int(position)])))
        return self._results_from_rows(state, [row for row, _ in selected], [s for _, s in selected], space, request)

    def _results_from_rows(
        self,
        state: _Collection,
        rows: list[Any],
        scores: list[float],
        space: VectorSpaceConfig,
        request: SearchRequest,
    ) -> list[SearchResult]:
        results = []
        for row, score in zip(rows, scores):
            record = self._row_to_record(state, row)
            metric = space.distance or DistanceMetric.COSINE
            results.append(
                SearchResult(
                    chunk_id=record.chunk.id,
                    doc_id=record.chunk.doc_id,
                    score=self._canonical_score(score, metric),
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

    def _canonical_score(self, score: float, metric: DistanceMetric) -> float:
        """Convert a FAISS score into the canonical score of the metric."""
        if metric is DistanceMetric.EUCLIDEAN:
            return score**0.5
        return score

    def _normalize(self, vector: list[float]) -> list[float]:
        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0.0:
            return list(vector)
        return [value / norm for value in vector]

    # ------------------------------------------------------------------
    # indexes
    # ------------------------------------------------------------------
    def create_index(self, collection: str, index: IndexConfig) -> IndexDescription:
        """Create (or rebuild) an index over the records of a collection."""
        self._ensure_open()
        state = self._open_collection(collection)
        if index.kind is IndexKind.PAYLOAD:
            field_type = index.field_type or PayloadFieldType.KEYWORD
            state.store.create_payload_index(index.target)
            return IndexDescription(
                kind=IndexKind.PAYLOAD,
                target=index.target,
                field_type=field_type,
                build_params=dict(index.build_params),
            )
        if index.kind is IndexKind.SPARSE_INVERTED:
            raise UnsupportedIndexOperationError(
                "FAISS only indexes dense vectors"
            )
        if index.kind is IndexKind.MULTI_VECTOR:
            raise UnsupportedIndexOperationError(
                "FAISS has no late interaction search; multi-vectors are rejected "
                "instead of being averaged"
            )
        space = state.config.vector_spaces.get(index.target)
        if space is None:
            raise VectorStoreConfigError(
                f"collection {collection!r} has no vector space {index.target!r}"
            )
        default_type = "hnsw" if index.kind is IndexKind.HNSW else "flat"
        index_type = str(index.build_params.get("index_type", default_type))
        self._build_index(state, index_type, index.build_params)
        self._rebuild_index(state)
        self._save_index(state)
        return IndexDescription(
            kind=IndexKind.HNSW if index_type == "hnsw" else IndexKind.EXACT,
            target=space.name,
            build_params={**dict(index.build_params), "index_type": index_type},
        )

    def delete_index(self, collection: str, name: str) -> bool:
        """Delete an index by name. Returns True when an index was deleted."""
        self._ensure_open()
        state = self._open_collection(collection)
        for description in self._describe_indexes(state):
            if description.index_name != name:
                continue
            if description.kind is IndexKind.PAYLOAD:
                return state.store.drop_payload_index(description.target)
            raise UnsupportedIndexOperationError(
                "the vector index of a collection is rebuilt, not dropped; use "
                "rebuild_index() to change its kind"
            )
        return False

    def list_indexes(self, collection: str) -> list[IndexDescription]:
        """List the indexes of a collection."""
        self._ensure_open()
        state = self._open_collection(collection)
        return self._describe_indexes(state)

    def rebuild_index(self, collection: str, index_type: str | None = None) -> IndexDescription:
        """Rebuild the vector index of a collection from SQLite.

        The index is always rebuilt from the authoritative records, so this
        operation also repairs an index that is stale or out of sync.
        """
        self._ensure_open()
        state = self._open_collection(collection)
        kind = index_type or state.index_type
        state.index_type = kind
        self._build_index(state, kind)
        self._rebuild_index(state)
        self._save_index(state)
        return IndexDescription(
            kind=IndexKind.HNSW if kind == "hnsw" else IndexKind.EXACT,
            target=self._default_space(state.config).name,
            build_params={"index_type": kind},
        )

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release the backend resources held by this store."""
        if self._closed:
            return
        self._closed = True
        for collection in self._collections.values():
            try:
                self._save_index(collection)
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("could not save the index of %s: %s", collection.name, exc)
            collection.store.close()
        self._collections.clear()
        if self._temporary_path:
            shutil.rmtree(self._temporary_path, ignore_errors=True)
            self._temporary_path = None

    def __enter__(self) -> FaissVectorStore:  # noqa: PYI034 - typing.Self needs 3.11
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _ensure_open(self) -> None:
        if self._closed:
            raise VectorStoreError("this vector store is closed")

    def _default_space(self, config: CollectionConfig) -> VectorSpaceConfig:
        for space in config.vector_spaces.values():
            return space
        raise VectorStoreConfigError(  # pragma: no cover - validated earlier
            f"collection {config.name!r} declares no vector space"
        )

    def _open_collection(
        self,
        name: str,
        directory: Path | None = None,
        config: CollectionConfig | None = None,
    ) -> _Collection:
        if name in self._collections:
            return self._collections[name]
        directory = directory or (self._root / name)
        database = directory / DATABASE_NAME
        if not database.exists():
            if config is None:
                raise CollectionNotFoundError(f"collection {name!r} does not exist")
            directory.mkdir(parents=True, exist_ok=True)
        store = SQLiteRecordStore(database)
        stored_config = store.get_config()
        if stored_config is None and config is None:
            store.close()
            raise VectorStoreConfigError(
                f"collection {name!r} carries no Prome configuration: it was not "
                "created by this adapter. Recreate it to use the vector store "
                "contract."
            )
        state = _Collection(name, directory, config or stored_config)
        state.store = store
        self._collections[name] = state
        self._load_index(state)
        return state

    def _load_collection(self, name: str) -> _Collection | None:
        try:
            return self._open_collection(name)
        except (CollectionNotFoundError, VectorStoreConfigError):
            return None

    def _index_path(self, state: _Collection, space: VectorSpaceConfig) -> Path:
        return state.directory / f"{INDEX_PREFIX}{space.name}{INDEX_SUFFIX}"

    def _build_index(
        self,
        state: _Collection,
        index_type: str,
        build_params: dict[str, Any] | None = None,
    ) -> None:
        """Build the FAISS index of a collection from the stored vectors."""
        faiss = import_faiss()
        build_params = dict(build_params or {})
        space = self._default_space(state.config)
        dimensions = space.dimensions or 0
        metric = space.distance or DistanceMetric.COSINE
        use_ip = metric is not DistanceMetric.EUCLIDEAN
        faiss_metric = faiss.METRIC_INNER_PRODUCT if use_ip else faiss.METRIC_L2

        if index_type == "flat":
            index = (
                faiss.IndexFlatIP(dimensions)
                if use_ip
                else faiss.IndexFlatL2(dimensions)
            )
        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(
                dimensions,
                int(build_params.get("m", self._config.hnsw_m)),
                faiss_metric,
            )
            index.hnsw.efConstruction = int(
                build_params.get("ef_construct", self._config.hnsw_ef_construction)
            )
            index.hnsw.efSearch = int(
                build_params.get("ef_search", self._config.hnsw_ef_search)
            )
        elif index_type == "ivf":
            nlist = int(build_params.get("nlist", self._config.ivf_nlist))
            rows = list(state.store.iter_records())
            if len(rows) < nlist:
                raise VectorStoreConfigError(
                    f"an IVF index needs at least {nlist} vectors to train, "
                    f"the collection holds {len(rows)}"
                )
            quantizer = (
                faiss.IndexFlatIP(dimensions)
                if use_ip
                else faiss.IndexFlatL2(dimensions)
            )
            index = faiss.IndexIVFFlat(quantizer, dimensions, nlist, faiss_metric)
        else:
            raise UnsupportedIndexOperationError(
                f"unknown FAISS index type {index_type!r} "
                f"(supported: {', '.join(INDEX_TYPES)})"
            )

        state.index = index
        state.index_type = index_type
        state.dimensions = dimensions
        state.metric = metric

    def _rebuild_index(self, state: _Collection) -> None:
        """Rebuild the index from the records and renumber the FAISS ids.

        FAISS renumbers the ids when a vector is removed, so the adapter
        never removes a vector from the index: it rebuilds it instead.  The
        rebuild is also the recovery path for a missing or corrupt index.
        """
        faiss = import_faiss()
        np = numpy()
        self._build_index(state, state.index_type)
        rows = sorted(
            state.store.iter_records(), key=lambda row: int(row["faiss_id"])
        )
        if rows:
            vectors = np.vstack(
                [np.frombuffer(row["vector"], dtype="float32") for row in rows]
            )
            if state.index_type == "ivf":
                state.index.train(vectors)
            state.index.add(vectors)
        for position, row in enumerate(rows):
            if int(row["faiss_id"]) != position:
                state.store.set_faiss_id(str(row["chunk_id"]), position)
        _ = faiss

    def _next_faiss_id(self, state: _Collection) -> int:
        """Return the next free slot of the index.

        The FAISS id of a record is the slot its vector occupies in the
        index, so the ids stay valid as long as the index is not rebuilt.
        """
        if state.index is None:
            return 0
        return int(state.index.ntotal)

    def _append_vector(
        self,
        state: _Collection,
        chunk_id: str,
        vector: list[float],
        space: VectorSpaceConfig,
    ) -> None:
        """Add the vector of a record to the index, keeping the ids in sync."""
        np = numpy()
        if state.index is None:
            self._rebuild_index(state)
            return
        rows = state.store.get_records([chunk_id])
        if not rows:  # pragma: no cover - defensive
            return
        faiss_id = int(rows[0]["faiss_id"])
        if faiss_id != int(state.index.ntotal):
            # The index and the store disagree: rebuild the index, which also
            # renumbers the records.
            self._rebuild_index(state)
            return
        values = np.asarray([vector], dtype="float32")
        if is_cosine(space):
            norm = float(np.linalg.norm(values))
            if norm > 0:
                values = values / norm
        state.index.add(values)

    def _save_index(self, state: _Collection) -> None:
        faiss = import_faiss()
        if state.index is None:
            return
        space = self._default_space(state.config)
        path = self._index_path(state, space)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(state.index, str(path))

    def _load_index(self, state: _Collection) -> None:
        """Load the index of a collection, rebuilding it when it is missing."""
        faiss = import_faiss()
        space = self._default_space(state.config)
        path = self._index_path(state, space)
        state.index_type = self._config.index_type
        if path.exists():
            try:
                state.index = faiss.read_index(str(path))
                state.dimensions = state.index.d
                state.index_type = self._detect_index_type(state.index)
                self._check_index_sync(state)
                return
            except Exception as exc:
                logger.warning(
                    "could not read the FAISS index of %s (%s); rebuilding it",
                    state.name,
                    exc,
                )
        state.index_type = self._config.index_type
        self._build_index(state, state.index_type)
        self._rebuild_index(state)
        self._save_index(state)

    def _check_index_sync(self, state: _Collection) -> None:
        """Rebuild the index when it does not match the stored records."""
        live = len(state.store.live_faiss_ids())
        if int(state.index.ntotal) < live:
            logger.warning(
                "the FAISS index of %s holds fewer vectors than the records; "
                "rebuilding it",
                state.name,
            )
            self._build_index(state, state.index_type)
            self._rebuild_index(state)
            self._save_index(state)

    @staticmethod
    def _detect_index_type(index: Any) -> str:
        faiss = import_faiss()
        if isinstance(index, faiss.IndexHNSWFlat):
            return "hnsw"
        if isinstance(index, faiss.IndexIVFFlat):
            return "ivf"
        return "flat"

    def _describe_indexes(self, state: _Collection) -> list[IndexDescription]:
        """Describe the indexes of a collection."""
        descriptions: list[IndexDescription] = []
        space = self._default_space(state.config)
        kind = {
            "hnsw": IndexKind.HNSW,
            "ivf": IndexKind.IVF,
        }.get(state.index_type, IndexKind.EXACT)
        descriptions.append(
            IndexDescription(
                kind=kind,
                target=space.name,
                build_params={
                    "index_type": state.index_type,
                    "vectors": int(state.index.ntotal) if state.index is not None else 0,
                },
            )
        )
        for field in state.store.list_payload_indexes():
            descriptions.append(
                IndexDescription(kind=IndexKind.PAYLOAD, target=field)
            )
        return descriptions

    def _row_to_record(self, state: _Collection, row: Any) -> StoredRecord:
        payload = state.store.row_payload(row)
        record = decode_contract_payload(payload)
        record.metadata = state.store.row_metadata(row)
        space = self._default_space(state.config)
        record.vectors = {
            space.name: SpaceVectors(
                space=space.name,
                embedding_space=space.embedding_space,
                dense=decode_vector(row["vector"], space),
            )
        }
        return record
