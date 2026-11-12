"""Shared fixtures and test doubles for the retrieval test suites.

The doubles are deliberately simple and dependency free:

* :class:`FakeVectorStore` is an in-memory vector store that answers the
  subset of the contract the retrieval layer uses (dense search with
  metadata filters, upsert, get, delete);
* :class:`FakeEmbeddingModel` is a deterministic model whose vector depends
  on the tokens of the text, so two texts that share tokens get similar
  vectors.  Every call is recorded, which lets the tests assert that the
  query is embedded exactly once.

No model is downloaded and no inference over real weights is run.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pytest

from chunkers.contracts.models import BlockReference, Chunk
from embeddings.contracts.models import (
    EmbeddingCapabilities,
    EmbeddingInput,
    EmbeddingResult,
    EmbeddingSpaceDescriptor,
    Normalization,
    RepresentationType,
)
from parsers.contracts.models import BlockType, ParsedBlock, ParsedDocument
from retrieval.contracts.exceptions import UnsupportedRetrievalCapabilityError
from vector_stores.contracts.capabilities import StoreCapabilities
from vector_stores.contracts.exceptions import CollectionNotFoundError
from vector_stores.contracts.filters import FIELD_OPS, FilterOp, MetadataFilter
from vector_stores.contracts.models import (
    CollectionConfig,
    CollectionInfo,
    DistanceMetric,
    ScoreDirection,
    ScrollPage,
    SearchRequest,
    SearchResult,
    SpaceVectors,
    StoredRecord,
    VectorSpaceConfig,
)
from vector_stores.contracts.protocols import VectorStore

DIMENSIONS = 16


class FakeEmbeddingModel:
    """A deterministic embedding model whose vector depends on the tokens.

    The vector of a text is a hashed bag of its tokens, so two texts that
    share tokens get similar vectors and unrelated texts do not.
    """

    name = "test/fake-model"

    def __init__(
        self, dimension: int = DIMENSIONS, model_id: str = "test/fake-model"
    ) -> None:
        self.dimension = dimension
        self.model_id = model_id
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    @property
    def capabilities(self) -> EmbeddingCapabilities:
        return EmbeddingCapabilities(
            dense=True,
            query_embedding=True,
            document_embedding=True,
            batch_inference=True,
            embedding_dimensions=self.dimension,
        )

    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "adapter_name": "test",
            "dimensions": self.dimension,
            "representation_type": "dense",
            "normalization": "l2",
        }

    def get_embedding_space(self) -> EmbeddingSpaceDescriptor:
        return EmbeddingSpaceDescriptor(
            model_id=self.model_id,
            model_revision=None,
            adapter_name="test",
            dimensions=self.dimension,
            representation_type=RepresentationType.DENSE,
            normalization=Normalization.L2,
            supported_languages=None,
        )

    def embed_documents(self, inputs: list[EmbeddingInput]) -> list[EmbeddingResult]:
        self.document_calls.append([item.text for item in inputs])
        return [self._result(item) for item in inputs]

    def embed_query(self, input: EmbeddingInput) -> EmbeddingResult:
        self.query_calls.append(input.text)
        return self._result(input)

    def embed_queries(self, inputs: list[EmbeddingInput]) -> list[EmbeddingResult]:
        self.query_calls.extend(item.text for item in inputs)
        return [self._result(item) for item in inputs]

    @property
    def total_calls(self) -> int:
        """Number of inference calls, whatever their kind."""
        return len(self.document_calls) + len(self.query_calls)

    def _result(self, item: EmbeddingInput) -> EmbeddingResult:
        return EmbeddingResult(
            model_id=self.model_id,
            dimensions=self.dimension,
            dense_vector=self.vector(item.text),
            representation_type=RepresentationType.DENSE,
            normalization=Normalization.L2,
            metadata={"adapter_name": "test"},
            doc_id=item.doc_id,
            chunk_id=item.chunk_id,
        )

    @staticmethod
    def vector(text: str) -> list[float]:
        """Return the deterministic vector of ``text``."""
        import hashlib

        vector = [0.0] * DIMENSIONS
        for token in (text or "").lower().split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            vector[value % DIMENSIONS] += 1.0 if (value >> 63) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]


#: Backwards-compatible alias of the earlier name of the double.
RecordingEmbeddingModel = FakeEmbeddingModel


class FakeVectorStore:
    """An in-memory vector store that supports dense search and filtering."""

    name = "fake"

    def __init__(
        self,
        *,
        filter_ops: frozenset[FilterOp] = frozenset(FIELD_OPS),
        filter_requires_declared_fields: bool = False,
    ) -> None:
        self._records: dict[str, StoredRecord] = {}
        self._collections: dict[str, CollectionConfig] = {}
        self._filter_ops = filter_ops
        self._filter_requires_declared_fields = filter_requires_declared_fields
        self.closed = False

    # -- capabilities --------------------------------------------------
    @property
    def capabilities(self) -> StoreCapabilities:
        return StoreCapabilities(
            backend=self.name,
            dense_search=True,
            exact_search=True,
            filter_ops=self._filter_ops,
            filter_requires_declared_fields=self._filter_requires_declared_fields,
            delete_by_filter=True,
            delete_by_document=True,
        )

    # -- collections ---------------------------------------------------
    def create_collection(
        self, config: CollectionConfig, *, exist_ok: bool = False
    ) -> CollectionInfo:
        if config.name in self._collections and not exist_ok:
            from vector_stores.contracts.exceptions import CollectionExistsError

            raise CollectionExistsError(f"collection {config.name!r} already exists")
        self._collections[config.name] = config
        return self.describe_collection(config.name)

    def describe_collection(self, name: str) -> CollectionInfo:
        if name not in self._collections:
            raise CollectionNotFoundError(f"collection {name!r} does not exist")
        config = self._collections[name]
        return CollectionInfo(
            name=name,
            vector_spaces=dict(config.vector_spaces),
            point_count=len(self._records),
            backend=self.name,
        )

    def list_collections(self) -> list[CollectionInfo]:
        return [self.describe_collection(name) for name in self._collections]

    def delete_collection(self, name: str) -> bool:
        return self._collections.pop(name, None) is not None

    # -- records -------------------------------------------------------
    def upsert(self, collection: str, records: Sequence[StoredRecord]) -> int:
        self._require(collection)
        for record in records:
            self._records[record.chunk.id] = record
        return len(records)

    def upsert_one(self, collection: str, record: StoredRecord) -> int:
        return self.upsert(collection, [record])

    def get(self, collection: str, ids: Sequence[str]) -> list[StoredRecord]:
        return [self._records[i] for i in ids if i in self._records]

    def delete(self, collection: str, ids: Sequence[str]) -> int:
        removed = 0
        for chunk_id in ids:
            if self._records.pop(chunk_id, None) is not None:
                removed += 1
        return removed

    def delete_by_document(self, collection: str, doc_id: str) -> int:
        targets = [k for k, v in self._records.items() if v.chunk.doc_id == doc_id]
        return self.delete(collection, targets)

    def delete_by_filter(self, collection: str, filter: MetadataFilter) -> int:
        targets = [
            k for k, v in self._records.items() if self._matches(filter, v)
        ]
        return self.delete(collection, targets)

    def count(self, collection: str, filter: MetadataFilter | None = None) -> int:
        if filter is None:
            return len(self._records)
        return sum(
            1 for record in self._records.values() if self._matches(filter, record)
        )

    def scroll(
        self,
        collection: str,
        *,
        limit: int = 100,
        offset: int | str | None = None,
        filter: MetadataFilter | None = None,
        with_vectors: bool = False,
    ) -> ScrollPage:
        records = [
            record
            for record in self._records.values()
            if filter is None or self._matches(filter, record)
        ]
        records.sort(key=lambda record: record.chunk.id)
        start = int(offset) if offset is not None else 0
        page = records[start : start + limit]
        next_offset = start + limit if start + limit < len(records) else None
        return ScrollPage(records=page, next_offset=next_offset)

    # -- search --------------------------------------------------------
    def search(self, request: SearchRequest) -> list[SearchResult]:
        self._require(request.collection)
        space = self._collections[request.collection].vector_spaces[request.space]
        query = list(request.dense or [])
        results: list[SearchResult] = []
        for record in self._records.values():
            if request.filter is not None and not self._matches(request.filter, record):
                continue
            vector = record.vectors[request.space].dense or []
            score = _cosine(query, vector)
            results.append(
                SearchResult(
                    chunk_id=record.chunk.id,
                    doc_id=record.chunk.doc_id,
                    score=score,
                    space=request.space,
                    metric=space.distance or DistanceMetric.COSINE,
                    score_direction=ScoreDirection.HIGHER_IS_BETTER,
                    chunk=record.chunk,
                    text=record.chunk.text,
                    metadata=dict(record.metadata),
                    provenance=provenance_of(record),
                )
            )
        results.sort(key=lambda item: (-item.score, item.chunk_id))
        return results[: request.limit]

    def create_index(self, collection, index):
        raise NotImplementedError

    def delete_index(self, collection, name):
        raise NotImplementedError

    def list_indexes(self, collection):
        return ()

    def close(self) -> None:
        self.closed = True

    # -- internals -----------------------------------------------------
    def _require(self, collection: str) -> None:
        if collection not in self._collections:
            raise CollectionNotFoundError(f"collection {collection!r} does not exist")

    def _matches(self, node: MetadataFilter, record: StoredRecord) -> bool:
        """Evaluate a portable filter against a record."""
        op = node.op
        if op is FilterOp.AND:
            return all(self._matches(child, record) for child in node.children)
        if op is FilterOp.OR:
            return any(self._matches(child, record) for child in node.children)
        if op is FilterOp.NOT:
            return not self._matches(node.children[0], record)

        if node.field == "chunk_id":
            value: Any = record.chunk.id
        elif node.field == "doc_id":
            value = record.chunk.doc_id
        else:
            value = record.metadata.get(node.field)

        if op is FilterOp.EQ:
            return value == node.value
        if op is FilterOp.NE:
            return value != node.value
        if op is FilterOp.IN:
            return value in list(node.value)
        if op is FilterOp.NOT_IN:
            return value not in list(node.value)
        if op is FilterOp.GT:
            return value is not None and value > node.value
        if op is FilterOp.GTE:
            return value is not None and value >= node.value
        if op is FilterOp.LT:
            return value is not None and value < node.value
        if op is FilterOp.LTE:
            return value is not None and value <= node.value
        if op is FilterOp.EXISTS:
            return (value is not None) == bool(node.value)
        raise UnsupportedRetrievalCapabilityError(f"unsupported filter {op.value!r}")


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine similarity of two vectors."""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def provenance_of(record: StoredRecord):
    """Return the provenance of a stored record."""
    from vector_stores.contracts.models import ChunkProvenance

    return ChunkProvenance.from_chunk(record.chunk)


@pytest.fixture()
def embedding_model() -> FakeEmbeddingModel:
    """Return the deterministic fake embedding model."""
    return FakeEmbeddingModel()


@pytest.fixture()
def fake_store() -> FakeVectorStore:
    """Return the in-memory fake vector store."""
    return FakeVectorStore()


@pytest.fixture()
def store():
    """Return an in-memory vector store, when the backend is installed."""
    qdrant = pytest.importorskip("qdrant_client")
    from vector_stores.qdrant import QdrantVectorStore

    instance = QdrantVectorStore.memory()
    try:
        yield instance
    finally:
        instance.close()


@pytest.fixture()
def collection_config() -> CollectionConfig:
    """Return a collection configuration for the fake store."""
    space = VectorSpaceConfig(
        name="dense",
        vector_type=RepresentationType.DENSE,
        embedding_space=FakeEmbeddingModel().get_embedding_space(),
        dimensions=DIMENSIONS,
        distance=DistanceMetric.COSINE,
    )
    return CollectionConfig(name="docs", vector_spaces={"dense": space})


def make_document(
    source: str,
    sections: list[tuple[str, str]],
    *,
    heading_level: int = 1,
) -> ParsedDocument:
    """Build a parsed document from ``(heading, body)`` pairs."""
    blocks: list[ParsedBlock] = []
    for heading, body in sections:
        blocks.append(
            ParsedBlock(
                BlockType.HEADING,
                heading,
                metadata={"heading_level": heading_level},
            )
        )
        blocks.append(ParsedBlock(BlockType.TEXT, body))
    return ParsedDocument(
        source=source,
        parser_name="test",
        parser_version="1.0",
        blocks=blocks,
    )


def make_chunk(
    chunk_id: str,
    doc_id: str,
    text: str,
    *,
    index: int = 0,
    count: int = 1,
    metadata: dict[str, Any] | None = None,
    heading: str | None = None,
) -> Chunk:
    """Build a chunk for the tests."""
    return Chunk(
        id=chunk_id,
        doc_id=doc_id,
        text=text,
        index=index,
        count=count,
        length=len(text),
        length_unit="characters",
        block_refs=[BlockReference(block_index=0, page_number=1, block_type="text")],
        heading=heading,
        metadata={"source": doc_id, "pages": [1], **(metadata or {})},
    )


def make_record(
    chunk: Chunk,
    model: FakeEmbeddingModel,
    *,
    metadata: dict[str, Any] | None = None,
) -> StoredRecord:
    """Build a stored record of ``chunk``."""
    space = "dense"
    return StoredRecord(
        chunk=chunk,
        vectors={
            space: SpaceVectors(
                space=space,
                embedding_space=model.get_embedding_space(),
                dense=model.vector(chunk.text),
            )
        },
        metadata=dict(metadata or {}),
    )
