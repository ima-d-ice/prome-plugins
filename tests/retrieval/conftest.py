"""Shared fixtures for the retrieval test suites.

No model is downloaded and no inference over real weights is run: every
vector comes from the deterministic test double, so the tests are fast and
reproducible.
"""

from __future__ import annotations

from typing import Any

import pytest

from chunkers.contracts.models import BlockReference, Chunk
from embeddings.contracts.models import (
    EmbeddingInput,
    EmbeddingResult,
    Normalization,
    RepresentationType,
)
from parsers.contracts.models import BlockType, ParsedBlock, ParsedDocument

DIMENSIONS = 16


class RecordingEmbeddingModel:
    """A deterministic embedding model that records how it was called.

    The vector of a text is a hashed bag of its tokens, so two texts that
    share tokens get similar vectors and unrelated texts do not.  Every call
    is recorded, which lets the tests assert that the query is embedded
    exactly once.
    """

    name = "test/recording-model"

    def __init__(self, dimension: int = DIMENSIONS) -> None:
        self.dimension = dimension
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    @property
    def capabilities(self):
        from embeddings.contracts.models import EmbeddingCapabilities

        return EmbeddingCapabilities(
            dense=True,
            query_embedding=True,
            document_embedding=True,
            batch_inference=True,
            embedding_dimensions=self.dimension,
        )

    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_id": self.name,
            "adapter_name": "test",
            "dimensions": self.dimension,
            "representation_type": "dense",
            "normalization": "l2",
        }

    def get_embedding_space(self):
        from embeddings.contracts.models import EmbeddingSpaceDescriptor

        return EmbeddingSpaceDescriptor(
            model_id=self.name,
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
            model_id=self.name,
            dimensions=self.dimension,
            dense_vector=self._vector(item.text),
            representation_type=RepresentationType.DENSE,
            normalization=Normalization.L2,
            metadata={"adapter_name": "test"},
            doc_id=item.doc_id,
            chunk_id=item.chunk_id,
        )

    def _vector(self, text: str) -> list[float]:
        import hashlib
        import math

        vector = [0.0] * self.dimension
        for token in (text or "").lower().split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            vector[value % self.dimension] += 1.0 if (value >> 63) & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


@pytest.fixture()
def embedding_model() -> RecordingEmbeddingModel:
    """Return the recording embedding model."""
    return RecordingEmbeddingModel()


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
