"""Shared fixtures for the Qdrant adapter tests.

No embedding model is downloaded and no inference is run: every vector is
produced by a deterministic generator.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from chunkers.contracts.models import BlockReference, Chunk
from embeddings.contracts.models import (
    EmbeddingSpaceDescriptor,
    Normalization,
    RepresentationType,
)
from vector_stores.contracts import (
    CollectionConfig,
    DistanceMetric,
    HnswIndexConfig,
    PayloadIndexSpec,
    SpaceVectors,
    StoredRecord,
    VectorSpaceConfig,
)

MODEL = "test/embedding-model"
REVISION = "rev-1"
DIMENSIONS = 8


def unit_vector(seed: int, dimensions: int = DIMENSIONS) -> list[float]:
    """Return a deterministic unit vector for ``seed``."""
    raw = [math.sin(seed * (index + 1) * 0.7391) + 0.1 * (index + 1) for index in range(dimensions)]
    norm = math.sqrt(sum(value * value for value in raw)) or 1.0
    return [value / norm for value in raw]


def sparse_vector(seed: int) -> tuple[list[int], list[float]]:
    """Return a deterministic sparse vector for ``seed``."""
    indices = sorted({(seed * 7 + offset * 13) % 64 for offset in range(4)})
    values = [0.5 - 0.05 * ((seed + index) % 5) for index in range(len(indices))]
    return indices, values


def multi_vector(seed: int, rows: int = 3, dimensions: int = DIMENSIONS) -> list[list[float]]:
    """Return a deterministic multi-vector matrix for ``seed``."""
    return [unit_vector(seed * 100 + row, dimensions) for row in range(rows)]


def descriptor(
    *,
    model_id: str = MODEL,
    model_revision: str | None = REVISION,
    dimensions: int | None = DIMENSIONS,
    representation_type: RepresentationType = RepresentationType.DENSE,
    normalization: Normalization = Normalization.L2,
) -> EmbeddingSpaceDescriptor:
    return EmbeddingSpaceDescriptor(
        model_id=model_id,
        model_revision=model_revision,
        dimensions=dimensions,
        representation_type=representation_type,
        normalization=normalization,
    )


def dense_space(
    name: str = "dense",
    *,
    dimensions: int = DIMENSIONS,
    distance: DistanceMetric = DistanceMetric.COSINE,
    model_id: str = MODEL,
    model_revision: str | None = REVISION,
    hnsw: HnswIndexConfig | None = None,
) -> VectorSpaceConfig:
    return VectorSpaceConfig(
        name=name,
        vector_type=RepresentationType.DENSE,
        embedding_space=descriptor(
            model_id=model_id,
            model_revision=model_revision,
            dimensions=dimensions,
        ),
        dimensions=dimensions,
        distance=distance,
        hnsw=hnsw,
    )


def sparse_space(name: str = "sparse") -> VectorSpaceConfig:
    return VectorSpaceConfig(
        name=name,
        vector_type=RepresentationType.SPARSE,
        embedding_space=descriptor(
            dimensions=None, representation_type=RepresentationType.SPARSE
        ),
    )


def multi_space(name: str = "late") -> VectorSpaceConfig:
    return VectorSpaceConfig(
        name=name,
        vector_type=RepresentationType.MULTI_VECTOR,
        embedding_space=descriptor(
            representation_type=RepresentationType.MULTI_VECTOR
        ),
        dimensions=DIMENSIONS,
        distance=DistanceMetric.COSINE,
    )


def dense_config(
    name: str = "docs",
    *,
    dimensions: int = DIMENSIONS,
    payload_indexes: tuple[PayloadIndexSpec, ...] = (),
    hnsw: HnswIndexConfig | None = None,
    model_id: str = MODEL,
) -> CollectionConfig:
    return CollectionConfig(
        name=name,
        vector_spaces={
            "dense": dense_space(
                dimensions=dimensions, hnsw=hnsw, model_id=model_id
            )
        },
        description="test collection",
        payload_indexes=payload_indexes,
    )


def multi_space_config(name: str = "docs") -> CollectionConfig:
    return CollectionConfig(
        name=name,
        vector_spaces={"dense": dense_space(), "sparse": sparse_space(), "late": multi_space()},
    )


def make_chunk(
    chunk_id: str = "abc123-fixed-0000",
    doc_id: str = "doc.html",
    *,
    index: int = 0,
    count: int = 1,
    text: str = "hello world",
    metadata: dict[str, Any] | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        doc_id=doc_id,
        text=text,
        index=index,
        count=count,
        length=len(text),
        length_unit="characters",
        block_refs=[BlockReference(0, 1, "text", 0, len(text))],
        heading="Intro",
        metadata={"source": doc_id, "pages": [1], **(metadata or {})},
    )


def dense_record(
    chunk_id: str = "abc123-fixed-0000",
    doc_id: str = "doc.html",
    *,
    seed: int = 0,
    metadata: dict[str, Any] | None = None,
    space: VectorSpaceConfig | None = None,
) -> StoredRecord:
    space = space or dense_space()
    return StoredRecord(
        chunk=make_chunk(chunk_id, doc_id),
        vectors={
            space.name: SpaceVectors(
                space=space.name,
                embedding_space=space.embedding_space,
                dense=unit_vector(seed),
            )
        },
        metadata=dict(metadata or {}),
    )


@pytest.fixture()
def store():
    """Return an in-memory Qdrant store."""
    from vector_stores.qdrant import QdrantVectorStore

    instance = QdrantVectorStore.memory()
    try:
        yield instance
    finally:
        instance.close()
