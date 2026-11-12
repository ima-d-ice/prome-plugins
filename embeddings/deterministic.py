"""A dependency-free, reproducible embedding model for tests and benchmarks.

This model is **not** a semantic model and it is **not** the production
encoder.  It is a deterministic *hashed bag-of-words* projection: every token
is hashed into a fixed number of buckets, the buckets are accumulated, and the
result is L2-normalized.

Two texts that share tokens therefore get a high cosine similarity, and two
unrelated texts get a low one.  That is enough to exercise every code path of
a retrieval system -- indexing, filtering, ranking, deduplication -- without
downloading weights, and it produces the *same* vectors on every machine and
every run.

The production model is :class:`embeddings.qwen3.Qwen3EmbeddingModel`
(``Qwen/Qwen3-Embedding-0.6B``); use it when the semantic quality of the
embeddings matters.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from embeddings.contracts.models import (
    EmbeddingCapabilities,
    EmbeddingInput,
    EmbeddingResult,
    EmbeddingSpaceDescriptor,
    Normalization,
    RepresentationType,
)

#: Words are lower-cased and split on everything that is not a letter, a
#: digit or an underscore.
_TOKEN_RE = re.compile(r"[a-z0-9_]+")

#: The identifier reported as the model id.
MODEL_ID = "prome/deterministic-hashed-bow"


class DeterministicEmbeddingModel:
    """A reproducible hashed bag-of-words embedding model.

    Args:
        dimension: the width of the produced vectors.
        model_id: the identifier reported by the model.
    """

    name = MODEL_ID

    def __init__(self, dimension: int = 256, model_id: str = MODEL_ID) -> None:
        if not isinstance(dimension, int) or isinstance(dimension, bool):
            raise TypeError(f"dimension must be an integer, got {dimension!r}")
        if dimension < 1:
            raise ValueError(f"dimension must be >= 1, got {dimension}")
        self.dimension = dimension
        self.model_id = model_id
        self._capabilities = EmbeddingCapabilities(
            dense=True,
            sparse=False,
            multi_vector=False,
            query_embedding=True,
            document_embedding=True,
            batch_inference=True,
            max_sequence_length=8192,
            embedding_dimensions=dimension,
            supported_languages=None,
        )

    # ------------------------------------------------------------------
    @property
    def capabilities(self) -> EmbeddingCapabilities:
        """Return the capabilities of the model."""
        return self._capabilities

    def get_model_info(self) -> dict[str, Any]:
        """Return metadata about the model."""
        return {
            "model_id": self.model_id,
            "adapter_name": "deterministic",
            "dimensions": self.dimension,
            "representation_type": "dense",
            "normalization": "l2",
            "semantic": False,
            "notes": (
                "hashed bag-of-words projection; reproducible but not semantic. "
                "Use Qwen3-Embedding-0.6B for production quality."
            ),
        }

    def get_embedding_space(self) -> EmbeddingSpaceDescriptor:
        """Return the embedding-space descriptor of the model."""
        return EmbeddingSpaceDescriptor(
            model_id=self.model_id,
            model_revision=None,
            adapter_name="deterministic",
            dimensions=self.dimension,
            representation_type=RepresentationType.DENSE,
            normalization=Normalization.L2,
            query_instruction=None,
            document_instruction=None,
            supported_languages=None,
        )

    # ------------------------------------------------------------------
    def embed_documents(self, inputs: list[EmbeddingInput]) -> list[EmbeddingResult]:
        """Embed a batch of document inputs."""
        return [self._result(item) for item in inputs]

    def embed_query(self, input: EmbeddingInput) -> EmbeddingResult:
        """Embed a single query."""
        return self._result(input)

    def embed_queries(self, inputs: list[EmbeddingInput]) -> list[EmbeddingResult]:
        """Embed a batch of query inputs."""
        return [self._result(item) for item in inputs]

    # ------------------------------------------------------------------
    def _result(self, item: EmbeddingInput) -> EmbeddingResult:
        """Turn one input into an embedding result."""
        return EmbeddingResult(
            model_id=self.model_id,
            dimensions=self.dimension,
            dense_vector=self._vector(item.text),
            representation_type=RepresentationType.DENSE,
            normalization=Normalization.L2,
            metadata={"adapter_name": "deterministic"},
            doc_id=item.doc_id,
            chunk_id=item.chunk_id,
        )

    def _vector(self, text: str) -> list[float]:
        """Return the normalized vector of ``text``."""
        vector = [0.0] * self.dimension
        for token in _TOKEN_RE.findall((text or "").lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            bucket = value % self.dimension
            # The sign makes the projection closer to the random projection
            # family, which keeps unrelated texts near orthogonal.
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            # An empty (or token-free) text has no direction; the zero vector
            # is the honest representation of "no information".
            return vector
        return [value / norm for value in vector]
