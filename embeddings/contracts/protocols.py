"""Shared embedding model protocol.

The contract is deliberately small and provider-agnostic: an embedding
model turns text into vectors and describes the space it produces.
Consumers such as the semantic chunker depend on this interface only --
never on a specific provider, library or model weight file.

Document embedding (``embed_documents``) is the operation required for
indexing document segments. Query embedding is part of the same interface
because a retrieval system needs both sides of the comparison.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from embeddings.contracts.models import (
    EmbeddingCapabilities,
    EmbeddingInput,
    EmbeddingResult,
    EmbeddingSpaceDescriptor,
)


@runtime_checkable
class EmbeddingModel(Protocol):
    """Interface every Prome embedding model implements.

    Implementations must:

    - return one :class:`EmbeddingResult` per input, in the same order;
    - fill ``dense_vector`` for dense representations;
    - report the vector length in ``dimensions``;
    - raise instead of returning empty or partial results on failure.
    """

    name: str

    @property
    def capabilities(self) -> EmbeddingCapabilities:
        """Return the capabilities of this embedding model."""

        ...

    def get_model_info(self) -> dict[str, Any]:
        """Return metadata about the model."""

        ...

    def get_embedding_space(self) -> EmbeddingSpaceDescriptor:
        """Return the embedding space descriptor for this model."""

        ...

    def embed_documents(self, inputs: list[EmbeddingInput]) -> list[EmbeddingResult]:
        """Embed a batch of document inputs.

        Each input's source_doc_id and chunk_id are retained in the result.
        """

        ...

    def embed_query(self, input: EmbeddingInput) -> EmbeddingResult:
        """Embed a single query input.

        Query embedding may use different instructions or prefixes than
        document embedding.
        """

        ...

    def embed_queries(self, inputs: list[EmbeddingInput]) -> list[EmbeddingResult]:
        """Embed a list of query inputs.

        Each query may use different instructions or prefixes than document
        embedding.
        """

        ...