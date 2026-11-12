"""Shared data models for the embedding ecosystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RepresentationType(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    MULTI_VECTOR = "multi-vector"


class Normalization(str, Enum):
    NONE = "none"
    L2 = "l2"
    RELU = "relu"


@dataclass
class EmbeddingInput:
    """Input to an embedding model.

    Supports text content, optional metadata, source document ID, chunk ID,
    and source references for provenance.
    """

    text: str
    """The input text to embed."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Optional metadata associated with the input."""

    doc_id: str | None = None
    """Source document ID."""

    chunk_id: str | None = None
    """Source chunk ID."""

    source_refs: list[str] | None = None
    """List of source reference IDs for provenance."""


@dataclass
class EmbeddingResult:
    """Result from an embedding model.

    Contains dense and/or sparse vectors, multi-vector representations,
    model identity, dimensions, and normalization information.
    """

    model_id: str
    """Identifier of the model that produced this embedding."""

    dimensions: int
    """Dimensionality of the dense vector."""

    dense_vector: list[float] | None = None
    """Dense embedding vector when available."""

    sparse_indices: list[int] | None = None
    """Sparse vector token indices when available."""

    sparse_values: list[float] | None = None
    """Sparse vector token weights when available."""

    multi_vector: list[list[float]] | None = None
    """Multi-vector / ColBERT-style representation when available."""

    representation_type: RepresentationType = RepresentationType.DENSE
    """The type of representation returned."""

    normalization: Normalization = Normalization.NONE
    """Normalization applied to the vector."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata from the embedding operation."""

    doc_id: str | None = None
    """Source document ID retained from input."""

    chunk_id: str | None = None
    """Source chunk ID retained from input."""

    query_id: str | None = None
    """Identifier for query embedding, if different from chunk ID."""


@dataclass
class EmbeddingCapabilities:
    """Capabilities description for an embedding model."""

    dense: bool = True
    """Dense embedding support."""

    sparse: bool = False
    """Sparse lexical embedding support."""

    multi_vector: bool = False
    """Multi-vector / ColBERT-style representation support."""

    query_embedding: bool = True
    """Query embedding support."""

    document_embedding: bool = True
    """Document embedding support."""

    batch_inference: bool = True
    """Batch inference support."""

    max_sequence_length: int = 8192
    """Maximum supported input sequence length."""

    embedding_dimensions: int | None = None
    """Default embedding dimensions, if fixed."""

    supported_languages: list[str] | None = None
    """Languages the model supports, if known."""


@dataclass
class EmbeddingSpaceDescriptor:
    """Descriptor of an embedding space with compatibility checking."""

    model_id: str
    """Model identifier."""

    model_revision: str | None = None
    """Model revision / tag used."""

    adapter_name: str | None = None
    """Name of the adapter implementation."""

    dimensions: int | None = None
    """Vector dimensionality."""

    representation_type: RepresentationType = RepresentationType.DENSE
    """Primary representation type."""

    normalization: Normalization = Normalization.NONE
    """Normalization configuration."""

    query_instruction: str | None = None
    """Instruction/prompt for query encoding."""

    document_instruction: str | None = None
    """Instruction/prompt for document encoding."""

    supported_languages: list[str] | None = None
    """Languages supported by the model."""

    def is_compatible_with(
        self,
        other: EmbeddingSpaceDescriptor,
        *,
        allow_sparse: bool = False,
        allow_multi_vector: bool = False,
    ) -> bool:
        """Check compatibility between two embedding spaces.

        Two embeddings are compatible only if they share the same model
        identity and revision. Dimensional match alone is NOT sufficient
        for compatibility.

        Args:
            other: The other embedding space descriptor to compare against.
            allow_sparse: If True, allow sparse representation compatibility.
            allow_multi_vector: If True, allow multi-vector representation compatibility.

        Returns:
            True if the spaces are compatible, False otherwise.
        """
        if self.model_id != other.model_id:
            return False
        if self.model_revision != other.model_revision:
            return False
        if self.dimensions != other.dimensions:
            return False
        if self.representation_type != other.representation_type:
            return False
        if self.normalization != other.normalization:
            return False
        # Check language overlap
        if (
            self.supported_languages
            and other.supported_languages
            and not set(self.supported_languages).intersection(
                other.supported_languages,
            )
        ):
            return False
        return True