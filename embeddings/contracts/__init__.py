"""Shared contracts for Prome RAG embedding models.

See :mod:`embeddings.contracts.models` for the data model
(:class:`EmbeddingInput`, :class:`EmbeddingResult`,
:class:`EmbeddingCapabilities`, :class:`EmbeddingSpaceDescriptor`) and
:mod:`embeddings.contracts.protocols` for the model interface.
"""

from embeddings.contracts.models import (
    EmbeddingCapabilities,
    EmbeddingInput,
    EmbeddingResult,
    EmbeddingSpaceDescriptor,
    Normalization,
    RepresentationType,
)
from embeddings.contracts.protocols import EmbeddingModel

# The exception and validation helpers are convenience utilities: they are
# re-exported when available, but the core contract above must always
# import cleanly.
try:  # pragma: no cover - depends on the optional helper modules
    from embeddings.contracts.exceptions import (
        EmbeddingConfigError,
        EmbeddingError,
        EmbeddingInferenceError,
    )
except ImportError:  # pragma: no cover
    EmbeddingConfigError = EmbeddingError = EmbeddingInferenceError = None  # type: ignore[assignment]

try:  # pragma: no cover - depends on the optional helper modules
    from embeddings.contracts.validation import cosine_similarity, validate_dimensions
except ImportError:  # pragma: no cover
    cosine_similarity = validate_dimensions = None  # type: ignore[assignment]

__all__ = [
    "EmbeddingCapabilities",
    "EmbeddingConfigError",
    "EmbeddingError",
    "EmbeddingInferenceError",
    "EmbeddingInput",
    "EmbeddingModel",
    "EmbeddingResult",
    "EmbeddingSpaceDescriptor",
    "Normalization",
    "RepresentationType",
    "cosine_similarity",
    "validate_dimensions",
]
