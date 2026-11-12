"""Common exceptions for Prome RAG embedding models."""


class EmbeddingError(Exception):
    """Base class for all embedding-related errors."""


class EmbeddingConfigError(EmbeddingError, ValueError):
    """Invalid embedding model configuration (bad model, dimension, ...)."""


class EmbeddingInferenceError(EmbeddingError):
    """The model failed while producing embeddings.

    Raised instead of returning partial or empty results so that callers
    (for example the semantic chunker) never silently continue with
    degraded input.
    """
