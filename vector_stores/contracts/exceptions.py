"""Common exceptions for Prome RAG vector stores.

The hierarchy mirrors the other ecosystems: a single base error
(:class:`VectorStoreError`) plus narrow subclasses so callers can react to a
specific failure without parsing messages.  Everything raised by a vector
store adapter is a :class:`VectorStoreError`; nothing leaks a vendor
exception to the caller.
"""


class VectorStoreError(Exception):
    """Base class for all vector-store errors."""


class VectorStoreConfigError(VectorStoreError, ValueError):
    """Invalid configuration (bad collection config, bad search request, ...)."""


class VectorStoreDependencyError(VectorStoreError, ImportError):
    """An optional backend dependency is not installed.

    The message always names the extra to install, e.g.
    ``pip install 'prome-rag-plugins[qdrant]'``.
    """


class CollectionNotFoundError(VectorStoreError, KeyError):
    """The requested collection does not exist in the backend."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        # ``KeyError.__str__`` wraps the message in quotes; undo that so the
        # error reads like every other exception in the project.
        return VectorStoreError.__str__(self)


class CollectionExistsError(VectorStoreError, ValueError):
    """A collection with the same name already exists."""


class RecordNotFoundError(VectorStoreError, KeyError):
    """A requested record does not exist in the collection."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return VectorStoreError.__str__(self)


class InvalidRecordError(VectorStoreError, ValueError):
    """A record is malformed (bad vectors, bad metadata, duplicate ids, ...)."""


class DimensionMismatchError(InvalidRecordError):
    """A vector does not have the number of dimensions declared by the space."""


class EmbeddingSpaceMismatchError(InvalidRecordError):
    """A vector belongs to a different model / embedding space than the target."""


class UnsupportedVectorTypeError(VectorStoreError, ValueError):
    """The backend or the target vector space does not support this vector type."""


class UnsupportedFilterOperationError(VectorStoreError, ValueError):
    """The backend cannot express the requested metadata filter operation."""


class UnsupportedIndexOperationError(VectorStoreError, ValueError):
    """The backend cannot create or delete that kind of index."""


class BackendConnectionError(VectorStoreError, ConnectionError):
    """The backend could not be reached or failed while serving a request."""
