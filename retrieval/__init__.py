"""Prome RAG Retrieval Ecosystem.

Retrieval turns a query into ranked passages.  The ecosystem is built on the
existing Prome contracts and never duplicates them:

- :class:`chunkers.contracts.models.Chunk` for the identity and provenance of
  a passage;
- :class:`embeddings.contracts.protocols.EmbeddingModel` for the embeddings
  (the model is always injected, never loaded here);
- :class:`vector_stores.contracts.protocols.VectorStore` for the storage and
  the search.

The strategies share one result contract, so a dense search and a
hierarchical search can be swapped for one another downstream.
"""

from retrieval.contracts import (
    ContextChunk,
    RetrievalConfigError,
    RetrievalDiagnostics,
    RetrievalError,
    RetrievalResponse,
    RetrievalResult,
    RetrievalStrategy,
    UnsupportedRetrievalCapabilityError,
)
from retrieval.dense import DenseRetriever

__all__ = [
    "ContextChunk",
    "DenseRetriever",
    "RetrievalConfigError",
    "RetrievalDiagnostics",
    "RetrievalError",
    "RetrievalResponse",
    "RetrievalResult",
    "RetrievalStrategy",
    "UnsupportedRetrievalCapabilityError",
]
