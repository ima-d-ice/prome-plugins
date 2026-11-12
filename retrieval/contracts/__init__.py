"""Shared contracts for the Prome RAG retrieval ecosystem.

The contracts are backend and strategy independent: they describe the query
a caller submits (:class:`~retrieval.contracts.models.RetrievalQuery`), the
result a strategy returns
(:class:`~retrieval.contracts.models.RetrievalResponse`) and the interface
every retriever implements
(:class:`~retrieval.contracts.protocols.Retriever`).

They reuse the upstream contracts instead of duplicating them:

- :class:`chunkers.contracts.models.Chunk` identity and provenance are carried
  by the vector-store contracts
  (:class:`~vector_stores.contracts.models.ChunkProvenance`);
- the metadata filter of a query is the vector-store filter AST
  (:class:`~vector_stores.contracts.filters.MetadataFilter`), so every channel
  understands the *same* filter;
- the score direction of a hit comes from the vector-store contract
  (:class:`~vector_stores.contracts.models.ScoreDirection`).
"""

from retrieval.contracts.capabilities import RetrievalCapabilities
from retrieval.contracts.exceptions import (
    FTS5UnavailableError,
    HierarchicalIntegrityError,
    LexicalIndexError,
    RetrievalConfigError,
    RetrievalDependencyError,
    RetrievalError,
    UnsupportedRetrievalCapabilityError,
)
from retrieval.contracts.models import (
    ContextChunk,
    RetrievalDiagnostics,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalResult,
    RetrievedItem,
)
from retrieval.contracts.protocols import Retriever, RetrievalStrategy

__all__ = [
    "ContextChunk",
    "FTS5UnavailableError",
    "HierarchicalIntegrityError",
    "LexicalIndexError",
    "RetrievalCapabilities",
    "RetrievalConfigError",
    "RetrievalDependencyError",
    "RetrievalDiagnostics",
    "RetrievalError",
    "RetrievalQuery",
    "RetrievalResponse",
    "RetrievalResult",
    "RetrievalStrategy",
    "Retriever",
    "RetrievedItem",
    "UnsupportedRetrievalCapabilityError",
]
