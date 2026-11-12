"""Shared contracts for the Prome RAG retrieval ecosystem.

The contracts are backend and strategy independent: they describe the result
a strategy returns (:class:`~retrieval.contracts.models.RetrievalResult`), the
response envelope (:class:`~retrieval.contracts.models.RetrievalResponse`) and
the strategy interface
(:class:`~retrieval.contracts.protocols.RetrievalStrategy`).

Every strategy — dense, hybrid, hierarchical — answers with the same models, so
one can be swapped for another without touching the caller.
"""

from retrieval.contracts.exceptions import (
    HierarchicalIntegrityError,
    RetrievalConfigError,
    RetrievalDependencyError,
    RetrievalError,
    UnsupportedRetrievalCapabilityError,
)
from retrieval.contracts.models import (
    ContextChunk,
    RetrievalDiagnostics,
    RetrievalResponse,
    RetrievalResult,
)
from retrieval.contracts.protocols import RetrievalStrategy

__all__ = [
    "ContextChunk",
    "HierarchicalIntegrityError",
    "RetrievalConfigError",
    "RetrievalDependencyError",
    "RetrievalDiagnostics",
    "RetrievalError",
    "RetrievalResponse",
    "RetrievalResult",
    "RetrievalStrategy",
    "UnsupportedRetrievalCapabilityError",
]
