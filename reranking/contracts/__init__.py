"""Shared contracts for the Prome RAG reranking ecosystem.

The contracts are independent of Qwen and Sentence Transformers: they describe
the data models and protocol that a reranker implementation must satisfy.

They reuse the upstream chunk and retrieval contracts instead of duplicating
them: :class:`RerankCandidate` references the original retrieval result so the
candidate object remains recoverable after reranking.
"""

from __future__ import annotations

from reranking.contracts.exceptions import (
    RerankConfigError,
    RerankDependencyError,
    RerankError,
    RerankInferenceError,
)
from reranking.contracts.models import (
    RerankCandidate,
    RerankedItem,
    RerankRequest,
    RerankResponse,
)
from reranking.contracts.protocols import Reranker

__all__ = [
    "RerankCandidate",
    "RerankConfigError",
    "RerankDependencyError",
    "RerankError",
    "RerankInferenceError",
    "RerankRequest",
    "RerankResponse",
    "RerankedItem",
    "Reranker",
]