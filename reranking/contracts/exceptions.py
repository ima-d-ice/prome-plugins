"""Common exceptions for the Prome RAG reranking ecosystem.

The hierarchy mirrors the other ecosystems: a single base error
(:class:`RerankError`) plus narrow subclasses so callers can react to a
specific failure without parsing messages.
"""

from __future__ import annotations


class RerankError(Exception):
    """Base class for all reranking errors."""


class RerankConfigError(RerankError, ValueError):
    """Invalid reranking configuration (bad limits, bad instruction, ...)."""


class RerankDependencyError(RerankError, ImportError):
    """An optional dependency needed by the reranker is not installed.

    The message always names the extra to install.
    """


class RerankInferenceError(RerankError):
    """The model failed while producing relevance scores.

    Raised instead of returning partial or empty results so that callers
    never silently continue with degraded input.
    """


class UnsupportedDeviceError(RerankError, ValueError):
    """The requested computing device is not available or supported.

    Examples: requesting CUDA when no GPU is available, requesting MPS on a
    system without Apple silicon.
    """


class InvalidScoreError(RerankError, ValueError):
    """The model returned an invalid score (NaN or infinity)."""