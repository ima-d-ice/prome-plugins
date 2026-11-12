"""Lazy access to the optional ``qdrant-client`` dependency.

The Qdrant adapter is optional: importing :mod:`vector_stores.qdrant` must
work without ``qdrant-client`` installed, and the error must only surface
when the backend is actually used.  This mirrors the convention used by the
parser and embedding adapters of this repository.
"""

from __future__ import annotations

from types import ModuleType

from vector_stores.contracts.exceptions import VectorStoreDependencyError

_INSTALL_HINT = (
    "The 'qdrant-client' package is required for the Qdrant vector store. "
    "Install it with: pip install 'prome-rag-plugins[qdrant]' "
    "(or: pip install 'qdrant-client>=1.12')"
)


def import_qdrant() -> tuple[ModuleType, ModuleType]:
    """Import and return ``(qdrant_client, qdrant_client.models)``.

    Raises:
        VectorStoreDependencyError: if ``qdrant-client`` is not installed.
    """
    try:
        import qdrant_client
        from qdrant_client import models
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise VectorStoreDependencyError(_INSTALL_HINT) from exc
    return qdrant_client, models


def models() -> ModuleType:
    """Return the ``qdrant_client.models`` module."""
    return import_qdrant()[1]
