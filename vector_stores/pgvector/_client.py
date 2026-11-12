"""Lazy access to the optional PostgreSQL dependencies.

The pgvector adapter needs two optional packages:

- ``psycopg`` (v3), the PostgreSQL driver;
- ``pgvector``, the Python adapter for the ``vector`` / ``sparsevec`` types.

Both are imported lazily, so importing :mod:`vector_stores.pgvector` works
without them.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

from vector_stores.contracts.exceptions import VectorStoreDependencyError

_INSTALL_HINT = (
    "The 'psycopg' and 'pgvector' packages are required for the PostgreSQL "
    "vector store. Install them with: "
    "pip install 'prome-rag-plugins[pgvector]' "
    "(or: pip install 'psycopg[binary]>=3.2' 'pgvector>=0.4')"
)


def import_psycopg() -> ModuleType:
    """Import and return the ``psycopg`` module.

    Raises:
        VectorStoreDependencyError: if psycopg is not installed.
    """
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise VectorStoreDependencyError(_INSTALL_HINT) from exc
    return psycopg


def import_pgvector() -> ModuleType:
    """Import and return the ``pgvector.psycopg`` module.

    Raises:
        VectorStoreDependencyError: if pgvector is not installed.
    """
    try:
        from pgvector import psycopg as pgvector_psycopg
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise VectorStoreDependencyError(_INSTALL_HINT) from exc
    return pgvector_psycopg


def register_vector(connection: Any) -> None:
    """Register the ``vector``/``sparsevec`` types on ``connection``."""
    import_pgvector().register_vector(connection)
