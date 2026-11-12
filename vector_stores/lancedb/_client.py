"""Lazy access to the optional ``lancedb`` dependency.

LanceDB is an optional dependency: importing :mod:`vector_stores.lancedb`
must work without it, and the error must only surface when the backend is
actually used.  This mirrors the convention used by the other adapters of
this repository.
"""

from __future__ import annotations

from types import ModuleType

from vector_stores.contracts.exceptions import VectorStoreDependencyError

_INSTALL_HINT = (
    "The 'lancedb' package is required for the LanceDB vector store. "
    "Install it with: pip install 'prome-rag-plugins[lancedb]' "
    "(or: pip install 'lancedb>=0.41')"
)


def import_lancedb() -> ModuleType:
    """Import and return the ``lancedb`` module.

    Raises:
        VectorStoreDependencyError: if ``lancedb`` is not installed.
    """
    try:
        import lancedb
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise VectorStoreDependencyError(_INSTALL_HINT) from exc
    return lancedb


def pyarrow() -> ModuleType:
    """Return the ``pyarrow`` module (a LanceDB dependency)."""
    try:
        import pyarrow
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise VectorStoreDependencyError(_INSTALL_HINT) from exc
    return pyarrow


def index_module() -> ModuleType:
    """Return the ``lancedb.index`` module (index configurations)."""
    try:
        from lancedb import index
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise VectorStoreDependencyError(_INSTALL_HINT) from exc
    return index
