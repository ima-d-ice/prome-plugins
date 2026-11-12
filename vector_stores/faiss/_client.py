"""Lazy access to the optional ``faiss`` dependency.

FAISS is an optional dependency: importing :mod:`vector_stores.faiss` must
work without it, and the error must only surface when the backend is
actually used.  This mirrors the convention used by the other adapters of
this repository.

**FAISS and torch cannot live in the same process on macOS.**  The
``faiss-cpu`` wheel links its own copy of ``libomp`` and so does torch: when
the second OpenMP runtime initializes, the interpreter aborts with
``OMP: Error #15`` (this is a hard abort, not a catchable exception).  The
guard below turns the most common case - a process that already loaded torch
- into a clear error instead of a crash.
"""

from __future__ import annotations

import sys
from types import ModuleType

from vector_stores.contracts.exceptions import VectorStoreDependencyError

_INSTALL_HINT = (
    "The 'faiss-cpu' package is required for the FAISS vector store. "
    "Install it with: pip install 'prome-rag-plugins[faiss]' "
    "(or: pip install 'faiss-cpu>=1.16')"
)

_OPENMP_HINT = (
    "FAISS and torch cannot be loaded in the same process: the 'faiss-cpu' "
    "wheel and torch each link their own copy of the OpenMP runtime and the "
    "second one aborts the interpreter ('OMP: Error #15'). Use the FAISS "
    "adapter in a process that does not import torch, or use another backend."
)


def import_faiss() -> ModuleType:
    """Import and return the ``faiss`` module.

    Raises:
        VectorStoreDependencyError: if FAISS is not installed, or if torch is
            already loaded (the two OpenMP runtimes are incompatible).
    """
    if "torch" in sys.modules:
        raise VectorStoreDependencyError(_OPENMP_HINT)
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise VectorStoreDependencyError(_INSTALL_HINT) from exc
    return faiss


def numpy() -> ModuleType:
    """Return the ``numpy`` module (a FAISS dependency)."""
    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise VectorStoreDependencyError(_INSTALL_HINT) from exc
    return numpy
