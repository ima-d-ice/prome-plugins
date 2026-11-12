"""Common exceptions for the Prome RAG retrieval ecosystem.

The hierarchy mirrors the other ecosystems: a single base error
(:class:`RetrievalError`) plus narrow subclasses so callers can react to a
specific failure without parsing messages.
"""

from __future__ import annotations


class RetrievalError(Exception):
    """Base class for all retrieval errors."""


class RetrievalConfigError(RetrievalError, ValueError):
    """Invalid configuration (bad limits, bad filter, ...)."""


class RetrievalDependencyError(RetrievalError, ImportError):
    """An optional dependency needed by the retriever is not installed.

    The message always names the extra to install.
    """


class UnsupportedRetrievalCapabilityError(RetrievalError, ValueError):
    """The backend cannot do what the retriever needs.

    Raised instead of silently degrading: a retriever that needs a filtered
    search on a backend without filter support must say so rather than search
    everything and pretend it filtered.
    """


class HierarchicalIntegrityError(RetrievalError, ValueError):
    """The parent/child relationship of the indexed hierarchy is broken.

    Examples: a child that references a parent of another document, a parent
    that claims children which do not exist, or duplicate chunk ids.
    """


class LexicalIndexError(RetrievalError):
    """The lexical index could not be opened, written or searched."""


class FTS5UnavailableError(LexicalIndexError, RetrievalDependencyError):
    """The SQLite build of this interpreter does not provide FTS5.

    The lexical channel needs FTS5 for its BM25 ranking.  It is never replaced
    by an unrelated implementation: the capability is reported instead.
    """
