"""Common exceptions for the Prome RAG retrieval ecosystem.

The hierarchy mirrors the other ecosystems: a single base error
(:class:`RetrievalError`) plus narrow subclasses so callers can react to a
specific failure without parsing messages.
"""


class RetrievalError(Exception):
    """Base class for all retrieval errors."""


class RetrievalConfigError(RetrievalError, ValueError):
    """Invalid configuration (bad limits, bad filter, ...)."""


class RetrievalDependencyError(RetrievalError, ImportError):
    """An optional dependency needed by the strategy is not installed.

    The message always names the extra to install.
    """


class UnsupportedRetrievalCapabilityError(RetrievalError, ValueError):
    """The backend cannot do what the strategy needs.

    Raised instead of silently degrading: a strategy that needs a filtered
    search on a backend without filter support must say so rather than search
    everything and pretend it filtered.
    """


class HierarchicalIntegrityError(RetrievalError, ValueError):
    """The parent/child relationship of the indexed hierarchy is broken.

    Examples: a child that references a parent of another document, a parent
    that claims children which do not exist, or duplicate chunk ids.
    """
