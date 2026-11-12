"""The protocol every Prome retrieval strategy implements.

A strategy turns a query into a :class:`~retrieval.contracts.models.RetrievalResponse`.
It never embeds the query itself when a caller wants to reuse one embedding
across several strategies, and it never hides *how* it answered: the
diagnostics travel with the results.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from retrieval.contracts.models import RetrievalResponse


@runtime_checkable
class RetrievalStrategy(Protocol):
    """Interface every Prome retrieval strategy implements."""

    name: str
    """Stable identifier of the strategy (``"dense"``, ``"hybrid"``, ...)."""

    def retrieve(self, query: str) -> RetrievalResponse:
        """Retrieve the passages that answer ``query``.

        Implementations must:

        - validate the query and fail loudly on an empty one;
        - embed the query at most once;
        - never return a hit from a document the strategy was restricted to
          excluding;
        - report what it searched through
          :class:`~retrieval.contracts.models.RetrievalDiagnostics`.
        """

        ...
