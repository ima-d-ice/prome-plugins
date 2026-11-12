"""The protocols every Prome retrieval component implements.

A *retriever* turns a :class:`~retrieval.contracts.models.RetrievalQuery`
into a :class:`~retrieval.contracts.models.RetrievalResponse`.  The dense
channel, the lexical channel and the hybrid strategy all implement it, so a
caller can use any of them alone or swap one for another.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from retrieval.contracts.capabilities import RetrievalCapabilities
from retrieval.contracts.models import RetrievalQuery, RetrievalResponse


@runtime_checkable
class Retriever(Protocol):
    """Interface every Prome retriever implements."""

    name: str
    """Stable identifier of the retriever (``"dense"``, ``"bm25"``, ...)."""

    @property
    def capabilities(self) -> RetrievalCapabilities:
        """Return what the retriever can do."""

        ...

    def retrieve(self, query: RetrievalQuery) -> RetrievalResponse:
        """Retrieve the passages that answer ``query``.

        Implementations must:

        - validate the query and fail loudly on an empty one;
        - apply the metadata filter of the query to every candidate;
        - never return a passage the filter excludes;
        - report what they searched through
          :class:`~retrieval.contracts.models.RetrievalDiagnostics`.
        """

        ...


#: Backwards-compatible alias of :class:`Retriever`.
#:
#: The name was changed when the retrieval contracts were aligned with the
#: vocabulary of the retrieval ecosystem; the alias keeps the earlier import
#: path working.
RetrievalStrategy = Retriever
