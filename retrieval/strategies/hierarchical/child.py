"""The fine stage of the hierarchical strategy: finding the evidence children.

The child stage answers a second question: *inside the selected sections,
which passages are the evidence?*  It searches only the records that carry
``chunk_level == "child"`` **and** whose ``parent_id`` is one of the selected
parents.

The restriction is expressed in the portable filter language and pushed down
to the backend, so the backend itself only ever considers the children of the
selected parents.  The stage never falls back to searching every child: if the
backend cannot express the restriction, the search fails loudly instead of
silently widening the search space.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from retrieval.contracts.exceptions import RetrievalConfigError
from retrieval.dense.retriever import DenseRetriever
from retrieval.strategies.hierarchical.metadata import child_filter
from vector_stores.contracts.filters import MetadataFilter, and_
from vector_stores.contracts.models import SearchResult


@dataclass(frozen=True)
class ChildStageResult:
    """One child chunk selected by the fine stage."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    rank: int
    parent_id: str
    """Id of the parent this child belongs to."""
    metadata: dict
    provenance: object | None

    @property
    def heading_path(self) -> tuple[str, ...]:
        """Return the heading path of the chunk, when the backend stored it."""
        path = self.metadata.get("heading_path")
        return tuple(path) if path else ()


class ChildStage:
    """Selects the child chunks of the selected parents.

    Args:
        retriever: the shared dense retriever (owns the store and the model).
        top_k: how many children the stage may retrieve.
        score_threshold: minimum score a child must reach.
        filters: additional metadata filter, combined with the level and
            parent filters.
    """

    name = "hierarchical-child"

    def __init__(
        self,
        retriever: DenseRetriever,
        *,
        top_k: int = 40,
        score_threshold: float | None = None,
        filters: MetadataFilter | None = None,
    ) -> None:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise RetrievalConfigError(f"top_k must be an integer >= 1, got {top_k!r}")
        self.retriever = retriever
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.filters = filters

    # ------------------------------------------------------------------
    def filter_for(self, parent_ids: Sequence[str]) -> MetadataFilter:
        """Return the filter that restricts the search to ``parent_ids``.

        The filter is ``chunk_level == "child" AND parent_id IN (...)``, plus
        the user filter when one is configured.
        """
        if not parent_ids:
            raise RetrievalConfigError(
                "the child stage needs at least one selected parent; searching "
                "without one would search every child in the collection"
            )
        restriction = child_filter(list(parent_ids))
        if self.filters is None:
            return restriction
        return and_(restriction, self.filters)

    def search(
        self,
        query_vector: Sequence[float],
        parent_ids: Sequence[str],
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[ChildStageResult]:
        """Return the children of ``parent_ids`` that match ``query_vector``.

        Args:
            query_vector: the embedded query.
            parent_ids: the ids of the selected parents; the search is
                restricted to their children.
            top_k: overrides the configured limit.
            score_threshold: overrides the configured threshold.
        """
        limit = self.top_k if top_k is None else top_k
        threshold = (
            self.score_threshold if score_threshold is None else score_threshold
        )
        hits = self.retriever.search(
            query_vector,
            limit=limit,
            filter=self.filter_for(parent_ids),
            score_threshold=threshold,
        )
        return [self._to_result(hit, rank) for rank, hit in enumerate(hits, start=1)]

    def search_text(
        self, query: str, parent_ids: Sequence[str], **kwargs
    ) -> list[ChildStageResult]:
        """Embed ``query`` with the injected model and search."""
        vector = self.retriever.embed_query(query)
        return self.search(vector, parent_ids, **kwargs)

    # ------------------------------------------------------------------
    @staticmethod
    def _to_result(hit: SearchResult, rank: int) -> ChildStageResult:
        """Turn a backend hit into a child stage result."""
        metadata = dict(hit.metadata or {})
        parent_id = str(metadata.get("parent_id") or "")
        return ChildStageResult(
            chunk_id=hit.chunk_id,
            doc_id=hit.doc_id,
            text=hit.text or "",
            score=hit.score,
            rank=rank,
            parent_id=parent_id,
            metadata=metadata,
            provenance=hit.provenance,
        )


__all__ = ["ChildStage", "ChildStageResult"]
