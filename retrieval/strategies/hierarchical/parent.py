"""The coarse stage of the hierarchical strategy: finding the parent sections.

The parent stage answers one question: *which sections of the corpus are
relevant to the query?*  It searches only the records that carry
``chunk_level == "parent"``, so a parent search can never return a child
record and the fine stage is never fed a child as if it were a section.

The stage is independently usable and independently testable: it takes a query
vector (or a query string plus an embedding model) and returns the matching
parent records.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from retrieval.contracts.exceptions import RetrievalConfigError
from retrieval.dense.retriever import DenseRetriever
from retrieval.strategies.hierarchical.metadata import (
    LEVEL_PARENT,
    parent_filter,
)
from vector_stores.contracts.filters import MetadataFilter, and_
from vector_stores.contracts.models import SearchResult


@dataclass(frozen=True)
class ParentStageResult:
    """One parent section selected by the coarse stage."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    rank: int
    metadata: dict
    provenance: object | None
    child_count: int
    """Number of children the parent declares."""
    child_ids: tuple[str, ...]
    """Ordered ids of the children of this parent."""

    @property
    def heading_path(self) -> tuple[str, ...]:
        """Return the heading path of the section, when the backend stored it."""
        path = self.metadata.get("heading_path")
        return tuple(path) if path else ()


class ParentStage:
    """Selects the parent sections relevant to a query.

    Args:
        retriever: the shared dense retriever (owns the store and the model).
        top_k: how many parents the stage may select.
        score_threshold: minimum score a parent must reach.
        filters: additional metadata filter, combined with the level filter.
    """

    name = "hierarchical-parent"

    def __init__(
        self,
        retriever: DenseRetriever,
        *,
        top_k: int = 8,
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
    @property
    def filter(self) -> MetadataFilter:
        """Return the filter this stage searches with.

        It is always the conjunction of the parent-level filter and the
        user-supplied filter, so a child record can never slip through.
        """
        level = parent_filter()
        if self.filters is None:
            return level
        return and_(level, self.filters)

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[ParentStageResult]:
        """Return the parent sections that match ``query_vector``.

        Args:
            query_vector: the embedded query.
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
            filter=self.filter,
            score_threshold=threshold,
        )
        return [self._to_result(hit, rank) for rank, hit in enumerate(hits, start=1)]

    def search_text(self, query: str, **kwargs) -> list[ParentStageResult]:
        """Embed ``query`` with the injected model and search."""
        vector = self.retriever.embed_query(query)
        return self.search(vector, **kwargs)

    # ------------------------------------------------------------------
    @staticmethod
    def _to_result(hit: SearchResult, rank: int) -> ParentStageResult:
        """Turn a backend hit into a parent stage result."""
        metadata = dict(hit.metadata or {})
        child_ids = tuple(metadata.get("child_ids") or ())
        child_count = int(metadata.get("child_count") or len(child_ids))
        return ParentStageResult(
            chunk_id=hit.chunk_id,
            doc_id=hit.doc_id,
            text=hit.text or "",
            score=hit.score,
            rank=rank,
            metadata=metadata,
            provenance=hit.provenance,
            child_count=child_count,
            child_ids=child_ids,
        )


__all__ = ["LEVEL_PARENT", "ParentStage", "ParentStageResult"]
