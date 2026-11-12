"""Shared data models of the Prome RAG retrieval ecosystem.

Every retrieval strategy (dense, hybrid, hierarchical, ...) answers with the
same :class:`RetrievalResult`, so a downstream system can swap one strategy for
another without changing its code.

The models reuse the upstream contracts instead of duplicating them:

- :class:`chunkers.contracts.models.Chunk` identity and provenance are carried
  by the vector-store contracts (:class:`~vector_stores.contracts.models.ChunkProvenance`);
- the score direction of a hit comes from the vector-store contract
  (:class:`~vector_stores.contracts.models.ScoreDirection`), because a cosine
  similarity is not comparable with a euclidean distance.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from vector_stores.contracts.models import ChunkProvenance, ScoreDirection


@dataclass(frozen=True)
class ContextChunk:
    """A chunk attached to a result as *context* rather than as the match.

    A hierarchical result carries the matched child (the evidence) plus the
    parent section that contains it; an adjacent-child expansion adds the
    neighbouring children of the same parent.  All of them are context: they
    explain the match, they are not the match.
    """

    chunk_id: str
    text: str
    role: str
    """Role of the context chunk (``"parent"`` or ``"adjacent"``)."""

    doc_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: ChunkProvenance | None = None


@dataclass(frozen=True)
class RetrievalResult:
    """One ranked retrieval hit.

    The ``chunk_id``/``text`` of a result are always the **evidence**: the
    chunk that actually matched the query.  Context (the enclosing parent
    section, neighbouring chunks) is kept separately in ``context`` so a
    caller can never confuse the two.
    """

    chunk_id: str
    doc_id: str
    text: str
    score: float
    rank: int
    strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: ChunkProvenance | None = None
    score_direction: ScoreDirection | None = None
    context: tuple[ContextChunk, ...] = ()

    @property
    def parent_context(self) -> ContextChunk | None:
        """Return the parent-section context, when the strategy attached one."""
        for item in self.context:
            if item.role == "parent":
                return item
        return None

    @property
    def adjacent_context(self) -> tuple[ContextChunk, ...]:
        """Return the adjacent-chunk context, when the strategy attached any."""
        return tuple(item for item in self.context if item.role == "adjacent")


@dataclass(frozen=True)
class RetrievalDiagnostics:
    """Non-score diagnostics a strategy reports alongside its results.

    They describe *how* the answer was produced (how many candidates were
    searched, how many parents were selected, ...).  They never influence the
    ranking and are never used to fabricate a result.

    ``candidates_total`` may be given as a callable, in which case the total
    is resolved only when it is read.  Counting the records of a collection
    is not free, and a caller that does not look at the diagnostics should not
    pay for it.
    """

    candidates_searched: int = 0
    parents_selected: int = 0
    selected_parent_ids: tuple[str, ...] = ()
    """Ids of the parent sections the strategy selected, in selection order."""
    notes: tuple[str, ...] = ()
    _candidates_total: int | Callable[[], int] | None = None

    @property
    def candidates_total(self) -> int:
        """Return the number of candidates that existed before the search."""
        total = self._candidates_total
        if callable(total):
            return total()
        return int(total or 0)

    @property
    def candidate_reduction_ratio(self) -> float:
        """Fraction of the candidate space that was *not* searched.

        ``1.0`` means nothing was skipped; ``0.0`` means every candidate was
        searched.  Returns ``0.0`` when the total is unknown.
        """
        total = self.candidates_total
        if total <= 0:
            return 0.0
        searched = max(0, min(self.candidates_searched, total))
        return 1.0 - (searched / total)


@dataclass
class RetrievalResponse:
    """The answer of a retrieval strategy: the hits plus the diagnostics."""

    results: list[RetrievalResult] = field(default_factory=list)
    diagnostics: RetrievalDiagnostics = field(default_factory=RetrievalDiagnostics)

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, index: int) -> RetrievalResult:
        return self.results[index]
