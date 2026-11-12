"""Shared data models of the Prome RAG retrieval ecosystem.

Every retrieval strategy (dense, hybrid, hierarchical, ...) answers with the
same :class:`RetrievalResponse`, so a downstream system can swap one strategy
for another without changing its code.

The models reuse the upstream contracts instead of duplicating them:

- :class:`chunkers.contracts.models.Chunk` identity and provenance are carried
  by the vector-store contracts (:class:`~vector_stores.contracts.models.ChunkProvenance`);
- the score direction of a hit comes from the vector-store contract
  (:class:`~vector_stores.contracts.models.ScoreDirection`), because a cosine
  similarity is not comparable with a euclidean distance and a BM25 score is
  not comparable with either;
- the metadata filter of a query is the vector-store filter AST
  (:class:`~vector_stores.contracts.filters.MetadataFilter`), so the dense and
  the lexical channel understand the *same* filter.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from vector_stores.contracts.filters import MetadataFilter
from vector_stores.contracts.models import ChunkProvenance, ScoreDirection


@dataclass(frozen=True)
class RetrievalQuery:
    """A retrieval request.

    The same query object drives every channel, so a dense search, a lexical
    search and a hybrid search always apply the *same* restrictions.

    Attributes:
        text: the query text.
        query_id: an optional identifier of the query, echoed in the response.
        limit: how many results the caller wants.
        dense_limit: how many candidates the dense channel may return.
            Defaults to ``limit`` when it is not set.
        lexical_limit: how many candidates the lexical channel may return.
            Defaults to ``limit`` when it is not set.
        filters: the metadata filter every channel must apply.
        doc_ids: optional restriction to a set of documents.
        tenant_id: optional restriction to a tenant.
        params: free-form search parameters passed to the channels.
    """

    text: str
    query_id: str | None = None
    limit: int = 10
    dense_limit: int | None = None
    lexical_limit: int | None = None
    filters: MetadataFilter | None = None
    doc_ids: tuple[str, ...] | None = None
    tenant_id: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError(f"the query text must be a string, got {type(self.text).__name__}")
        if not isinstance(self.limit, int) or isinstance(self.limit, bool) or self.limit < 1:
            raise ValueError(f"the query limit must be an integer >= 1, got {self.limit!r}")
        for name in ("dense_limit", "lexical_limit"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1 or None, got {value!r}")

    @property
    def effective_dense_limit(self) -> int:
        """Return the candidate limit of the dense channel."""
        return self.dense_limit if self.dense_limit is not None else self.limit

    @property
    def effective_lexical_limit(self) -> int:
        """Return the candidate limit of the lexical channel."""
        return self.lexical_limit if self.lexical_limit is not None else self.limit


@dataclass(frozen=True)
class ContextChunk:
    """A chunk attached to a result as *context* rather than as the match.

    A hierarchical result carries the matched child (the evidence) plus the
    parent section that contains it; an adjacent-chunk expansion adds the
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
class RetrievedItem:
    """One retrieved passage.

    The ``chunk_id``/``text`` of an item are always the **evidence**: the
    chunk that actually matched.  Context (the enclosing parent section, the
    neighbouring chunks, the merged neighbours) is kept separately in
    ``context`` so a caller can never confuse the two.

    Attributes:
        chunk_id: the original chunk id, never a backend-specific one.
        doc_id: the document the chunk belongs to.
        text: the text of the chunk.
        score: the score of the item *in the scale of its channel*.
        score_direction: whether a larger or a smaller score is better.
        score_interpretation: what the score means (``"cosine"``, ``"bm25"``,
            ``"rrf"``, ...).  Two scores of different interpretations are not
            comparable.
        rank: the 1-based rank of the item in the channel that produced it.
        strategy: the strategy that produced the item.
        metadata: the metadata of the chunk.
        provenance: the page and block provenance of the chunk.
        methods: the retrieval channels that returned the item.
        channel_ranks: the rank the item had in each channel.
        channel_scores: the score the item had in each channel.
        context: the contextual chunks attached to the item.
    """

    chunk_id: str
    doc_id: str
    text: str
    score: float
    score_direction: ScoreDirection = ScoreDirection.HIGHER_IS_BETTER
    score_interpretation: str = "unknown"
    rank: int = 1
    strategy: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: ChunkProvenance | None = None
    methods: tuple[str, ...] = ()
    channel_ranks: Mapping[str, int] = field(default_factory=dict)
    channel_scores: Mapping[str, float] = field(default_factory=dict)
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


#: Backwards-compatible alias of :class:`RetrievedItem`.
#:
#: The name was changed when the retrieval contracts were aligned with the
#: vocabulary of the retrieval ecosystem; the alias keeps the earlier import
#: path working.
RetrievalResult = RetrievedItem


@dataclass(frozen=True)
class RetrievalDiagnostics:
    """Non-score diagnostics a strategy reports alongside its results.

    They describe *how* the answer was produced (how many candidates were
    searched, which channel failed, ...).  They never influence the ranking and
    are never used to fabricate a result.

    ``candidates_total`` may be given as a callable, in which case the total
    is resolved only when it is read.  Counting the records of a collection is
    not free, and a caller that does not look at the diagnostics should not
    pay for it.
    """

    candidates_searched: int = 0
    channel_candidates: Mapping[str, int] = field(default_factory=dict)
    """Number of candidates each channel produced."""
    channel_failures: Mapping[str, str] = field(default_factory=dict)
    """The channels that failed, and why."""
    parents_selected: int = 0
    selected_parent_ids: tuple[str, ...] = ()
    """Ids of the parent sections the strategy selected, in selection order."""
    notes: tuple[str, ...] = ()
    _candidates_total: int | Callable[[], int] | None = None

    @property
    def candidates_total(self) -> int:
        """Return the number of candidates that existed before the search.

        The total may be given as a callable, in which case it is resolved
        only when it is read: counting the records of a collection is not
        free, and a caller that does not look at the diagnostics should not
        pay for it.
        """
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
    """The answer of a retrieval strategy: the hits plus the diagnostics.

    Attributes:
        results: the ordered results, best first.
        query: the query the response answers.
        strategy: the strategy that produced the response.
        diagnostics: how the answer was produced.
        timings_ms: the wall-clock time of each channel, when it was measured.
    """

    results: list[RetrievedItem] = field(default_factory=list)
    query: RetrievalQuery | None = None
    strategy: str = ""
    diagnostics: RetrievalDiagnostics = field(default_factory=RetrievalDiagnostics)
    timings_ms: Mapping[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, index: int) -> RetrievedItem:
        return self.results[index]

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        """Return the ids of the results, in order."""
        return tuple(item.chunk_id for item in self.results)


def merge_documents(
    results: Sequence[RetrievedItem],
) -> list[RetrievedItem]:
    """Return the results ordered by rank, then by chunk id.

    Two backends do not agree on the order of equal scores, so the ordering
    is made deterministic here: the rank comes first, the chunk id breaks the
    tie.
    """
    return sorted(results, key=lambda item: (item.rank, item.chunk_id))
