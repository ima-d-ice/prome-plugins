"""Reciprocal Rank Fusion.

RRF combines ranked lists that do not share a score scale.  The score of a
document is the sum of the reciprocals of its ranks:

.. code-block:: text

    RRF_score(d) = sum over the systems that retrieved d of  1 / (k + rank(d))

``rank(d)`` starts at **1**, and ``k`` defaults to **60**.  The constant
damps the influence of the very top ranks, so a document that one channel
loves and another ignores does not automatically win.

Only the *ranks* are used.  The raw scores of the channels are never
compared with each other, because a cosine similarity and a BM25 score do
not share a scale.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from retrieval.contracts.models import RetrievedItem
from vector_stores.contracts.models import ScoreDirection

#: The default damping constant of the reciprocal rank function.
DEFAULT_K = 60


@dataclass(frozen=True)
class RankedList:
    """One ranked list of a fusion.

    Attributes:
        channel: the name of the channel that produced the list.
        items: the items, best first.
        weight: the weight of the channel in the fused score.
    """

    channel: str
    items: Sequence[RetrievedItem]
    weight: float = 1.0


@dataclass
class FusedItem:
    """One item of a fused result."""

    item: RetrievedItem
    """The item, with the content of the first channel that returned it."""

    score: float
    """The fused score."""

    rank: int = 0
    channel_ranks: dict[str, int] = field(default_factory=dict)
    channel_scores: dict[str, float] = field(default_factory=dict)
    methods: tuple[str, ...] = ()


def reciprocal_rank_fusion(
    lists: Sequence[RankedList],
    *,
    k: int = DEFAULT_K,
    limit: int | None = None,
) -> list[FusedItem]:
    """Fuse ``lists`` with the reciprocal rank function.

    The items are deduplicated by their chunk id.  The content and the
    provenance of the item are the ones of the **first** channel that
    returned it; the ranks and the scores of every channel are recorded.

    An item that only one channel retrieved keeps that channel's
    contribution -- it is never discarded.

    Args:
        lists: the ranked lists to fuse.
        k: the damping constant of the reciprocal rank function.
        limit: how many fused items to return.

    Returns:
        The fused items, best first.  The tie is broken by the chunk id, so
        the order is deterministic.
    """
    if k < 1:
        raise ValueError(f"the RRF constant k must be >= 1, got {k}")

    best: dict[str, FusedItem] = {}
    order: list[str] = []
    for ranked in lists:
        for position, item in enumerate(ranked.items, start=1):
            rank = position
            contribution = ranked.weight / (k + rank)
            existing = best.get(item.chunk_id)
            if existing is None:
                existing = FusedItem(
                    item=item,
                    score=0.0,
                    channel_ranks={},
                    channel_scores={},
                    methods=(),
                )
                best[item.chunk_id] = existing
                order.append(item.chunk_id)
            existing.score += contribution
            existing.channel_ranks[ranked.channel] = rank
            existing.channel_scores[ranked.channel] = item.score
            if ranked.channel not in existing.methods:
                existing.methods = existing.methods + (ranked.channel,)

    fused = [best[chunk_id] for chunk_id in order]
    fused.sort(key=lambda entry: (-entry.score, entry.item.chunk_id))
    for rank, entry in enumerate(fused, start=1):
        entry.rank = rank
    if limit is not None:
        fused = fused[:limit]
    return fused


def to_retrieved_items(
    fused: Sequence[FusedItem],
    *,
    strategy: str,
    score_interpretation: str = "rrf",
) -> list[RetrievedItem]:
    """Turn the fused items into standardized retrieval items.

    The fused score is the sum of reciprocals, so a larger score is better.
    """
    items: list[RetrievedItem] = []
    for entry in fused:
        source = entry.item
        items.append(
            RetrievedItem(
                chunk_id=source.chunk_id,
                doc_id=source.doc_id,
                text=source.text,
                score=entry.score,
                score_direction=ScoreDirection.HIGHER_IS_BETTER,
                score_interpretation=score_interpretation,
                rank=entry.rank,
                strategy=strategy,
                metadata=dict(source.metadata),
                provenance=source.provenance,
                methods=entry.methods,
                channel_ranks=dict(entry.channel_ranks),
                channel_scores=dict(entry.channel_scores),
                context=source.context,
            )
        )
    return items


def fuse(
    lists: Sequence[RankedList],
    *,
    k: int = DEFAULT_K,
    limit: int | None = None,
    strategy: str = "hybrid",
) -> list[RetrievedItem]:
    """Fuse ``lists`` and return the standardized items, best first."""
    fused = reciprocal_rank_fusion(lists, k=k, limit=limit)
    return to_retrieved_items(fused, strategy=strategy)
