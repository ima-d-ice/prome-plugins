"""Deduplication of retrieved items.

Two channels can return the same chunk: the dense search and the lexical
search may both rank it highly.  The deduplication keeps **one** entry per
chunk id -- the best ranked one -- and records that the other channels also
returned it, so the information is not lost.

Two *different* chunks are never merged, even when their text happens to be
identical: they are different pieces of evidence with different ids.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence

from retrieval.contracts.models import RetrievedItem


def deduplicate(items: Iterable[RetrievedItem]) -> list[RetrievedItem]:
    """Remove the repeated chunk ids, keeping the best ranked occurrence.

    The order of the remaining items is preserved, and the channels that also
    returned a kept item are merged into its ``methods`` /
    ``channel_ranks`` / ``channel_scores``, so a caller can still see that
    the item was found by more than one channel.

    Args:
        items: the items to deduplicate.

    Returns:
        The deduplicated items, in their original order.
    """
    best: dict[str, RetrievedItem] = {}
    order: list[str] = []
    for item in items:
        existing = best.get(item.chunk_id)
        if existing is None:
            best[item.chunk_id] = item
            order.append(item.chunk_id)
            continue
        # Keep the better ranked occurrence and merge the channel
        # information of the other one into it.
        keep, drop = (
            (item, existing)
            if (item.rank, item.chunk_id) < (existing.rank, existing.chunk_id)
            else (existing, item)
        )
        best[item.chunk_id] = dataclasses.replace(
            keep,
            methods=_merge_methods(keep.methods, drop.methods),
            channel_ranks={**drop.channel_ranks, **keep.channel_ranks},
            channel_scores={**drop.channel_scores, **keep.channel_scores},
        )
    return [best[chunk_id] for chunk_id in order]


def _merge_methods(
    first: Sequence[str], second: Sequence[str]
) -> tuple[str, ...]:
    """Return the union of two method tuples, in a stable order."""
    merged = list(first)
    for method in second:
        if method not in merged:
            merged.append(method)
    return tuple(merged)
