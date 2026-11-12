"""Reciprocal Rank Fusion.

``RRF_score(d) = sum over the systems that retrieved d of 1 / (k + rank(d))``
with ``rank`` starting at 1 and ``k`` defaulting to 60.  Only the ranks are
used, so the raw scores of two channels are never compared.
"""

from retrieval.fusion.rrf.rrf import (
    DEFAULT_K,
    FusedItem,
    RankedList,
    fuse,
    reciprocal_rank_fusion,
    to_retrieved_items,
)

__all__ = [
    "DEFAULT_K",
    "FusedItem",
    "RankedList",
    "fuse",
    "reciprocal_rank_fusion",
    "to_retrieved_items",
]
