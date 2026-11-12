"""Fusion of several retrieval channels.

A fusion combines ranked lists that do not share a score scale.
"""

from retrieval.fusion.rrf import (
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
