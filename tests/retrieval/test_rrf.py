"""Tests of the reciprocal rank fusion.

The tests use synthetic ranked lists, so the behaviour of the fusion is
verified without a store, a model or a database.
"""

from __future__ import annotations

import pytest

from retrieval.contracts.models import RetrievedItem
from retrieval.fusion.rrf import (
    DEFAULT_K,
    RankedList,
    fuse,
    reciprocal_rank_fusion,
    to_retrieved_items,
)
from vector_stores.contracts.models import ScoreDirection


def item(chunk_id: str, score: float = 0.0, *, rank: int = 1) -> RetrievedItem:
    """Build a retrieved item for the tests."""
    return RetrievedItem(
        chunk_id=chunk_id,
        doc_id="doc.md",
        text=f"text of {chunk_id}",
        score=score,
        rank=rank,
        strategy="test",
        metadata={"origin": chunk_id},
    )


def test_the_score_is_the_sum_of_the_reciprocals():
    """The fused score is the sum of 1 / (k + rank) over the channels."""
    lists = [
        RankedList(channel="dense", items=[item("a", 0.9), item("b", 0.8)]),
        RankedList(channel="bm25", items=[item("b", -1.0), item("c", -2.0)]),
    ]

    fused = reciprocal_rank_fusion(lists, k=60)
    scores = {entry.item.chunk_id: entry.score for entry in fused}

    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 62)


def test_the_rank_starts_at_one():
    """The first item of a channel contributes 1 / (k + 1)."""
    lists = [RankedList(channel="dense", items=[item("a")])]

    fused = reciprocal_rank_fusion(lists, k=60)

    assert fused[0].score == pytest.approx(1 / 61)


def test_the_default_constant_is_sixty():
    """The default damping constant is 60."""
    assert DEFAULT_K == 60


def test_a_custom_constant_is_used():
    """A different constant changes the fused score."""
    lists = [RankedList(channel="dense", items=[item("a")])]

    fused = reciprocal_rank_fusion(lists, k=1)

    assert fused[0].score == pytest.approx(1 / 2)


def test_an_invalid_constant_is_rejected():
    """A constant below one is a configuration error."""
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], k=0)


def test_the_items_are_deduplicated_by_chunk_id():
    """A chunk that both channels returned appears once."""
    lists = [
        RankedList(channel="dense", items=[item("a"), item("b")]),
        RankedList(channel="bm25", items=[item("b"), item("c")]),
    ]

    fused = reciprocal_rank_fusion(lists)

    assert sorted(entry.item.chunk_id for entry in fused) == ["a", "b", "c"]


def test_the_content_of_the_item_is_preserved():
    """The fused item keeps the content and the metadata of the chunk."""
    lists = [RankedList(channel="dense", items=[item("a", 0.5)])]

    fused = reciprocal_rank_fusion(lists)

    assert fused[0].item.text == "text of a"
    assert fused[0].item.metadata == {"origin": "a"}


def test_the_provenance_of_the_item_is_preserved():
    """The fused item keeps the provenance of the chunk."""
    from vector_stores.contracts.models import ChunkProvenance

    provenance = ChunkProvenance(source="doc.md", pages=(1,))
    original = RetrievedItem(
        chunk_id="a",
        doc_id="doc.md",
        text="text",
        score=0.5,
        rank=1,
        provenance=provenance,
    )
    lists = [RankedList(channel="dense", items=[original])]

    fused = reciprocal_rank_fusion(lists)

    assert fused[0].item.provenance is provenance


def test_the_channel_ranks_are_recorded():
    """The rank the item had in each channel is recorded."""
    lists = [
        RankedList(channel="dense", items=[item("a", 0.9), item("b", 0.8)]),
        RankedList(channel="bm25", items=[item("b", -1.0)]),
    ]

    fused = {entry.item.chunk_id: entry for entry in reciprocal_rank_fusion(lists)}

    assert fused["a"].channel_ranks == {"dense": 1}
    assert fused["b"].channel_ranks == {"dense": 2, "bm25": 1}


def test_the_channel_scores_are_recorded():
    """The score the item had in each channel is recorded."""
    lists = [
        RankedList(channel="dense", items=[item("a", 0.9)]),
        RankedList(channel="bm25", items=[item("a", -3.5)]),
    ]

    fused = reciprocal_rank_fusion(lists)

    assert fused[0].channel_scores == {"dense": 0.9, "bm25": -3.5}


def test_the_channels_that_found_the_item_are_recorded():
    """The channels that retrieved the item are recorded."""
    lists = [
        RankedList(channel="dense", items=[item("a")]),
        RankedList(channel="bm25", items=[item("a"), item("b")]),
    ]

    fused = {entry.item.chunk_id: entry for entry in reciprocal_rank_fusion(lists)}

    assert fused["a"].methods == ("dense", "bm25")
    assert fused["b"].methods == ("bm25",)


def test_an_item_of_a_single_channel_is_not_discarded():
    """A chunk that only one channel found keeps its contribution."""
    lists = [
        RankedList(channel="dense", items=[item("a")]),
        RankedList(channel="bm25", items=[item("z")]),
    ]

    fused = {entry.item.chunk_id: entry for entry in reciprocal_rank_fusion(lists)}

    assert "z" in fused
    assert fused["z"].score == pytest.approx(1 / 61)


def test_a_chunk_found_by_both_channels_wins():
    """A chunk that both channels found outranks a chunk only one found."""
    lists = [
        RankedList(channel="dense", items=[item("shared", 0.9), item("dense-only", 0.8)]),
        RankedList(channel="bm25", items=[item("shared", -1.0)]),
    ]

    fused = reciprocal_rank_fusion(lists)

    assert fused[0].item.chunk_id == "shared"


def test_the_tie_is_broken_deterministically():
    """Two items with the same fused score are ordered by their chunk id."""
    lists = [
        RankedList(channel="dense", items=[item("b")]),
        RankedList(channel="bm25", items=[item("a")]),
    ]

    fused = reciprocal_rank_fusion(lists)

    # Both are at rank one of their own channel, so they are tied.
    assert [entry.item.chunk_id for entry in fused] == ["a", "b"]


def test_the_ranks_are_consecutive():
    """The fused items are numbered from one."""
    lists = [
        RankedList(channel="dense", items=[item("a"), item("b")]),
        RankedList(channel="bm25", items=[item("c")]),
    ]

    fused = reciprocal_rank_fusion(lists)

    assert [entry.rank for entry in fused] == [1, 2, 3]


def test_the_limit_truncates_the_result():
    """Only the requested number of fused items is returned."""
    lists = [RankedList(channel="dense", items=[item("a"), item("b"), item("c")])]

    fused = reciprocal_rank_fusion(lists, limit=2)

    assert len(fused) == 2


def test_the_channel_weights_scale_the_contribution():
    """A weighted channel contributes proportionally more."""
    lists = [
        RankedList(channel="dense", items=[item("a")], weight=2.0),
        RankedList(channel="bm25", items=[item("b")], weight=1.0),
    ]

    fused = {entry.item.chunk_id: entry for entry in reciprocal_rank_fusion(lists)}

    assert fused["a"].score == pytest.approx(2 / 61)
    assert fused["b"].score == pytest.approx(1 / 61)


def test_a_weighted_channel_wins_over_an_unweighted_one():
    """The weighted channel decides the order."""
    lists = [
        RankedList(channel="dense", items=[item("dense-first")], weight=2.0),
        RankedList(channel="bm25", items=[item("bm25-first")], weight=1.0),
    ]

    fused = reciprocal_rank_fusion(lists)

    assert fused[0].item.chunk_id == "dense-first"


def test_the_fused_score_is_a_similarity():
    """The fused score is bigger when it is better."""
    lists = [RankedList(channel="dense", items=[item("a")])]

    items = to_retrieved_items(reciprocal_rank_fusion(lists), strategy="hybrid")

    assert items[0].score_direction is ScoreDirection.HIGHER_IS_BETTER
    assert items[0].score_interpretation == "rrf"


def test_the_fused_items_carry_the_channel_information():
    """The standardized items keep the channel ranks and scores."""
    lists = [
        RankedList(channel="dense", items=[item("a", 0.9)]),
        RankedList(channel="bm25", items=[item("a", -2.0)]),
    ]

    items = fuse(lists, strategy="hybrid")

    assert items[0].methods == ("dense", "bm25")
    assert items[0].channel_ranks == {"dense": 1, "bm25": 1}
    assert items[0].strategy == "hybrid"


def test_the_raw_scores_are_never_compared():
    """A BM25 score and a cosine score are not added together."""
    lists = [
        RankedList(channel="dense", items=[item("a", 0.99)]),
        RankedList(channel="bm25", items=[item("b", -99.0)]),
    ]

    fused = {entry.item.chunk_id: entry for entry in reciprocal_rank_fusion(lists)}

    # Both contribute 1 / (k + rank); the raw scores play no part.
    assert fused["a"].score == pytest.approx(fused["b"].score)


def test_an_empty_fusion_returns_nothing():
    """Fusing nothing returns nothing."""
    assert reciprocal_rank_fusion([]) == []
