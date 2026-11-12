"""Tests of the deduplication of retrieved items."""

from __future__ import annotations

from retrieval.contracts.models import RetrievedItem
from retrieval.postprocessing.deduplication import deduplicate


def item(
    chunk_id: str,
    *,
    rank: int = 1,
    score: float = 0.0,
    methods: tuple[str, ...] = ("dense",),
    channel_ranks: dict[str, int] | None = None,
    channel_scores: dict[str, float] | None = None,
    text: str | None = None,
) -> RetrievedItem:
    """Build a retrieved item for the tests."""
    return RetrievedItem(
        chunk_id=chunk_id,
        doc_id="doc.md",
        text=text if text is not None else f"text of {chunk_id}",
        score=score,
        rank=rank,
        strategy="test",
        methods=methods,
        channel_ranks=channel_ranks or {m: rank for m in methods},
        channel_scores=channel_scores or {m: score for m in methods},
    )


def test_the_items_are_deduplicated_by_chunk_id():
    """A chunk that appears twice appears once in the result."""
    items = [item("a"), item("b"), item("a", rank=3)]

    result = deduplicate(items)

    assert [i.chunk_id for i in result] == ["a", "b"]


def test_the_best_ranked_occurrence_is_kept():
    """The best ranked occurrence of a chunk is the one that is kept."""
    items = [item("a", rank=5), item("a", rank=1)]

    result = deduplicate(items)

    assert len(result) == 1
    assert result[0].rank == 1


def test_the_channel_information_is_preserved():
    """The channels that also returned the chunk are recorded."""
    items = [
        item("a", rank=1, methods=("dense",), channel_ranks={"dense": 1}),
        item("a", rank=2, methods=("bm25",), channel_ranks={"bm25": 4}),
    ]

    result = deduplicate(items)

    assert len(result) == 1
    assert set(result[0].methods) == {"dense", "bm25"}
    assert result[0].channel_ranks == {"dense": 1, "bm25": 4}


def test_the_channel_scores_are_preserved():
    """The score of every channel is kept."""
    items = [
        item("a", rank=1, channel_scores={"dense": 0.9}),
        item("a", rank=2, channel_scores={"bm25": -2.5}),
    ]

    result = deduplicate(items)

    assert result[0].channel_scores == {"dense": 0.9, "bm25": -2.5}


def test_two_different_chunks_with_the_same_text_are_kept():
    """Chunks are never merged because their text happens to be equal."""
    items = [item("a", text="the same text"), item("b", text="the same text")]

    result = deduplicate(items)

    assert [i.chunk_id for i in result] == ["a", "b"]


def test_the_order_is_preserved():
    """The order of the items is preserved."""
    items = [item("a", rank=1), item("b", rank=2), item("c", rank=3)]

    result = deduplicate(items)

    assert [i.chunk_id for i in result] == ["a", "b", "c"]


def test_an_empty_list_returns_an_empty_list():
    """Deduplicating nothing returns nothing."""
    assert deduplicate([]) == []


def test_a_single_item_is_returned_unchanged():
    """An item that occurs once is not modified."""
    items = [item("a", rank=1)]

    result = deduplicate(items)

    assert result == items
