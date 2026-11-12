"""Tests of the adjacent-chunk merging."""

from __future__ import annotations

import pytest

from retrieval.contracts.models import RetrievedItem
from retrieval.postprocessing.adjacent_merge import (
    AdjacentMergeConfig,
    AdjacentMerger,
)

from tests.retrieval.support import make_chunk


def item(chunk_id: str, *, rank: int = 1, text: str | None = None) -> RetrievedItem:
    """Build a retrieved item for the tests."""
    return RetrievedItem(
        chunk_id=chunk_id,
        doc_id="doc.md",
        text=text if text is not None else f"text of {chunk_id}",
        score=1.0,
        rank=rank,
        strategy="test",
    )


@pytest.fixture()
def corpus():
    """Return three consecutive chunks of one document and one of another."""
    return {
        "c0": make_chunk("c0", "doc.md", "text of c0", index=0),
        "c1": make_chunk("c1", "doc.md", "text of c1", index=1),
        "c2": make_chunk("c2", "doc.md", "text of c2", index=2),
        "x0": make_chunk("x0", "other.md", "other document", index=0),
    }


def test_the_merging_is_disabled_by_default(corpus):
    """Without the option the items are returned unchanged."""
    merger = AdjacentMerger(chunks=corpus)

    merged, report = merger.merge([item("c0")])

    assert merged[0].text == "text of c0"
    assert report.merged == 0


def test_the_next_chunk_is_merged(corpus):
    """The chunk that follows the hit is merged into it."""
    merger = AdjacentMerger(
        chunks=corpus, config=AdjacentMergeConfig(enabled=True, max_merged_chars=1000)
    )

    merged, report = merger.merge([item("c0")])

    assert "text of c0" in merged[0].text
    assert "text of c1" in merged[0].text
    assert report.merged == 1


def test_the_previous_chunk_is_not_merged_by_default(corpus):
    """Only the following chunk is merged unless it is asked for."""
    merger = AdjacentMerger(
        chunks=corpus, config=AdjacentMergeConfig(enabled=True, max_merged_chars=1000)
    )

    merged, _ = merger.merge([item("c1")])

    assert "text of c1" in merged[0].text
    assert "text of c0" not in merged[0].text


def test_the_previous_chunk_can_be_included(corpus):
    """The preceding chunk is merged when the option is set."""
    merger = AdjacentMerger(
        chunks=corpus,
        config=AdjacentMergeConfig(
            enabled=True, include_previous=True, max_merged_chars=1000
        ),
    )

    merged, _ = merger.merge([item("c1")])

    assert "text of c0" in merged[0].text
    assert "text of c1" in merged[0].text


def test_the_chunk_ids_of_the_contributors_are_kept(corpus):
    """The merged item records the chunks that contributed to it."""
    merger = AdjacentMerger(
        chunks=corpus, config=AdjacentMergeConfig(enabled=True, max_merged_chars=1000)
    )

    merged, _ = merger.merge([item("c0")])

    assert merged[0].chunk_id == "c0"
    assert merged[0].metadata["merged_chunk_ids"] == ["c0", "c1"]


def test_the_evidence_reference_is_kept(corpus):
    """The merged item keeps the id of the chunk it was built from."""
    merger = AdjacentMerger(
        chunks=corpus, config=AdjacentMergeConfig(enabled=True, max_merged_chars=1000)
    )

    merged, _ = merger.merge([item("c1", rank=4)])

    assert merged[0].chunk_id == "c1"
    assert merged[0].rank == 4


def test_chunks_of_another_document_are_never_merged(corpus):
    """A chunk of another document is not merged."""
    merger = AdjacentMerger(
        chunks=corpus, config=AdjacentMergeConfig(enabled=True, max_merged_chars=1000)
    )
    # x0 is the only chunk of other.md, at index 0; c0 is at index 0 of doc.md.
    foreign = RetrievedItem(
        chunk_id="x0", doc_id="other.md", text="other document", score=1.0, rank=1
    )

    merged, report = merger.merge([foreign])

    assert merged[0].text == "other document"
    assert report.merged == 0
    assert report.skipped == 1


def test_the_size_limit_is_respected(corpus):
    """A merged text that would exceed the limit is not merged."""
    merger = AdjacentMerger(
        chunks=corpus, config=AdjacentMergeConfig(enabled=True, max_merged_chars=3)
    )

    merged, report = merger.merge([item("c0")])

    assert merged[0].text == "text of c0"
    assert report.merged == 0
    assert report.skipped == 1
    assert report.notes


def test_a_chunk_without_a_neighbour_is_not_merged(corpus):
    """The last chunk of a document has no neighbour to merge with."""
    merger = AdjacentMerger(
        chunks=corpus, config=AdjacentMergeConfig(enabled=True, max_merged_chars=1000)
    )

    merged, report = merger.merge([item("c2")])

    assert merged[0].text == "text of c2"
    assert report.merged == 0


def test_the_merging_is_configurable(corpus):
    """The merging can be turned on and off."""
    config = AdjacentMergeConfig(enabled=True)
    merger = AdjacentMerger(chunks=corpus, config=config)

    merged, _ = merger.merge([item("c0")])

    assert len(merged[0].text) > len("text of c0")


def test_an_invalid_size_limit_is_rejected():
    """A limit below one is a configuration error."""
    with pytest.raises(ValueError):
        AdjacentMergeConfig(max_merged_chars=0)


def test_an_invalid_chunk_limit_is_rejected():
    """A chunk limit below one is a configuration error."""
    with pytest.raises(ValueError):
        AdjacentMergeConfig(max_merged_chunks=0)


def test_the_merger_without_a_corpus_returns_the_items():
    """Without the corpus of the corpus no neighbour can be found."""
    merger = AdjacentMerger(config=AdjacentMergeConfig(enabled=True))

    merged, report = merger.merge([item("c0")])

    assert merged[0].text == "text of c0"
    assert report.merged == 0
