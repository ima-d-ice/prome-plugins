"""Tests for the fixed-size chunker."""

from itertools import pairwise

import pytest

from chunkers.contracts import ApproxTokenCounter, ChunkerConfigError
from chunkers.fixed import FixedSizeChunker, FixedSizeConfig
from parsers.contracts import BlockType, ParsedBlock, ParsedDocument, ParsedPage

from .helpers import assert_provenance, coverage_stats, flattened_text


def _doc(blocks, source="synthetic.html"):
    return ParsedDocument(source=source, parser_name="test", parser_version="1", blocks=blocks)


def _long_text(words: int) -> str:
    return " ".join(f"word{i}" for i in range(words))


def test_empty_document():
    chunker = FixedSizeChunker()
    assert chunker.chunk(ParsedDocument(source="e", parser_name="t", parser_version="1")) == []


def test_short_document_single_chunk():
    doc = _doc([ParsedBlock(BlockType.TEXT, "short text")])
    chunks = FixedSizeChunker().chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "short text"
    assert chunks[0].count == 1
    assert chunks[0].length == len("short text")
    assert chunks[0].length_unit == "characters"


def test_chunk_size_limit_respected_on_long_document():
    doc = _doc([ParsedBlock(BlockType.TEXT, _long_text(2000))])
    chunks = FixedSizeChunker(FixedSizeConfig(chunk_size=200, overlap=50)).chunk(doc)
    assert len(chunks) > 3
    for chunk in chunks:
        assert chunk.length <= 200


def test_unbreakable_text_hard_cut():
    doc = _doc([ParsedBlock(BlockType.TEXT, "x" * 2500)])
    chunks = FixedSizeChunker(FixedSizeConfig(chunk_size=1000, overlap=100)).chunk(doc)
    assert [c.length for c in chunks] == [1000, 1000, 700]


def test_exact_overlap_without_word_snapping():
    doc = _doc([ParsedBlock(BlockType.TEXT, "a" * 1000)])
    chunks = FixedSizeChunker(FixedSizeConfig(chunk_size=400, overlap=100, respect_word_boundaries=True)).chunk(doc)
    for prev, nxt in pairwise(chunks):
        assert nxt.metadata["char_start"] == prev.metadata["char_end"] - 100


def test_overlap_repeats_content():
    doc = _doc([ParsedBlock(BlockType.TEXT, _long_text(400))])
    chunks = FixedSizeChunker(FixedSizeConfig(chunk_size=100, overlap=30)).chunk(doc)
    assert len(chunks) > 2
    for prev, nxt in pairwise(chunks):
        assert nxt.metadata["char_start"] < prev.metadata["char_end"]
        assert prev.metadata["char_end"] - nxt.metadata["char_start"] <= 100  # overlap <= chunk_size


def test_zero_overlap_produces_disjoint_chunks():
    doc = _doc([ParsedBlock(BlockType.TEXT, _long_text(300))])
    chunks = FixedSizeChunker(FixedSizeConfig(chunk_size=100, overlap=0)).chunk(doc)
    for prev, nxt in pairwise(chunks):
        assert nxt.metadata["char_start"] >= prev.metadata["char_end"]


def test_word_boundaries_respected():
    doc = _doc([ParsedBlock(BlockType.TEXT, _long_text(600))])
    text = flattened_text(doc)
    chunks = FixedSizeChunker(FixedSizeConfig(chunk_size=120, overlap=20)).chunk(doc)
    for chunk in chunks:
        start = chunk.metadata["char_start"]
        if start > 0:
            assert text[start - 1].isspace()


def test_word_boundary_snapping_can_be_disabled():
    doc = _doc([ParsedBlock(BlockType.TEXT, _long_text(500))])
    chunks = FixedSizeChunker(FixedSizeConfig(chunk_size=50, overlap=0, respect_word_boundaries=False)).chunk(doc)
    lengths = [c.length for c in chunks]
    assert max(lengths) == 50


def test_content_coverage_full():
    doc = _doc([ParsedBlock(BlockType.TEXT, _long_text(500))])
    chunks = FixedSizeChunker(FixedSizeConfig(chunk_size=100, overlap=25)).chunk(doc)
    stats = coverage_stats(doc, chunks)
    assert stats["coverage"] == pytest.approx(1.0)


def test_no_empty_chunks():
    doc = _doc([ParsedBlock(BlockType.TEXT, " " * 50 + "real content here")])
    chunks = FixedSizeChunker(FixedSizeConfig(chunk_size=20, overlap=5)).chunk(doc)
    assert all(c.text.strip() for c in chunks)


def test_provenance_block_refs():
    doc = _doc(
        [
            ParsedBlock(BlockType.TEXT, "alpha " * 100),
            ParsedBlock(BlockType.TEXT, "beta " * 100),
        ]
    )
    chunks = FixedSizeChunker(FixedSizeConfig(chunk_size=150, overlap=30)).chunk(doc)
    assert_provenance(doc, chunks)


def test_metadata_preserved():
    doc = ParsedDocument(
        source="meta.html",
        parser_name="test-parser",
        parser_version="9.9",
        metadata={"title": "Title"},
        pages=[ParsedPage(page_number=3, blocks=[ParsedBlock(BlockType.TEXT, "page three content " * 30, page_number=3)])],
    )
    chunks = FixedSizeChunker(FixedSizeConfig(chunk_size=100, overlap=10)).chunk(doc)
    for chunk in chunks:
        assert chunk.metadata["source"] == "meta.html"
        assert chunk.metadata["parser_name"] == "test-parser"
        assert chunk.metadata["parser_version"] == "9.9"
        assert chunk.metadata["strategy"] == "fixed"
        assert 3 in chunk.metadata["pages"]
    assert_provenance(doc, chunks)


def test_chunk_text_matches_metadata_range():
    doc = _doc([ParsedBlock(BlockType.TEXT, _long_text(300))])
    text = flattened_text(doc)
    chunks = FixedSizeChunker(FixedSizeConfig(chunk_size=80, overlap=10)).chunk(doc)
    for chunk in chunks:
        assert text[chunk.metadata["char_start"] : chunk.metadata["char_end"]] == chunk.text


def test_tokenizer_based_sizing():
    doc = _doc([ParsedBlock(BlockType.TEXT, _long_text(400))])
    counter = ApproxTokenCounter()
    chunks = FixedSizeChunker(FixedSizeConfig(chunk_size=50, overlap=10, tokenizer=counter)).chunk(doc)
    assert chunks
    for chunk in chunks:
        assert chunk.length <= 50
        assert chunk.length_unit == "approx_tokens"
        assert chunk.length == counter.count(chunk.text)


def test_plain_callable_tokenizer():
    doc = _doc([ParsedBlock(BlockType.TEXT, _long_text(300))])
    chunks = FixedSizeChunker(FixedSizeConfig(chunk_size=10, overlap=2, tokenizer=lambda s: len(s.split()))).chunk(doc)
    assert chunks
    assert all(len(c.text.split()) <= 10 for c in chunks)
    assert all(c.length_unit == "tokens" for c in chunks)


def test_unicode_content():
    doc = _doc([ParsedBlock(BlockType.TEXT, "héllo wörld ✨ " * 100)])
    chunks = FixedSizeChunker(FixedSizeConfig(chunk_size=100, overlap=20)).chunk(doc)
    stats = coverage_stats(doc, chunks)
    assert stats["coverage"] == pytest.approx(1.0)
    for chunk in chunks:
        assert "✨" in chunk.text or stats["total_chunk_chars"] > 0


def test_multiple_pages_preserve_page_numbers():
    doc = ParsedDocument(
        source="pages.pdf",
        parser_name="test",
        parser_version="1",
        pages=[
            ParsedPage(page_number=1, blocks=[ParsedBlock(BlockType.TEXT, "page one " * 50, page_number=1)]),
            ParsedPage(page_number=2, blocks=[ParsedBlock(BlockType.TEXT, "page two " * 50, page_number=2)]),
        ],
    )
    chunks = FixedSizeChunker(FixedSizeConfig(chunk_size=100, overlap=20)).chunk(doc)
    referenced_pages = {ref.page_number for chunk in chunks for ref in chunk.block_refs}
    assert referenced_pages == {1, 2}
    assert_provenance(doc, chunks)


def test_deterministic_repeated_runs():
    doc = _doc([ParsedBlock(BlockType.HEADING, "H", metadata={"heading_level": 1}), ParsedBlock(BlockType.TEXT, _long_text(400))])
    first = [c.as_dict() for c in FixedSizeChunker(FixedSizeConfig(chunk_size=100, overlap=10)).chunk(doc)]
    second = [c.as_dict() for c in FixedSizeChunker(FixedSizeConfig(chunk_size=100, overlap=10)).chunk(doc)]
    assert first == second


def test_count_and_index_consistency():
    doc = _doc([ParsedBlock(BlockType.TEXT, _long_text(500))])
    chunks = FixedSizeChunker(FixedSizeConfig(chunk_size=100, overlap=10)).chunk(doc)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.count == len(chunks) for c in chunks)


def test_invalid_config_rejected():
    with pytest.raises(ChunkerConfigError):
        FixedSizeConfig(chunk_size=10, overlap=10)
    with pytest.raises(ChunkerConfigError):
        FixedSizeConfig(chunk_size=-1, overlap=0)
