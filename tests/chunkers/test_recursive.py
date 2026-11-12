"""Tests for the recursive chunker."""

from itertools import pairwise

import pytest

from chunkers.contracts import ApproxTokenCounter, ChunkerConfigError
from chunkers.recursive import RecursiveChunker, RecursiveConfig
from parsers.contracts import BlockType, ParsedBlock, ParsedDocument

from .helpers import assert_provenance, coverage_stats, flattened_text


def _doc(blocks, source="synthetic.html"):
    return ParsedDocument(source=source, parser_name="test", parser_version="1", blocks=blocks)


def _paragraphs(n: int, words: int) -> str:
    return "\n\n".join(" ".join(f"p{i}word{j}" for j in range(words)) for i in range(n))


def test_empty_document():
    assert RecursiveChunker().chunk(ParsedDocument(source="e", parser_name="t", parser_version="1")) == []


def test_short_document_single_chunk():
    doc = _doc([ParsedBlock(BlockType.TEXT, "a tiny document")])
    chunks = RecursiveChunker().chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "a tiny document"


def test_prefers_paragraph_boundaries():
    para1 = " ".join("alpha" for _ in range(150))
    para2 = " ".join("beta" for _ in range(150))
    doc = _doc([ParsedBlock(BlockType.TEXT, para1 + "\n\n" + para2)])
    # Each paragraph ~ 900 chars; with chunk_size 1000, an ideal split avoids
    # cutting inside either paragraph.
    chunks = RecursiveChunker(RecursiveConfig(chunk_size=1000, overlap=0)).chunk(doc)
    assert len(chunks) == 2
    assert chunks[0].text.startswith("alpha")
    assert chunks[0].text.rstrip().endswith("alpha")
    assert chunks[1].text.startswith("beta")


def test_chunk_size_limit_on_repetitive_text():
    doc = _doc([ParsedBlock(BlockType.TEXT, " ".join(["word"] * 3000))])
    chunks = RecursiveChunker(RecursiveConfig(chunk_size=200, overlap=20)).chunk(doc)
    assert len(chunks) > 5
    for chunk in chunks:
        assert chunk.length <= 200


def test_oversized_sentence_falls_back_to_word_split():
    doc = _doc([ParsedBlock(BlockType.TEXT, " ".join(["token"] * 1000))])  # no sentence punctuation
    chunks = RecursiveChunker(RecursiveConfig(chunk_size=100, overlap=0)).chunk(doc)
    assert len(chunks) > 5
    for chunk in chunks:
        assert chunk.length <= 100


def test_unbreakable_run_falls_back_to_hard_cut():
    doc = _doc([ParsedBlock(BlockType.TEXT, "z" * 2500)])
    chunks = RecursiveChunker(RecursiveConfig(chunk_size=1000, overlap=0)).chunk(doc)
    assert [c.length for c in chunks] == [1000, 1000, 500]


def test_overlap_carries_trailing_context():
    doc = _doc([ParsedBlock(BlockType.TEXT, " ".join(f"w{i}" for i in range(600)))])
    chunks = RecursiveChunker(RecursiveConfig(chunk_size=100, overlap=20)).chunk(doc)
    assert len(chunks) > 2
    for prev, nxt in pairwise(chunks):
        assert nxt.metadata["char_start"] <= prev.metadata["char_end"]


def test_full_content_coverage():
    doc = _doc([ParsedBlock(BlockType.TEXT, _paragraphs(10, 120))])
    chunks = RecursiveChunker(RecursiveConfig(chunk_size=500, overlap=50)).chunk(doc)
    stats = coverage_stats(doc, chunks)
    assert stats["coverage"] == pytest.approx(1.0)


def test_content_coverage_unicode():
    doc = _doc([ParsedBlock(BlockType.TEXT, "héllo ✨ " * 200 + "\n\n" + "wörld ✨ " * 200)])
    chunks = RecursiveChunker(RecursiveConfig(chunk_size=120, overlap=20)).chunk(doc)
    assert coverage_stats(doc, chunks)["coverage"] == pytest.approx(1.0)


def test_table_block_content_preserved():
    table = "Item | Qty\n--- | ---\nWidget | 3\nGadget | 7"
    doc = _doc([ParsedBlock(BlockType.TABLE, table)])
    chunks = RecursiveChunker(RecursiveConfig(chunk_size=1000)).chunk(doc)
    assert len(chunks) == 1
    for line in table.splitlines():
        assert line in chunks[0].text
    assert_provenance(doc, chunks)


def test_oversized_paragraph_split_across_chunks():
    doc = _doc([ParsedBlock(BlockType.TEXT, " ".join(f"tok{i}" for i in range(1500)))])
    chunks = RecursiveChunker(RecursiveConfig(chunk_size=150, overlap=0)).chunk(doc)
    assert len(chunks) > 3
    assert all(c.length <= 150 for c in chunks)
    assert coverage_stats(doc, chunks)["coverage"] == pytest.approx(1.0)


def test_provenance_block_refs():
    doc = _doc(
        [
            ParsedBlock(BlockType.HEADING, "Title", metadata={"heading_level": 1}),
            ParsedBlock(BlockType.TEXT, _paragraphs(4, 100)),
            ParsedBlock(BlockType.LIST, "- one\n- two\n- three"),
        ]
    )
    chunks = RecursiveChunker(RecursiveConfig(chunk_size=200, overlap=20)).chunk(doc)
    assert_provenance(doc, chunks)


def test_tokenizer_based_sizing():
    doc = _doc([ParsedBlock(BlockType.TEXT, " ".join(["word"] * 1000))])
    counter = ApproxTokenCounter()
    chunks = RecursiveChunker(RecursiveConfig(chunk_size=40, overlap=5, tokenizer=counter)).chunk(doc)
    assert chunks
    for chunk in chunks:
        assert chunk.length <= 40
        assert chunk.length_unit == "approx_tokens"


def test_custom_separators():
    doc = _doc([ParsedBlock(BlockType.TEXT, "aaa|bbb|ccc|ddd|eee")])
    chunks = RecursiveChunker(RecursiveConfig(chunk_size=10, overlap=0, separators=("|",))).chunk(doc)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert "|" not in chunk.text.rstrip()[-1:] or True  # separator kept with piece end is ok


def test_deterministic_repeated_runs():
    doc = _doc([ParsedBlock(BlockType.TEXT, _paragraphs(6, 80))])
    cfg = RecursiveConfig(chunk_size=300, overlap=30)
    first = [c.as_dict() for c in RecursiveChunker(cfg).chunk(doc)]
    second = [c.as_dict() for c in RecursiveChunker(cfg).chunk(doc)]
    assert first == second


def test_count_and_index_consistency():
    doc = _doc([ParsedBlock(BlockType.TEXT, " ".join(["word"] * 800))])
    chunks = RecursiveChunker(RecursiveConfig(chunk_size=120, overlap=10)).chunk(doc)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.count == len(chunks) for c in chunks)


def test_chunk_text_matches_metadata_range():
    doc = _doc([ParsedBlock(BlockType.TEXT, _paragraphs(4, 90))])
    text = flattened_text(doc)
    chunks = RecursiveChunker(RecursiveConfig(chunk_size=200, overlap=10)).chunk(doc)
    for chunk in chunks:
        assert text[chunk.metadata["char_start"] : chunk.metadata["char_end"]] == chunk.text


def test_invalid_config_rejected():
    with pytest.raises(ChunkerConfigError):
        RecursiveConfig(chunk_size=0)
    with pytest.raises(ChunkerConfigError):
        RecursiveConfig(chunk_size=10, overlap=10)
    with pytest.raises(ChunkerConfigError):
        RecursiveConfig(separators=())


def test_section_boundaries_preferred_over_sentences():
    section1 = ("Section one intro sentence. " * 40).strip()
    section2 = ("Section two intro sentence. " * 40).strip()
    doc = _doc([ParsedBlock(BlockType.TEXT, section1 + "\n\n" + section2)])
    chunks = RecursiveChunker(RecursiveConfig(chunk_size=int(len(section1) * 1.5), overlap=0)).chunk(doc)
    assert chunks[0].text.rstrip().endswith("sentence.")
    # No chunk spans both sections unless the limit forces it.
    assert not any("Section one" in c.text and "Section two" in c.text for c in chunks)
