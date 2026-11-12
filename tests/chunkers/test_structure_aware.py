"""Tests for the structure-aware chunker."""

import pytest

from chunkers.contracts import ApproxTokenCounter, ChunkerConfigError
from chunkers.structure_aware import StructureAwareChunker, StructureAwareConfig
from parsers.contracts import BlockType, ParsedBlock, ParsedDocument, ParsedPage

from .helpers import assert_provenance, coverage_stats, flattened_text


def _doc(blocks, source="synthetic.html"):
    return ParsedDocument(source=source, parser_name="test", parser_version="1", blocks=blocks)


def _heading(text, level, **block_kwargs):
    return ParsedBlock(BlockType.HEADING, text, metadata={"heading_level": level}, **block_kwargs)


def _text(text):
    return ParsedBlock(BlockType.TEXT, text)


def test_empty_document():
    chunker = StructureAwareChunker()
    assert chunker.chunk(ParsedDocument(source="e", parser_name="t", parser_version="1")) == []


def test_short_document_single_chunk():
    doc = _doc([_heading("Intro", 1), _text("body")])
    chunks = StructureAwareChunker().chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].heading == "Intro"
    assert chunks[0].metadata["heading_path"] == ["Intro"]
    assert chunks[0].metadata["heading_level"] == 1
    assert chunks[0].metadata["block_types"] == ["heading", "text"]


def test_major_section_boundary_closes_chunk():
    doc = _doc(
        [
            _heading("Section A", 1),
            _text("alpha body"),
            _heading("Section B", 1),
            _text("beta body"),
        ]
    )
    chunks = StructureAwareChunker().chunk(doc)
    assert len(chunks) == 2
    assert chunks[0].text == "Section A\n\nalpha body"
    assert chunks[1].text == "Section B\n\nbeta body"
    assert chunks[1].heading == "Section B"


def test_same_level_headings_split_but_deeper_glue_on():
    doc = _doc(
        [
            _heading("Top", 1),
            _text("top body"),
            _heading("Nested", 2),
            _text("nested body"),
            _heading("Sibling", 2),
            _text("sibling body"),
        ]
    )
    chunks = StructureAwareChunker().chunk(doc)
    # Nested (h2) stays with Top; Sibling (h2) opens a new chunk within Top.
    assert chunks[0].text.startswith("Top")
    assert "Nested" in chunks[0].text
    assert chunks[1].text.startswith("Sibling")
    assert chunks[1].metadata["heading_path"] == ["Top", "Sibling"]


def test_lower_level_heading_does_not_close():
    doc = _doc([_heading("A", 1), _text("a body"), _heading("A.1", 2), _text("a1 body")])
    chunks = StructureAwareChunker().chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].metadata["heading_path"] == ["A"]


def test_headings_never_glued_to_previous_section():
    doc = _doc([_heading("One", 1), _text("one body"), _heading("Two", 1), _text("two body")])
    chunks = StructureAwareChunker().chunk(doc)
    for chunk in chunks:
        assert not chunk.text.rstrip().endswith(("One", "Two"))


def test_unleveled_headings_treated_as_level_one():
    blocks = [
        ParsedBlock(BlockType.HEADING, "DocTitle"),  # no heading_level metadata
        _text("body text"),
        ParsedBlock(BlockType.HEADING, "NextSection"),
        _text("more text"),
    ]
    chunks = StructureAwareChunker().chunk(_doc(blocks))
    assert len(chunks) == 2
    assert chunks[0].text.startswith("DocTitle")
    assert chunks[1].text.startswith("NextSection")


def test_heading_glued_to_following_content_when_overflow():
    doc = _doc([_heading("Big Section", 1), _text(" ".join(["word"] * 800))])
    chunks = StructureAwareChunker(StructureAwareConfig(chunk_size=200)).chunk(doc)
    assert chunks[0].text.startswith("Big Section")
    # The heading must never appear as the sole content of a chunk while
    # content follows it in the document.
    for index, chunk in enumerate(chunks[:-1]):
        assert chunk.text.strip() != "Big Section" or chunks[index + 1].heading == "Big Section"


def test_table_kept_atomic():
    table = "Item | Qty\n--- | ---\nWidget | 3\nGadget | 7\nGizmo | 12"
    doc = _doc([_heading("Results", 1), _text("intro"), ParsedBlock(BlockType.TABLE, table), _text("outro")])
    chunks = StructureAwareChunker().chunk(doc)
    table_chunks = [c for c in chunks if "Widget | 3" in c.text]
    assert len(table_chunks) == 1
    assert table in table_chunks[0].text


def test_oversized_table_split_but_all_content_retained():
    table = "Item | Qty\n" + "\n".join(f"row{i} | {i}" for i in range(400))
    doc = _doc([ParsedBlock(BlockType.TABLE, table)])
    chunks = StructureAwareChunker(StructureAwareConfig(chunk_size=200)).chunk(doc)
    assert len(chunks) > 1
    joined = "\n".join(c.text for c in chunks)
    for i in range(400):
        assert f"row{i} | {i}" in joined
    table_refs = {
        ref.block_index for c in chunks for ref in c.block_refs if ref.block_type == "table"
    }
    assert table_refs == {0}


def test_list_kept_atomic_when_it_fits():
    lst = "- alpha\n- beta\n- gamma\n- delta"
    doc = _doc([_heading("List Section", 1), ParsedBlock(BlockType.LIST, lst)])
    chunks = StructureAwareChunker().chunk(doc)
    assert any(lst in c.text for c in chunks)


def test_oversized_list_split_by_items():
    lst = "\n".join(f"- item number {i} with some extra words" for i in range(200))
    doc = _doc([ParsedBlock(BlockType.LIST, lst)])
    chunks = StructureAwareChunker(StructureAwareConfig(chunk_size=150)).chunk(doc)
    joined = "\n".join(c.text for c in chunks)
    for i in range(200):
        assert f"item number {i}" in joined


def test_page_and_block_references_preserved_multipage():
    doc = ParsedDocument(
        source="multi.pdf",
        parser_name="test",
        parser_version="1",
        pages=[
            ParsedPage(page_number=1, blocks=[_heading("Page1", 1, page_number=1), _text("page one body")]),
            ParsedPage(page_number=2, blocks=[_heading("Page2", 1, page_number=2), _text("page two body")]),
        ],
    )
    chunks = StructureAwareChunker().chunk(doc)
    assert [c.text.split("\n\n")[0] for c in chunks] == ["Page1", "Page2"]
    for chunk in chunks:
        pages = {ref.page_number for ref in chunk.block_refs}
        assert chunk.metadata["pages"] == sorted(pages)
    assert_provenance(doc, chunks)


def test_document_order_preserved():
    doc = _doc(
        [_heading("A", 1), _text("alpha"), _heading("B", 1), _text("beta"), _heading("C", 1), _text("gamma")]
    )
    chunks = StructureAwareChunker().chunk(doc)
    order = [c.text.split("\n\n")[0] for c in chunks]
    assert order == ["A", "B", "C"]


def test_full_content_coverage():
    blocks = []
    for i in range(3):
        blocks.append(_heading(f"Section {i}", 1))
        blocks.append(_text("body " + " ".join([f"w{i}_{j}" for j in range(80)])))
    doc = _doc(blocks)
    chunks = StructureAwareChunker(StructureAwareConfig(chunk_size=150)).chunk(doc)
    assert coverage_stats(doc, chunks)["coverage"] == pytest.approx(1.0)


def test_heading_path_tracks_open_sections():
    doc = _doc(
        [
            _heading("A", 1),
            _heading("A1", 2),
            _heading("A1a", 3),
            _text("deep body"),
        ]
    )
    chunks = StructureAwareChunker().chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].metadata["heading_path"] == ["A", "A1", "A1a"]
    assert chunks[0].heading == "A1a"


def test_heading_path_resets_at_same_level_boundary():
    doc = _doc(
        [
            _heading("A", 1),
            _heading("A1", 2),
            _text("a1 body"),
            _heading("A2", 2),
            _text("a2 body"),
        ]
    )
    chunks = StructureAwareChunker().chunk(doc)
    paths = [c.metadata["heading_path"] for c in chunks]
    assert paths[0] == ["A", "A1"]
    assert paths[-1] == ["A", "A2"]


def test_unicode_content():
    doc = _doc([_heading("Ünïcödé", 1), _text("héllo ✨ " * 100)])
    chunks = StructureAwareChunker(StructureAwareConfig(chunk_size=120)).chunk(doc)
    assert coverage_stats(doc, chunks)["coverage"] == pytest.approx(1.0)
    assert chunks[0].metadata["heading_path"] == ["Ünïcödé"]


def test_string_heading_level_accepted():
    doc = _doc([ParsedBlock(BlockType.HEADING, "Tit", metadata={"heading_level": "2"}), _text("body")])
    chunks = StructureAwareChunker().chunk(doc)
    assert chunks[0].metadata["heading_level"] == 2


def test_close_on_heading_disabled_keeps_sections_together():
    doc = _doc([_heading("A", 1), _text("alpha"), _heading("B", 1), _text("beta")])
    chunks = StructureAwareChunker(StructureAwareConfig(close_on_heading=False)).chunk(doc)
    assert len(chunks) == 1


def test_tokenizer_based_sizing():
    doc = _doc([_heading("S", 1), _text(" ".join(["word"] * 400))])
    counter = ApproxTokenCounter()
    chunks = StructureAwareChunker(StructureAwareConfig(chunk_size=60, tokenizer=counter)).chunk(doc)
    assert chunks
    assert all(c.length <= 60 for c in chunks)
    assert all(c.length_unit == "approx_tokens" for c in chunks)


def test_deterministic_repeated_runs():
    doc = _doc([_heading("A", 1), _text(" ".join(["word"] * 300)), _heading("B", 1), _text("more words")])
    cfg = StructureAwareConfig(chunk_size=150)
    first = [c.as_dict() for c in StructureAwareChunker(cfg).chunk(doc)]
    second = [c.as_dict() for c in StructureAwareChunker(cfg).chunk(doc)]
    assert first == second


def test_count_and_index_consistency():
    doc = _doc([_heading("A", 1), _text(" ".join(["word"] * 500))])
    chunks = StructureAwareChunker(StructureAwareConfig(chunk_size=100)).chunk(doc)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.count == len(chunks) for c in chunks)


def test_chunk_text_matches_metadata_range():
    doc = _doc([_heading("A", 1), _text("body " * 100)])
    text = flattened_text(doc)
    for chunk in StructureAwareChunker(StructureAwareConfig(chunk_size=80)).chunk(doc):
        assert text[chunk.metadata["char_start"] : chunk.metadata["char_end"]] == chunk.text


def test_provenance_block_refs():
    doc = _doc([_heading("A", 1), _text("alpha " * 60), _heading("B", 1), _text("beta " * 60)])
    chunks = StructureAwareChunker(StructureAwareConfig(chunk_size=120)).chunk(doc)
    assert_provenance(doc, chunks)


def test_table_metadata_retained_in_block_refs():
    table = ParsedBlock(BlockType.TABLE, "Item | Qty\nWidget | 3", metadata={"headers": ["Item", "Qty"], "rows": 2})
    doc = _doc([table])
    chunks = StructureAwareChunker().chunk(doc)
    assert any(ref.block_type == "table" for c in chunks for ref in c.block_refs)


def test_invalid_config_rejected():
    with pytest.raises(ChunkerConfigError):
        StructureAwareConfig(chunk_size=0)
    with pytest.raises(ChunkerConfigError):
        StructureAwareConfig(tokenizer=123)  # type: ignore[arg-type]


def test_chunk_size_limit_with_nested_heading_bridge():
    # Open h1 section nearly fills the limit; an h2 arrives (glued, not a
    # boundary), then a text block would overflow — the close must consider
    # the already accumulated content, not just pending headings.
    doc = _doc(
        [
            _heading("H1", 1),
            _text(" ".join(["alpha"] * 150)),   # ~900 chars
            _heading("H2", 2),
            _text(" ".join(["beta"] * 60)),     # another ~360 chars
        ]
    )
    chunks = StructureAwareChunker(StructureAwareConfig(chunk_size=1000)).chunk(doc)
    for chunk in chunks:
        assert chunk.length <= 1000, chunk.length


def test_chunk_length_invariant_on_fixture_like_mixed_content():
    blocks = []
    for i in range(6):
        blocks.append(_heading(f"Section {i}", 1))
        blocks.append(_text("body " + " ".join([f"w{j}" for j in range(120)])))
    doc = _doc(blocks)
    for size in (100, 300, 1000):
        chunks = StructureAwareChunker(StructureAwareConfig(chunk_size=size)).chunk(doc)
        for chunk in chunks:
            assert chunk.length <= size, (size, chunk.length)
        assert coverage_stats(doc, chunks)["coverage"] == pytest.approx(1.0)
