"""Tests for the hierarchical (parent/child) chunker."""

from itertools import pairwise

import pytest

from chunkers.contracts import ApproxTokenCounter, ChunkerConfigError
from chunkers.hierarchical import (
    HierarchicalChunker,
    HierarchicalConfig,
    HierarchicalResult,
)
from parsers.contracts import BlockType, ParsedBlock, ParsedDocument, ParsedPage

from .helpers import assert_provenance, coverage_stats, flattened_text


def _doc(blocks, source="synthetic.html"):
    return ParsedDocument(source=source, parser_name="test", parser_version="1", blocks=blocks)


def _heading(text, level, **block_kwargs):
    return ParsedBlock(BlockType.HEADING, text, metadata={"heading_level": level}, **block_kwargs)


def _text(text):
    return ParsedBlock(BlockType.TEXT, text)


def _words(count: int, prefix: str = "word") -> str:
    return " ".join(f"{prefix}{i}" for i in range(count))


# ---------------------------------------------------------------------------
# Structure of the result
# ---------------------------------------------------------------------------
def test_empty_document():
    result = HierarchicalChunker().chunk(ParsedDocument(source="e", parser_name="t", parser_version="1"))
    assert isinstance(result, HierarchicalResult)
    assert result.parents == []
    assert result.children == []


def test_result_has_parents_and_children():
    doc = _doc([_heading("A", 1), _text(_words(200))])
    result = HierarchicalChunker(HierarchicalConfig(parent_size=1000, child_size=100, child_overlap=0)).chunk(doc)
    assert result.parents
    assert result.children
    assert all(isinstance(c.text, str) for c in result.parents + result.children)


def test_short_document_smaller_than_parent_size():
    doc = _doc([_text("a tiny document")])
    result = HierarchicalChunker(HierarchicalConfig(parent_size=2000, child_size=500, child_overlap=0)).chunk(doc)
    assert len(result.parents) == 1
    assert len(result.children) == 1
    assert result.children[0].text == "a tiny document"
    assert result.parents[0].text == "a tiny document"


def test_document_within_parent_size_yields_single_parent():
    doc = _doc([_heading("A", 1), _text(_words(50))])
    result = HierarchicalChunker(HierarchicalConfig(parent_size=2000, child_size=100)).chunk(doc)
    assert len(result.parents) == 1


# ---------------------------------------------------------------------------
# Parent/child associations
# ---------------------------------------------------------------------------
def test_every_child_has_a_valid_parent():
    doc = _doc(
        [
            _heading("A", 1),
            _text(_words(200, "alpha")),
            _heading("B", 1),
            _text(_words(200, "beta")),
        ]
    )
    result = HierarchicalChunker(HierarchicalConfig(parent_size=1000, child_size=100, child_overlap=0)).chunk(doc)
    parent_ids = {parent.id for parent in result.parents}
    assert result.children
    for child in result.children:
        assert child.metadata["parent_id"] in parent_ids


def test_no_orphaned_children():
    doc = _doc(
        [
            _heading("A", 1),
            _text(_words(200, "alpha")),
            _heading("B", 1),
            _text(_words(200, "beta")),
        ]
    )
    result = HierarchicalChunker(HierarchicalConfig(parent_size=1000, child_size=100, child_overlap=0)).chunk(doc)
    child_ids = {child.id for child in result.children}
    for child in result.children:
        assert child.metadata["parent_id"] is not None
        assert child.id in child_ids
    # Every parent lists exactly its own children.
    for parent in result.parents:
        assert parent.metadata["child_ids"]
        assert set(parent.metadata["child_ids"]) <= child_ids
        assert parent.metadata["child_count"] == len(parent.metadata["child_ids"])


def test_every_parent_has_at_least_one_child():
    doc = _doc(
        [
            _heading("A", 1),
            _text(_words(120, "alpha")),
            _heading("B", 1),
            _text(_words(120, "beta")),
        ]
    )
    result = HierarchicalChunker(HierarchicalConfig(parent_size=800, child_size=60, child_overlap=0)).chunk(doc)
    assert result.parents
    for parent in result.parents:
        assert len(parent.metadata["child_ids"]) >= 1


def test_parent_child_relationship_is_bidirectional():
    doc = _doc([_heading("A", 1), _text(_words(300, "alpha"))])
    result = HierarchicalChunker(HierarchicalConfig(parent_size=1000, child_size=80, child_overlap=0)).chunk(doc)
    parent = result.parents[0]
    children = result.children_of(parent)
    assert len(children) == len(parent.metadata["child_ids"])
    for child in children:
        assert result.parent_of(child).id == parent.id
    assert result.parent_of("missing-id") is None
    assert result.children_of("missing-id") == []


def test_children_are_contained_in_their_parent():
    doc = _doc(
        [
            _heading("A", 1),
            _text(_words(150, "alpha")),
            _heading("B", 1),
            _text(_words(150, "beta")),
        ]
    )
    result = HierarchicalChunker(HierarchicalConfig(parent_size=1000, child_size=100, child_overlap=0)).chunk(doc)
    for parent in result.parents:
        for child in result.children_of(parent):
            assert child.metadata["char_start"] >= parent.metadata["char_start"]
            assert child.metadata["char_end"] <= parent.metadata["char_end"]
            assert child.text in parent.text


def test_children_partition_their_parent_content():
    doc = _doc([_heading("A", 1), _text(_words(200, "alpha"))])
    result = HierarchicalChunker(HierarchicalConfig(parent_size=1000, child_size=60, child_overlap=0)).chunk(doc)
    parent = result.parents[0]
    children = result.children_of(parent)
    # The children tile the parent: ordered, non-overlapping spans whose
    # gaps are only the whitespace trimmed by the shared chunk assembly.
    assert children[0].metadata["char_start"] == parent.metadata["char_start"]
    assert children[-1].metadata["char_end"] == parent.metadata["char_end"]
    text = flattened_text(doc)
    for previous, following in pairwise(children):
        assert previous.metadata["char_end"] <= following.metadata["char_start"]
        gap = text[previous.metadata["char_end"] : following.metadata["char_start"]]
        assert not gap.strip()
    joined = "".join(child.text for child in children)
    assert "".join(joined.split()) == "".join(parent.text.split())


# ---------------------------------------------------------------------------
# Size limits
# ---------------------------------------------------------------------------
def test_parent_size_limit_respected():
    doc = _doc(
        [
            _heading("A", 1),
            _text(_words(400, "alpha")),
            _heading("B", 1),
            _text(_words(400, "beta")),
        ]
    )
    result = HierarchicalChunker(HierarchicalConfig(parent_size=300, child_size=100, child_overlap=0)).chunk(doc)
    for parent in result.parents:
        assert parent.length <= 300


def test_child_size_limit_respected():
    doc = _doc(
        [
            _heading("A", 1),
            _text(_words(400, "alpha")),
            _heading("B", 1),
            _text(_words(400, "beta")),
        ]
    )
    result = HierarchicalChunker(HierarchicalConfig(parent_size=1000, child_size=120, child_overlap=0)).chunk(doc)
    for child in result.children:
        assert child.length <= 120


def test_child_size_limit_respected_for_large_document():
    blocks = []
    for i in range(8):
        blocks.append(_heading(f"Section {i}", 1))
        blocks.append(_text(_words(120, f"w{i}")))
    doc = _doc(blocks)
    result = HierarchicalChunker(HierarchicalConfig(parent_size=600, child_size=90, child_overlap=0)).chunk(doc)
    assert len(result.children) > 10
    for child in result.children:
        assert child.length <= 90


def test_tokenizer_based_sizing():
    doc = _doc([_heading("A", 1), _text(_words(400))])
    counter = ApproxTokenCounter()
    result = HierarchicalChunker(
        HierarchicalConfig(parent_size=120, child_size=40, child_overlap=0, tokenizer=counter)
    ).chunk(doc)
    assert result.children
    for child in result.children:
        assert child.length <= 40
        assert child.length_unit == "approx_tokens"


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------
def test_child_overlap_repeats_content():
    doc = _doc([_text(_words(400, "w"))])
    without = HierarchicalChunker(
        HierarchicalConfig(parent_size=1000, child_size=100, child_overlap=0)
    ).chunk(doc)
    with_overlap = HierarchicalChunker(
        HierarchicalConfig(parent_size=1000, child_size=100, child_overlap=40)
    ).chunk(doc)
    assert len(with_overlap.children) >= len(without.children)
    assert coverage_stats(doc, with_overlap.children)["duplicate_rate"] > 0.0
    assert coverage_stats(doc, without.children)["duplicate_rate"] == 0.0


def test_child_overlap_keeps_content_covered():
    doc = _doc([_text(_words(300, "w"))])
    result = HierarchicalChunker(
        HierarchicalConfig(parent_size=1000, child_size=80, child_overlap=30)
    ).chunk(doc)
    assert coverage_stats(doc, result.children)["coverage"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Section boundaries
# ---------------------------------------------------------------------------
def test_major_section_boundaries_respected_in_parents():
    doc = _doc(
        [
            _heading("Section A", 1),
            _text("alpha body"),
            _heading("Section B", 1),
            _text("beta body"),
        ]
    )
    result = HierarchicalChunker(HierarchicalConfig(parent_size=1000, child_size=100, child_overlap=0)).chunk(doc)
    assert len(result.parents) == 2
    assert result.parents[0].text == "Section A\n\nalpha body"
    assert result.parents[1].text == "Section B\n\nbeta body"


def test_parents_preserve_heading_context():
    doc = _doc(
        [
            _heading("Introduction", 1),
            _text(_words(100, "intro")),
            _heading("Conclusion", 1),
            _text(_words(100, "outro")),
        ]
    )
    result = HierarchicalChunker(HierarchicalConfig(parent_size=1000, child_size=60)).chunk(doc)
    headings = [parent.heading for parent in result.parents]
    assert "Introduction" in headings
    assert "Conclusion" in headings
    for parent in result.parents:
        assert parent.metadata["heading_path"]


def test_children_inherit_parent_heading_context():
    doc = _doc([_heading("Introduction", 1), _text(_words(200, "intro"))])
    result = HierarchicalChunker(HierarchicalConfig(parent_size=1000, child_size=60)).chunk(doc)
    assert result.children
    for child in result.children:
        assert child.heading == "Introduction"
        assert child.metadata["heading_path"] == ["Introduction"]


def test_children_do_not_cross_section_boundaries():
    doc = _doc(
        [
            _heading("Section A", 1),
            _text(_words(150, "alpha")),
            _heading("Section B", 1),
            _text(_words(150, "beta")),
        ]
    )
    result = HierarchicalChunker(HierarchicalConfig(parent_size=1000, child_size=50, child_overlap=0)).chunk(doc)
    for child in result.children:
        # A child never contains the start of the *next* major section.
        assert "Section A" not in child.text or "Section B" not in child.text


# ---------------------------------------------------------------------------
# Provenance and metadata
# ---------------------------------------------------------------------------
def test_source_document_id_preserved():
    doc = _doc([_text("body")], source="my-document.pdf")
    result = HierarchicalChunker().chunk(doc)
    for chunk in result.parents + result.children:
        assert chunk.doc_id == "my-document.pdf"
        assert chunk.metadata["source"] == "my-document.pdf"
        assert chunk.metadata["parser_name"] == "test"
        assert chunk.metadata["parser_version"] == "1"


def test_block_references_preserved():
    doc = _doc(
        [
            _heading("A", 1),
            _text(_words(200, "alpha")),
            _heading("B", 1),
            _text(_words(200, "beta")),
        ]
    )
    result = HierarchicalChunker(HierarchicalConfig(parent_size=1000, child_size=100, child_overlap=0)).chunk(doc)
    assert_provenance(doc, result.parents)
    assert_provenance(doc, result.children)


def test_page_references_preserved_multipage():
    doc = ParsedDocument(
        source="multi.pdf",
        parser_name="test",
        parser_version="1",
        pages=[
            ParsedPage(page_number=1, blocks=[_heading("P1", 1, page_number=1), _text("page one body")]),
            ParsedPage(page_number=2, blocks=[_heading("P2", 1, page_number=2), _text("page two body")]),
        ],
    )
    result = HierarchicalChunker(HierarchicalConfig(parent_size=1000, child_size=50, child_overlap=0)).chunk(doc)
    for chunk in result.parents + result.children:
        pages = {ref.page_number for ref in chunk.block_refs}
        assert chunk.metadata["pages"] == sorted(pages)
    assert {ref.page_number for chunk in result.children for ref in chunk.block_refs} == {1, 2}


def test_metadata_retained_on_children():
    doc = _doc([_heading("A", 1), _text(_words(100))])
    result = HierarchicalChunker(HierarchicalConfig(parent_size=1000, child_size=50, child_overlap=0)).chunk(doc)
    for child in result.children:
        assert child.metadata["strategy"] == "hierarchical"
        assert child.metadata["level"] == "child"
        assert child.metadata["parent_id"]
        assert child.metadata["parent_index"] >= 0
        assert child.metadata["child_index"] >= 0
        assert child.metadata["block_count"] >= 1
    for parent in result.parents:
        assert parent.metadata["strategy"] == "hierarchical"
        assert parent.metadata["level"] == "parent"


def test_result_as_dict_is_serializable():
    doc = _doc([_text("body")])
    result = HierarchicalChunker().chunk(doc)
    payload = result.as_dict()
    assert "parents" in payload and "children" in payload
    assert payload["parents"][0]["id"] == result.parents[0].id


def test_chunk_children_helper_returns_flat_list():
    doc = _doc([_text(_words(300))])
    chunker = HierarchicalChunker(HierarchicalConfig(parent_size=1000, child_size=80, child_overlap=0))
    children = chunker.chunk_children(doc)
    assert children
    assert all(chunk.metadata["level"] == "child" for chunk in children)


def test_iter_pairs():
    doc = _doc([_heading("A", 1), _text(_words(200, "alpha")), _heading("B", 1), _text(_words(200, "beta"))])
    chunker = HierarchicalChunker(HierarchicalConfig(parent_size=1000, child_size=80, child_overlap=0))
    pairs = list(chunker.iter_pairs(doc))
    assert len(pairs) == len(chunker.chunk(doc).parents)
    for parent, children in pairs:
        assert children
        assert all(child.metadata["parent_id"] == parent.id for child in children)


# ---------------------------------------------------------------------------
# Determinism, unicode, immutability
# ---------------------------------------------------------------------------
def test_deterministic_ids_across_runs():
    doc = _doc(
        [
            _heading("A", 1),
            _text(_words(200, "alpha")),
            _heading("B", 1),
            _text(_words(200, "beta")),
        ]
    )
    config = HierarchicalConfig(parent_size=600, child_size=80, child_overlap=0)
    first = HierarchicalChunker(config).chunk(doc)
    second = HierarchicalChunker(config).chunk(doc)
    assert [p.id for p in first.parents] == [p.id for p in second.parents]
    assert [c.id for c in first.children] == [c.id for c in second.children]
    assert [p.as_dict() for p in first.parents] == [p.as_dict() for p in second.parents]
    assert [c.as_dict() for c in first.children] == [c.as_dict() for c in second.children]


def test_ids_are_unique():
    doc = _doc([_text(_words(600))])
    result = HierarchicalChunker(HierarchicalConfig(parent_size=300, child_size=80, child_overlap=0)).chunk(doc)
    ids = [p.id for p in result.parents] + [c.id for c in result.children]
    assert len(ids) == len(set(ids))


def test_ids_depend_on_source_document():
    doc_a = _doc([_text("body")], source="a.txt")
    doc_b = _doc([_text("body")], source="b.txt")
    result_a = HierarchicalChunker().chunk(doc_a)
    result_b = HierarchicalChunker().chunk(doc_b)
    assert result_a.parents[0].id != result_b.parents[0].id


def test_count_and_index_consistency():
    doc = _doc([_text(_words(500))])
    result = HierarchicalChunker(HierarchicalConfig(parent_size=300, child_size=80, child_overlap=0)).chunk(doc)
    assert [p.index for p in result.parents] == list(range(len(result.parents)))
    assert all(p.count == len(result.parents) for p in result.parents)
    assert [c.index for c in result.children] == list(range(len(result.children)))
    assert all(c.count == len(result.children) for c in result.children)


def test_unicode_content_preserved():
    doc = _doc([_heading("Ünïcödé ✨", 1), _text("héllo wörld " * 60)])
    result = HierarchicalChunker(HierarchicalConfig(parent_size=200, child_size=60, child_overlap=0)).chunk(doc)
    assert coverage_stats(doc, result.children)["coverage"] == pytest.approx(1.0)
    assert "Ünïcödé ✨" in result.parents[0].text


def test_input_document_not_mutated():
    doc = _doc([_heading("A", 1), _text(_words(200))])
    before = doc.as_dict()
    HierarchicalChunker(HierarchicalConfig(parent_size=300, child_size=60, child_overlap=0)).chunk(doc)
    assert doc.as_dict() == before


def test_chunk_text_matches_metadata_range():
    doc = _doc([_heading("A", 1), _text(_words(300))])
    text = flattened_text(doc)
    result = HierarchicalChunker(HierarchicalConfig(parent_size=400, child_size=70, child_overlap=0)).chunk(doc)
    for chunk in result.parents + result.children:
        assert text[chunk.metadata["char_start"] : chunk.metadata["char_end"]] == chunk.text


# ---------------------------------------------------------------------------
# Oversized blocks
# ---------------------------------------------------------------------------
def test_oversized_source_block_is_split():
    doc = _doc([ParsedBlock(BlockType.TEXT, _words(900, "big"))])
    result = HierarchicalChunker(HierarchicalConfig(parent_size=300, child_size=100, child_overlap=0)).chunk(doc)
    assert len(result.parents) > 1
    assert len(result.children) > 1
    for parent in result.parents:
        assert parent.length <= 300
    for child in result.children:
        assert child.length <= 100
    assert coverage_stats(doc, result.children)["coverage"] == pytest.approx(1.0)


def test_oversized_table_block_is_split():
    table = "Item | Qty\n" + "\n".join(f"row{i} | {i}" for i in range(200))
    doc = _doc([ParsedBlock(BlockType.TABLE, table)])
    result = HierarchicalChunker(HierarchicalConfig(parent_size=400, child_size=150, child_overlap=0)).chunk(doc)
    assert len(result.children) > 1
    joined = "\n".join(child.text for child in result.children)
    for i in range(200):
        assert f"row{i} | {i}" in joined


def test_content_coverage_is_complete():
    blocks = []
    for i in range(4):
        blocks.append(_heading(f"Section {i}", 1))
        blocks.append(_text(_words(80, f"w{i}")))
    doc = _doc(blocks)
    result = HierarchicalChunker(HierarchicalConfig(parent_size=400, child_size=90, child_overlap=0)).chunk(doc)
    assert coverage_stats(doc, result.children)["coverage"] == pytest.approx(1.0)
    assert coverage_stats(doc, result.parents)["coverage"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------
def test_invalid_config_rejected():
    with pytest.raises(ChunkerConfigError):
        HierarchicalConfig(parent_size=0)
    with pytest.raises(ChunkerConfigError):
        HierarchicalConfig(child_size=0)
    with pytest.raises(ChunkerConfigError):
        HierarchicalConfig(child_overlap=-1)
    with pytest.raises(ChunkerConfigError):
        HierarchicalConfig(child_size=100, child_overlap=100)
    with pytest.raises(ChunkerConfigError):
        HierarchicalConfig(parent_size=100, child_size=200)
    with pytest.raises(ChunkerConfigError):
        HierarchicalConfig(tokenizer=123)  # type: ignore[arg-type]


def test_non_document_input_rejected():
    with pytest.raises(TypeError):
        HierarchicalChunker().chunk("not a document")  # type: ignore[arg-type]
