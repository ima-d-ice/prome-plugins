"""Integration tests: run the three chunkers on real parser fixtures."""

import pytest

from chunkers.fixed import FixedSizeChunker, FixedSizeConfig
from chunkers.recursive import RecursiveChunker, RecursiveConfig
from chunkers.structure_aware import StructureAwareChunker, StructureAwareConfig

from .helpers import assert_provenance, coverage_stats

CHUNKERS = [
    pytest.param(FixedSizeChunker, FixedSizeConfig(chunk_size=500, overlap=50), id="fixed"),
    pytest.param(RecursiveChunker, RecursiveConfig(chunk_size=500, overlap=50), id="recursive"),
    pytest.param(StructureAwareChunker, StructureAwareConfig(chunk_size=500), id="structure_aware"),
]


@pytest.mark.parametrize("chunker_cls, config", CHUNKERS)
def test_fixture_chunks_end_to_end(chunker_cls, config, parsed_fixture):
    doc = parsed_fixture
    chunks = chunker_cls(config).chunk(doc)
    if not doc.text.strip():
        assert chunks == []
        return
    assert chunks, f"expected chunks for {doc.source}"
    stats = coverage_stats(doc, chunks)
    assert stats["coverage"] == pytest.approx(1.0)
    for chunk in chunks:
        assert chunk.text.strip()
        assert chunk.length > 0
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.count == len(chunks) for c in chunks)
    assert_provenance(doc, chunks)


@pytest.mark.parametrize("chunker_cls, config", CHUNKERS)
def test_fixture_deterministic(chunker_cls, config, parsed_fixture):
    doc = parsed_fixture
    first = [c.as_dict() for c in chunker_cls(config).chunk(doc)]
    second = [c.as_dict() for c in chunker_cls(config).chunk(doc)]
    assert first == second


@pytest.mark.parametrize("chunker_cls, config", CHUNKERS)
def test_fixture_default_config_limits(chunker_cls, config, parsed_fixture):
    doc = parsed_fixture
    for chunk in chunker_cls(config).chunk(doc):
        assert chunk.length >= 1
        assert chunk.metadata["source"] == doc.source
        assert chunk.metadata["parser_name"] == doc.parser_name
        assert chunk.metadata["strategy"]


@pytest.mark.parametrize("chunker_cls, config", CHUNKERS)
def test_fixture_block_references_resolve(chunker_cls, config, parsed_fixture):
    doc = parsed_fixture
    chunks = chunker_cls(config).chunk(doc)
    if not doc.text.strip():
        pytest.skip("fixture carries no text content")
    referenced = {
        (ref.page_number, ref.block_index)
        for chunk in chunks
        for ref in chunk.block_refs
    }
    content_blocks = {
        (page.page_number, index)
        for page in doc.pages
        for index, block in enumerate(page.blocks)
        if block.content and block.content.strip()
    } | {
        (None, index)
        for index, block in enumerate(doc.blocks)
        if block.content and block.content.strip()
    }
    # Every referenced block must exist and carry content.
    assert referenced <= content_blocks
    # At least half of the content blocks should be referenced (tables may be
    # split across chunks but every content block should appear at least once).
    assert referenced == content_blocks
