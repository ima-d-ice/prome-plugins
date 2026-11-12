"""Contract tests: models, counters, config validation, non-mutation."""

import pytest

from chunkers.contracts import (
    ApproxTokenCounter,
    BlockReference,
    CallableCounter,
    CharacterCounter,
    Chunk,
    Chunker,
    ChunkerConfigError,
    ensure_counter,
    validate_size_overlap,
)
from chunkers.fixed import FixedSizeChunker, FixedSizeConfig
from chunkers.recursive import RecursiveChunker, RecursiveConfig
from chunkers.structure_aware import StructureAwareChunker, StructureAwareConfig
from parsers.contracts import BlockType, ParsedBlock, ParsedDocument, ParsedPage


def _sample_doc() -> ParsedDocument:
    return ParsedDocument(
        source="sample.html",
        parser_name="test",
        parser_version="1.0",
        metadata={"title": "Sample"},
        pages=[
            ParsedPage(
                page_number=1,
                blocks=[
                    ParsedBlock(BlockType.HEADING, "Intro", page_number=1, metadata={"heading_level": 1}),
                    ParsedBlock(BlockType.TEXT, "alpha beta gamma delta epsilon", page_number=1),
                ],
            )
        ],
    )


def test_chunk_defaults_and_as_dict():
    ref = BlockReference(block_index=0, page_number=1, block_type="text", char_start=0, char_end=5)
    chunk = Chunk(
        id="abc-fixed-0000",
        doc_id="doc",
        text="hello",
        index=0,
        count=1,
        length=5,
        length_unit="characters",
        block_refs=[ref],
        heading="Intro",
        metadata={"strategy": "fixed"},
    )
    data = chunk.as_dict()
    assert data["id"] == "abc-fixed-0000"
    assert data["count"] == 1
    assert data["block_refs"][0]["block_index"] == 0
    assert data["block_refs"][0]["char_end"] == 5
    assert data["heading"] == "Intro"
    assert data["length_unit"] == "characters"


def test_chunker_protocol_runtime_check():
    assert isinstance(FixedSizeChunker(), Chunker)
    assert isinstance(RecursiveChunker(), Chunker)
    assert isinstance(StructureAwareChunker(), Chunker)


def test_character_counter_is_honest():
    counter = CharacterCounter()
    assert counter.count("héllo") == 5
    assert counter.unit == "characters"
    assert counter.name == "character"


def test_approx_token_counter_is_labelled_approx():
    counter = ApproxTokenCounter()
    assert counter.count("") == 0
    assert counter.count("abcd") == 1
    assert counter.count("abcde") == 2
    assert counter.unit == "approx_tokens"
    assert "approx" in counter.name


def test_ensure_counter_none_defaults_to_characters():
    counter = ensure_counter(None)
    assert isinstance(counter, CharacterCounter)


def test_ensure_counter_wraps_plain_callable():
    counter = ensure_counter(lambda s: len(s.split()))
    assert counter.count("a b c") == 3
    assert counter.unit == "tokens"


def test_ensure_counter_passthrough_and_bad_type():
    counter = ensure_counter(ApproxTokenCounter())
    assert isinstance(counter, ApproxTokenCounter)
    with pytest.raises(TypeError):
        ensure_counter(123)  # type: ignore[arg-type]


def test_callable_counter_rejects_negative():
    counter = CallableCounter(lambda s: -1)
    with pytest.raises(ValueError):
        counter.count("text")


def test_validate_size_overlap_accepts_valid():
    validate_size_overlap(100, 0)
    validate_size_overlap(100, 99)


@pytest.mark.parametrize(
    "chunk_size, overlap",
    [(0, 0), (-1, 0), (100, 100), (100, 200), (100, -1), (True, 0), (100, False)],
)
def test_validate_size_overlap_rejects_invalid(chunk_size, overlap):
    with pytest.raises(ChunkerConfigError):
        validate_size_overlap(chunk_size, overlap)


@pytest.mark.parametrize("config_cls", [FixedSizeConfig, RecursiveConfig])
def test_config_rejects_bad_size_and_overlap(config_cls):
    with pytest.raises(ChunkerConfigError):
        config_cls(chunk_size=0)
    with pytest.raises(ChunkerConfigError):
        config_cls(chunk_size=100, overlap=100)
    with pytest.raises(ChunkerConfigError):
        config_cls(chunk_size=100, overlap=-1)


def test_structure_aware_config_validation():
    with pytest.raises(ChunkerConfigError):
        StructureAwareConfig(chunk_size=0)
    with pytest.raises(ChunkerConfigError):
        StructureAwareConfig(close_on_heading="yes")  # type: ignore[arg-type]


def test_recursive_config_separator_validation():
    with pytest.raises(ChunkerConfigError):
        RecursiveConfig(separators=())
    with pytest.raises(ChunkerConfigError):
        RecursiveConfig(separators=("a", 1))  # type: ignore[arg-type]


def test_fixed_config_word_boundary_flag_validation():
    with pytest.raises(ChunkerConfigError):
        FixedSizeConfig(respect_word_boundaries="yes")  # type: ignore[arg-type]


@pytest.mark.parametrize("chunker_cls", [FixedSizeChunker, RecursiveChunker, StructureAwareChunker])
def test_chunkers_do_not_mutate_document(chunker_cls):
    doc = _sample_doc()
    before = doc.as_dict()
    chunker_cls().chunk(doc)
    assert doc.as_dict() == before


@pytest.mark.parametrize("chunker_cls", [FixedSizeChunker, RecursiveChunker, StructureAwareChunker])
def test_chunkers_reject_non_document(chunker_cls):
    with pytest.raises(TypeError):
        chunker_cls().chunk("not a document")  # type: ignore[arg-type]


def test_empty_document_returns_no_chunks():
    doc = ParsedDocument(source="empty.html", parser_name="test", parser_version="1")
    for chunker_cls in (FixedSizeChunker, RecursiveChunker, StructureAwareChunker):
        assert chunker_cls().chunk(doc) == []


def test_whitespace_only_document_returns_no_chunks():
    doc = ParsedDocument(
        source="ws.html",
        parser_name="test",
        parser_version="1",
        blocks=[ParsedBlock(BlockType.TEXT, "   \n\n   ")],
    )
    for chunker_cls in (FixedSizeChunker, RecursiveChunker, StructureAwareChunker):
        assert chunker_cls().chunk(doc) == []


@pytest.mark.parametrize("chunker_cls", [FixedSizeChunker, RecursiveChunker, StructureAwareChunker])
def test_chunk_ids_unique_and_counts_consistent(chunker_cls):
    doc = ParsedDocument(
        source="ids.html",
        parser_name="test",
        parser_version="1",
        blocks=[ParsedBlock(BlockType.TEXT, " ".join(["word"] * 500))],
    )
    chunks = chunker_cls().chunk(doc)
    ids = [c.id for c in chunks]
    assert len(set(ids)) == len(ids)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.count == len(chunks) for c in chunks)


@pytest.mark.parametrize("chunker_cls", [FixedSizeChunker, RecursiveChunker, StructureAwareChunker])
def test_deterministic_output(chunker_cls):
    doc = _sample_doc()
    first = [c.as_dict() for c in chunker_cls().chunk(doc)]
    second = [c.as_dict() for c in chunker_cls().chunk(doc)]
    assert first == second


def test_flatten_order_pages_then_top_level_blocks():
    from chunkers.contracts.flatten import flatten_document

    doc = ParsedDocument(
        source="both.html",
        parser_name="test",
        parser_version="1",
        pages=[ParsedPage(page_number=1, blocks=[ParsedBlock(BlockType.TEXT, "paged content", page_number=1)])],
        blocks=[ParsedBlock(BlockType.TEXT, "top-level content")],
    )
    text, flats = flatten_document(doc)
    assert text == "paged content\n\ntop-level content"
    assert flats[0].page_number == 1
    assert flats[1].page_number is None
    assert flats[1].block_index == 0
