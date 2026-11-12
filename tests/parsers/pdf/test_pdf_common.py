import pytest

from parsers.contracts import (
    BlockType,
    BoundingBox,
    CorruptDocumentError,
    EmptyDocumentError,
    EncryptedDocumentError,
    ParsedBlock,
    ParsedDocument,
    ParsedPage,
    ParserError,
)

from tests.parsers.pdf.conftest import make_parser, parser_importable


def test_parse_simple_digital(parser, fixtures_dir):
    doc = parser.parse(fixtures_dir / "simple_digital.pdf")
    assert isinstance(doc, ParsedDocument)
    assert doc.parser_name in {"pymupdf", "pypdf", "docling"}
    assert doc.parser_version
    assert doc.page_count == 1
    assert "Quarterly Report Overview" in doc.text


def test_page_numbers_are_sequential_and_one_based(parser, fixtures_dir):
    doc = parser.parse(fixtures_dir / "multipage_structured.pdf")
    assert [p.page_number for p in doc.pages] == [1, 2, 3]
    for page in doc.pages:
        for block in page.blocks:
            assert block.page_number == page.page_number


def test_multipage_headings_extractable(parser, fixtures_dir):
    doc = parser.parse(fixtures_dir / "multipage_structured.pdf")
    assert doc.page_count == 3
    text = doc.text
    assert "Introduction" in text
    assert "Methods" in text
    assert "Results" in text


def test_blank_page_preserved(parser, fixtures_dir):
    doc = parser.parse(fixtures_dir / "blank_page.pdf")
    assert doc.page_count == 2
    blank = doc.pages[1]
    assert blank.page_number == 2
    assert blank.blocks == []
    assert any("2" in warning for warning in doc.warnings)


def test_corrupt_file_raises(parser, tmp_path):
    bad = tmp_path / "corrupt.pdf"
    bad.write_bytes(b"this is definitely not a pdf file %PDF-1.4 truncated garbage \x89\x90" * 4)
    with pytest.raises(ParserError):
        parser.parse(bad)


def test_zero_byte_file_raises(parser, tmp_path):
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    with pytest.raises(EmptyDocumentError):
        parser.parse(empty)


def test_missing_file_raises(parser, tmp_path):
    with pytest.raises(FileNotFoundError):
        parser.parse(tmp_path / "does_not_exist.pdf")


def test_encrypted_pdf_raises(parser):
    # All three parsers must reject the encrypted fixture with a meaningful
    # parser-level error; exact subclass may vary by backend.
    import pathlib

    encrypted = (
        pathlib.Path(__file__).resolve().parents[3]
        / "benchmarks"
        / "parsers"
        / "pdf"
        / "fixtures"
        / "encrypted.pdf"
    )
    if not encrypted.exists():
        pytest.skip("encrypted fixture missing")
    with pytest.raises(ParserError):
        parser.parse(encrypted)


def test_metadata_is_dict(parser, fixtures_dir):
    doc = parser.parse(fixtures_dir / "simple_digital.pdf")
    assert isinstance(doc.metadata, dict)


def test_warnings_is_list_of_str(parser, fixtures_dir):
    doc = parser.parse(fixtures_dir / "blank_page.pdf")
    assert isinstance(doc.warnings, list)
    assert all(isinstance(w, str) for w in doc.warnings)


def test_schema_conformance(parser, fixtures_dir, expectations):
    for filename, expected in expectations.items():
        doc = parser.parse(fixtures_dir / filename)
        assert isinstance(doc, ParsedDocument)
        assert doc.pages, filename
        for page in doc.pages:
            assert isinstance(page, ParsedPage)
            assert page.page_number >= 1
            for block in page.blocks:
                assert isinstance(block, ParsedBlock)
                assert isinstance(block.block_type, BlockType)
                assert isinstance(block.content, str)
                assert block.page_number == page.page_number
                assert isinstance(block.metadata, dict)
                if block.bbox is not None:
                    assert isinstance(block.bbox, BoundingBox)
                    assert block.bbox.x1 >= block.bbox.x0
                    assert block.bbox.y1 >= block.bbox.y0


def _known_gaps(expected: dict, parser_name: str) -> set[str]:
    # Documented, measured extraction gaps (see expectations.json and the
    # benchmark report). The benchmark still counts these as missed; they are
    # excluded here so the suite asserts "everything else must match".
    return set(expected.get("known_extraction_gaps", {}).get(parser_name, []))


def test_text_coverage(parser, fixtures_dir, expectations, parser_name):
    for filename, expected in expectations.items():
        parser_gaps = _known_gaps(expected, parser_name)
        doc = parser.parse(fixtures_dir / filename)
        text = doc.text
        for snippet in expected["expected_text"]:
            if snippet in parser_gaps:
                continue
            assert snippet.lower() in text.lower(), f"{filename}: missing snippet {snippet!r}"


def test_reading_order(parser, fixtures_dir, expectations, parser_name):
    for filename, expected in expectations.items():
        parser_gaps = _known_gaps(expected, parser_name)
        markers = [m for m in expected["reading_order"] if m not in parser_gaps]
        if len(markers) < 2:
            continue
        doc = parser.parse(fixtures_dir / filename)
        text = doc.text.lower()
        positions = [text.find(marker.lower()) for marker in markers]
        assert all(pos >= 0 for pos in positions), f"{filename}: missing marker in {markers}"
        assert positions == sorted(positions), f"{filename}: markers out of order"


@pytest.mark.parametrize("name", ["pymupdf", "pypdf", "docling"])
def test_encrypted_pdf_specific_error(name, fixtures_dir):
    if not parser_importable(name):
        pytest.skip(f"{name} not installed")
    parser = make_parser(name)
    encrypted = fixtures_dir / "encrypted.pdf"
    if name in {"pymupdf", "pypdf"}:
        with pytest.raises(EncryptedDocumentError):
            parser.parse(encrypted)
    else:
        with pytest.raises(ParserError):
            parser.parse(encrypted)


def test_corrupt_raises_corrupt_document_error(parser, tmp_path):
    bad = tmp_path / "corrupt.pdf"
    bad.write_bytes(b"\x00\x01\x02\x03 not a pdf" * 32)
    with pytest.raises((CorruptDocumentError, EmptyDocumentError)):
        # A zero-ish garbage file may surface as either; never as success.
        parser.parse(bad)
