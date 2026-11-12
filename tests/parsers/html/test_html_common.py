import pytest

from parsers.contracts import (
    BlockType,
    EmptyDocumentError,
    ParsedBlock,
    ParsedDocument,
    ParserError,
)

from tests.parsers.html.conftest import make_parser, parser_importable


def test_parse_simple_article(parser, fixtures_dir):
    doc = parser.parse(fixtures_dir / "simple_article.html")
    assert isinstance(doc, ParsedDocument)
    assert doc.parser_name in {"beautifulsoup", "trafilatura"}
    assert doc.parser_version
    assert "Getting Started with Widgets" in doc.text
    assert doc.pages == []          # HTML has no pages
    assert doc.page_count == 0
    assert doc.blocks               # content lives in top-level blocks
    assert all(block.page_number is None for block in doc.blocks)


def test_document_text_spans_all_blocks(parser, fixtures_dir):
    doc = parser.parse(fixtures_dir / "simple_article.html")
    assert "basics" not in doc.text  # not in this fixture
    assert "fundamental building blocks" in doc.text


def test_block_types_and_metadata_types(parser, fixtures_dir, expectations):
    for filename in expectations:
        doc = parser.parse(fixtures_dir / filename)
        for block in doc.blocks:
            assert isinstance(block, ParsedBlock)
            assert isinstance(block.block_type, BlockType)
            assert isinstance(block.content, str)
            assert isinstance(block.metadata, dict)
            assert block.page_number is None


def test_dom_order_metadata_is_strictly_increasing(parser, fixtures_dir):
    for filename in ("simple_article.html", "documentation_headings.html", "nested_lists.html", "long_document.html"):
        doc = parser.parse(fixtures_dir / filename)
        orders = [b.metadata.get("dom_order") for b in doc.blocks]
        assert orders == sorted(orders), filename
        assert len(set(orders)) == len(orders), filename


def test_missing_file_raises(parser, tmp_path):
    with pytest.raises(FileNotFoundError):
        parser.parse(tmp_path / "does_not_exist.html")


def test_zero_byte_file_raises(parser, tmp_path):
    empty = tmp_path / "empty.html"
    empty.write_bytes(b"")
    with pytest.raises(EmptyDocumentError):
        parser.parse(empty)


def test_empty_string_raises(parser):
    with pytest.raises(EmptyDocumentError):
        parser.parse("")


def test_raw_html_string_input(parser):
    html = "<html><head><title>Raw</title></head><body><h1>Hello Raw String</h1><p>Some raw html string content that is long enough to extract.</p></body></html>"
    doc = parser.parse(html)
    assert doc.source == "<string>"
    assert "Hello Raw String" in doc.text


def test_bytes_input(parser):
    html = b"<html><head><title>B</title></head><body><h1>Bytes Input Heading</h1><p>Some bytes input content that is long enough to extract.</p></body></html>"
    doc = parser.parse(html)
    assert "Bytes Input Heading" in doc.text


def test_malformed_html_does_not_raise(parser, fixtures_dir):
    doc = parser.parse(fixtures_dir / "malformed.html")
    assert "This paragraph is never closed" in doc.text
    assert "Second paragraph" in doc.text


def test_minimal_no_content_warns(parser, fixtures_dir):
    doc = parser.parse(fixtures_dir / "minimal_no_content.html")
    assert doc.blocks == []
    assert any("no main content" in w.lower() or "no extractable" in w.lower() for w in doc.warnings)
    # Title metadata is still available even with no body content
    assert doc.metadata.get("title") == "Under Construction"


def test_unicode_text_preserved(parser, fixtures_dir):
    doc = parser.parse(fixtures_dir / "unicode_text.html")
    for snippet in ("crème brûlée", "straße", "日本語のテキスト", "العربية", "Café—naïve"):
        assert snippet in doc.text


def test_schema_conformance(parser, fixtures_dir, expectations):
    for filename in expectations:
        doc = parser.parse(fixtures_dir / filename)
        assert isinstance(doc, ParsedDocument)
        assert isinstance(doc.metadata, dict)
        assert isinstance(doc.warnings, list)
        assert isinstance(doc.blocks, list)
        assert doc.as_dict()["blocks"] == [b.as_dict() for b in doc.blocks]
        for block in doc.blocks:
            assert block.content == block.content.strip()


def test_text_coverage(parser, fixtures_dir, expectations, parser_name):
    for filename, expected in expectations.items():
        snippets = expected["expected_text"]
        if not snippets:
            continue
        doc = parser.parse(fixtures_dir / filename)
        text = doc.text.lower()
        for snippet in snippets:
            assert snippet.lower() in text, f"{parser_name} {filename}: missing snippet {snippet!r}"


def test_reading_order(parser, fixtures_dir, expectations, parser_name):
    for filename, expected in expectations.items():
        markers = expected["reading_order"]
        if len(markers) < 2:
            continue
        doc = parser.parse(fixtures_dir / filename)
        text = doc.text.lower()
        positions = [text.find(marker.lower()) for marker in markers]
        assert all(pos >= 0 for pos in positions), f"{parser_name} {filename}: missing marker in {markers}"
        assert positions == sorted(positions), f"{parser_name} {filename}: markers out of order"


def test_metadata_is_dict_and_json_safe(parser, fixtures_dir):
    import json as _json

    doc = parser.parse(fixtures_dir / "metadata_og_jsonld.html")
    assert isinstance(doc.metadata, dict)
    _json.dumps(doc.metadata, default=str)  # must not raise


def test_source_url_provenance(parser, fixtures_dir):
    doc = parser.parse(fixtures_dir / "simple_article.html", source_url="https://example.com/provenance")
    assert doc.metadata.get("source_url") == "https://example.com/provenance"


def test_warnings_is_list_of_str(parser, fixtures_dir):
    doc = parser.parse(fixtures_dir / "minimal_no_content.html")
    assert isinstance(doc.warnings, list)
    assert all(isinstance(w, str) for w in doc.warnings)


def test_as_dict_round_trip(parser, fixtures_dir):
    doc = parser.parse(fixtures_dir / "simple_article.html")
    as_dict = doc.as_dict()
    assert set(as_dict) == {"source", "parser_name", "parser_version", "metadata", "pages", "blocks", "warnings"}
    assert as_dict["pages"] == []
    assert as_dict["blocks"]


@pytest.mark.parametrize("name", ["beautifulsoup", "trafilatura"])
def test_missing_dependency_raises_helpful_error(monkeypatch, name, fixtures_dir):
    module = "bs4" if name == "beautifulsoup" else "trafilatura"
    monkeypatch.setitem(__import__("sys").modules, module, None)
    parser = make_parser(name)
    with pytest.raises(ParserError) as excinfo:
        parser.parse(fixtures_dir / "simple_article.html")
    assert "pip install" in str(excinfo.value)
