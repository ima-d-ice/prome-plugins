import pytest
from parsers.contracts import (
    BlockType,
    CorruptDocumentError,
    EmptyDocumentError,
    ParsedDocument,
)
from parsers.office.docx.python_docx import PythonDocxParser


def test_python_docx_simple_text(office_fixtures_dir):
    parser = PythonDocxParser()
    doc = parser.parse(office_fixtures_dir / "simple_text.docx")
    assert isinstance(doc, ParsedDocument)
    assert doc.parser_name == "python_docx"
    assert doc.metadata.get("title") == "Quarterly Operations Brief"
    assert doc.metadata.get("author") == "A. Example"
    assert len(doc.blocks) == 3
    assert doc.page_count == 0
    assert "quarter closed on a steady note" in doc.text.lower()


def test_python_docx_headings(office_fixtures_dir):
    parser = PythonDocxParser()
    doc = parser.parse(office_fixtures_dir / "headings.docx")
    headings = [b for b in doc.blocks if b.block_type == BlockType.HEADING]
    assert len(headings) >= 5
    levels = [h.metadata.get("heading_level") for h in headings]
    assert 1 in levels
    assert 2 in levels
    assert 3 in levels


def test_python_docx_tables(office_fixtures_dir):
    parser = PythonDocxParser()
    doc = parser.parse(office_fixtures_dir / "tables.docx")
    tables = [b for b in doc.blocks if b.block_type == BlockType.TABLE]
    assert len(tables) == 2
    t1 = tables[0]
    assert t1.metadata["rows"] == 3
    assert t1.metadata["cols"] == 3
    assert "Widget Pro" in t1.content
    assert t1.metadata["headers"] == ["Item", "Quantity", "Total"]


def test_python_docx_nested_lists(office_fixtures_dir):
    parser = PythonDocxParser()
    doc = parser.parse(office_fixtures_dir / "nested_lists.docx")
    lists = [b for b in doc.blocks if b.block_type == BlockType.LIST]
    assert len(lists) > 0
    types = {b.metadata.get("list_type") for b in lists}
    assert "ul" in types or "ol" in types


def test_python_docx_sections_and_headers_footers(office_fixtures_dir):
    parser = PythonDocxParser()
    doc = parser.parse(office_fixtures_dir / "headers_footers.docx")
    regions = [b.metadata.get("region") for b in doc.blocks if b.metadata.get("region")]
    assert any("header" in str(r) for r in regions)
    assert any("footer" in str(r) for r in regions)


def test_python_docx_hyperlinks(office_fixtures_dir):
    parser = PythonDocxParser()
    doc = parser.parse(office_fixtures_dir / "hyperlinks.docx")
    links = [b for b in doc.blocks if b.block_type == BlockType.LINK]
    assert len(links) >= 1
    assert any("example.com" in b.metadata.get("href", "") for b in links)


def test_python_docx_unicode(office_fixtures_dir):
    parser = PythonDocxParser()
    doc = parser.parse(office_fixtures_dir / "unicode.docx")
    assert "café" in doc.text.lower()
    assert "日本語" in doc.text
    assert "русский" in doc.text.lower()


def test_python_docx_empty(office_fixtures_dir):
    parser = PythonDocxParser()
    with pytest.raises(EmptyDocumentError):
        parser.parse(office_fixtures_dir / "empty.docx")


def test_python_docx_malformed(office_fixtures_dir):
    parser = PythonDocxParser()
    with pytest.raises(CorruptDocumentError):
        parser.parse(office_fixtures_dir / "malformed.docx")
