import pytest
from parsers.contracts import (
    BlockType,
    CorruptDocumentError,
    EmptyDocumentError,
    ParsedDocument,
)
from parsers.office.docx.docling import DoclingDocxParser


def test_docling_docx_simple_text(office_fixtures_dir):
    parser = DoclingDocxParser()
    doc = parser.parse(office_fixtures_dir / "simple_text.docx")
    assert isinstance(doc, ParsedDocument)
    assert doc.parser_name == "docling_docx"
    assert len(doc.blocks) >= 3
    assert doc.page_count == 0
    assert "quarter closed on a steady note" in doc.text.lower()


def test_docling_docx_headings(office_fixtures_dir):
    parser = DoclingDocxParser()
    doc = parser.parse(office_fixtures_dir / "headings.docx")
    headings = [b for b in doc.blocks if b.block_type == BlockType.HEADING]
    assert len(headings) >= 4
    assert any("prome platform overview" in h.content.lower() for h in headings)


def test_docling_docx_tables(office_fixtures_dir):
    parser = DoclingDocxParser()
    doc = parser.parse(office_fixtures_dir / "tables.docx")
    tables = [b for b in doc.blocks if b.block_type == BlockType.TABLE]
    assert len(tables) == 2
    t1 = tables[0]
    assert "Widget Pro" in t1.content
    assert t1.metadata.get("rows") == 3


def test_docling_docx_empty(office_fixtures_dir):
    parser = DoclingDocxParser()
    with pytest.raises(EmptyDocumentError):
        parser.parse(office_fixtures_dir / "empty.docx")


def test_docling_docx_malformed(office_fixtures_dir):
    parser = DoclingDocxParser()
    with pytest.raises(CorruptDocumentError):
        parser.parse(office_fixtures_dir / "malformed.docx")
