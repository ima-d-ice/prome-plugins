import pytest
from parsers.contracts import (
    BlockType,
    CorruptDocumentError,
    EmptyDocumentError,
    ParsedDocument,
)
from parsers.office.pptx.docling import DoclingPptxParser


def test_docling_pptx_multi_slide(office_fixtures_dir):
    parser = DoclingPptxParser()
    doc = parser.parse(office_fixtures_dir / "multi_slide.pptx")
    assert isinstance(doc, ParsedDocument)
    assert doc.parser_name == "docling_pptx"
    assert doc.page_count == 3
    assert "Launch Overview" in doc.pages[0].text
    assert "Next Steps" in doc.pages[2].text


def test_docling_pptx_tables(office_fixtures_dir):
    parser = DoclingPptxParser()
    doc = parser.parse(office_fixtures_dir / "table.pptx")
    assert doc.page_count == 1
    tables = [b for b in doc.pages[0].blocks if b.block_type == BlockType.TABLE]
    assert len(tables) == 1
    assert "Diaz" in tables[0].content


def test_docling_pptx_empty(office_fixtures_dir):
    parser = DoclingPptxParser()
    with pytest.raises(EmptyDocumentError):
        parser.parse(office_fixtures_dir / "empty.pptx")


def test_docling_pptx_malformed(office_fixtures_dir):
    parser = DoclingPptxParser()
    with pytest.raises(CorruptDocumentError):
        parser.parse(office_fixtures_dir / "malformed.pptx")
