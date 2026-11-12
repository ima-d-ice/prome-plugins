import pytest
from parsers.contracts import (
    BlockType,
    CorruptDocumentError,
    EmptyDocumentError,
    ParsedDocument,
)
from parsers.office.pptx.python_pptx import PythonPptxParser


def test_python_pptx_multi_slide(office_fixtures_dir):
    parser = PythonPptxParser()
    doc = parser.parse(office_fixtures_dir / "multi_slide.pptx")
    assert isinstance(doc, ParsedDocument)
    assert doc.parser_name == "python_pptx"
    assert doc.page_count == 3
    assert doc.metadata.get("title") == "Prome Launch Review"
    p1 = doc.pages[0]
    assert p1.page_number == 1
    titles = [b.content for b in p1.blocks if b.block_type == BlockType.HEADING]
    assert "Launch Overview" in titles
    assert "We shipped the platform." in p1.text


def test_python_pptx_tables(office_fixtures_dir):
    parser = PythonPptxParser()
    doc = parser.parse(office_fixtures_dir / "table.pptx")
    p1 = doc.pages[0]
    tables = [b for b in p1.blocks if b.block_type == BlockType.TABLE]
    assert len(tables) == 1
    t = tables[0]
    assert t.metadata["rows"] == 3
    assert t.metadata["cols"] == 3
    assert "Diaz" in t.content


def test_python_pptx_text_boxes_and_bullets(office_fixtures_dir):
    parser = PythonPptxParser()
    doc = parser.parse(office_fixtures_dir / "text_boxes.pptx")
    p1 = doc.pages[0]
    lists = [b for b in p1.blocks if b.block_type == BlockType.LIST]
    assert len(lists) >= 1
    assert any(b.metadata.get("level") == 1 for b in lists)


def test_python_pptx_images_and_notes(office_fixtures_dir):
    parser = PythonPptxParser()
    doc = parser.parse(office_fixtures_dir / "images_notes.pptx")
    p1 = doc.pages[0]
    images = [b for b in p1.blocks if b.block_type == BlockType.IMAGE]
    assert len(images) == 1
    notes = [b for b in p1.blocks if b.metadata.get("is_speaker_note")]
    assert len(notes) == 1
    assert "Remind the audience" in notes[0].content


def test_python_pptx_unicode(office_fixtures_dir):
    parser = PythonPptxParser()
    doc = parser.parse(office_fixtures_dir / "unicode.pptx")
    assert "café" in doc.text.lower()
    assert "日本語" in doc.text


def test_python_pptx_empty(office_fixtures_dir):
    parser = PythonPptxParser()
    with pytest.raises(EmptyDocumentError):
        parser.parse(office_fixtures_dir / "empty.pptx")


def test_python_pptx_malformed(office_fixtures_dir):
    parser = PythonPptxParser()
    with pytest.raises(CorruptDocumentError):
        parser.parse(office_fixtures_dir / "malformed.pptx")
