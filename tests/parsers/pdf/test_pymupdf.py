import pytest

pytest.importorskip("pymupdf", reason="pymupdf not installed")

from parsers.contracts import BlockType
from parsers.pdf.pymupdf import PyMuPDFParser


@pytest.fixture()
def parser():
    return PyMuPDFParser()


def test_page_dimensions_and_bbox(parser, fixtures_dir):
    doc = parser.parse(fixtures_dir / "simple_digital.pdf")
    page = doc.pages[0]
    assert page.width == pytest.approx(612, abs=1)
    assert page.height == pytest.approx(792, abs=1)
    text_blocks = [b for b in page.blocks if b.block_type == BlockType.TEXT]
    assert text_blocks
    for block in text_blocks:
        assert block.bbox is not None
        assert block.bbox.x1 >= block.bbox.x0
        assert block.bbox.y1 >= block.bbox.y0


def test_heading_detected(parser, fixtures_dir):
    doc = parser.parse(fixtures_dir / "multipage_structured.pdf")
    headings = [
        b.content for page in doc.pages for b in page.blocks if b.block_type == BlockType.HEADING
    ]
    assert "Introduction" in headings
    assert "Methods" in headings
    assert "Results" in headings


def test_table_detected(parser, fixtures_dir):
    doc = parser.parse(fixtures_dir / "tables.pdf")
    tables = [
        b for page in doc.pages for b in page.blocks if b.block_type == BlockType.TABLE
    ]
    assert tables, "expected a table block"
    content = tables[0].content
    assert "Widget" in content
    assert "Gizmo" in content


def test_metadata_title(parser, fixtures_dir):
    doc = parser.parse(fixtures_dir / "simple_digital.pdf")
    assert doc.metadata.get("title") == "Quarterly Report Overview"
    assert doc.metadata.get("author") == "Prome Test Suite"
