import pytest

pytest.importorskip("pypdf", reason="pypdf not installed")

from parsers.contracts import BlockType
from parsers.pdf.pypdf import PyPDFParser


@pytest.fixture()
def parser():
    return PyPDFParser()


def test_blocks_are_plain_text(parser, fixtures_dir):
    doc = parser.parse(fixtures_dir / "multipage_structured.pdf")
    for page in doc.pages:
        for block in page.blocks:
            assert block.block_type in {BlockType.TEXT, BlockType.OTHER}
            assert block.bbox is None


def test_page_dimensions(parser, fixtures_dir):
    doc = parser.parse(fixtures_dir / "simple_digital.pdf")
    assert doc.pages[0].width == pytest.approx(612, abs=1)
    assert doc.pages[0].height == pytest.approx(792, abs=1)


def test_reading_order_preserved(parser, fixtures_dir):
    doc = parser.parse(fixtures_dir / "multipage_structured.pdf")
    text = doc.text
    assert text.index("Introduction") < text.index("Methods") < text.index("Results")


def test_metadata_keys_normalized(parser, fixtures_dir):
    doc = parser.parse(fixtures_dir / "simple_digital.pdf")
    assert doc.metadata.get("Title") == "Quarterly Report Overview"
    assert doc.metadata.get("Author") == "Prome Test Suite"
