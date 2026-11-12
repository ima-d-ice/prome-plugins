from pathlib import Path
import pytest
from parsers.contracts import (
    BlockType,
    CorruptDocumentError,
    EmptyDocumentError,
    ParsedDocument,
)
from parsers.images.tesseract import TesseractParser

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "parsers" / "images" / "fixtures"


def test_tesseract_clean_text():
    parser = TesseractParser()
    doc = parser.parse(FIXTURES_DIR / "clean_text.png")
    assert isinstance(doc, ParsedDocument)
    assert doc.parser_name == "tesseract_ocr"
    assert doc.page_count == 1
    page = doc.pages[0]
    assert page.width == 600.0
    assert page.height == 200.0
    assert len(page.blocks) >= 2
    assert "PROME" in doc.text
    for b in page.blocks:
        assert b.bbox is not None
        assert b.metadata.get("confidence", 0) > 0


def test_tesseract_receipt():
    parser = TesseractParser()
    doc = parser.parse(FIXTURES_DIR / "receipt.png")
    assert "RECEIPT" in doc.text
    assert "Total" in doc.text


def test_tesseract_blank_image():
    parser = TesseractParser()
    with pytest.raises(EmptyDocumentError):
        parser.parse(FIXTURES_DIR / "blank.png")


def test_tesseract_corrupt_image():
    parser = TesseractParser()
    with pytest.raises(CorruptDocumentError):
        parser.parse(FIXTURES_DIR / "corrupt.png")
