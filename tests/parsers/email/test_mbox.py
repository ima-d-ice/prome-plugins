from pathlib import Path
import pytest
from parsers.contracts import (
    EmptyDocumentError,
    ParsedDocument,
)
from parsers.email.mbox import MboxParser

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "parsers" / "email" / "fixtures"


def test_mbox_archive():
    parser = MboxParser()
    doc = parser.parse(FIXTURES_DIR / "archive.mbox")
    assert isinstance(doc, ParsedDocument)
    assert doc.parser_name == "mbox_stdlib"
    assert doc.page_count == 3
    assert doc.metadata["message_count"] == 3
    p1 = doc.pages[0]
    assert p1.page_number == 1
    assert "Newsletter volume 1" in p1.text


def test_mbox_empty():
    empty_mbox = FIXTURES_DIR / "empty.eml"
    parser = MboxParser()
    with pytest.raises(EmptyDocumentError):
        parser.parse(empty_mbox)
