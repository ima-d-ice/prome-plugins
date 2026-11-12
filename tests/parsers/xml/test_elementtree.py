from pathlib import Path
import pytest
from parsers.contracts import (
    BlockType,
    CorruptDocumentError,
    EmptyDocumentError,
    ParsedDocument,
)
from parsers.xml.elementtree import ElementTreeXmlParser

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "parsers" / "xml" / "fixtures"


def test_elementtree_catalog():
    parser = ElementTreeXmlParser()
    doc = parser.parse(FIXTURES_DIR / "catalog.xml")
    assert isinstance(doc, ParsedDocument)
    assert doc.parser_name == "elementtree_xml"
    assert doc.metadata["root_tag"] == "catalog"
    assert "XML Developer's Guide" in doc.text
    assert "Midnight Rain" in doc.text

    xpaths = {b.metadata.get("xpath") for b in doc.blocks}
    assert any("/catalog/book" in xp for xp in xpaths)


def test_elementtree_namespaces():
    parser = ElementTreeXmlParser()
    doc = parser.parse(FIXTURES_DIR / "namespaces.xml")
    assert "Document With Namespaces" in doc.text
    assert any(b.metadata.get("namespace") for b in doc.blocks)


def test_elementtree_cdata():
    parser = ElementTreeXmlParser()
    doc = parser.parse(FIXTURES_DIR / "cdata.xml")
    assert "function test()" in doc.text
    assert "x < y && a > b" in doc.text


def test_elementtree_unicode():
    parser = ElementTreeXmlParser()
    doc = parser.parse(FIXTURES_DIR / "unicode.xml")
    assert "café" in doc.text.lower()
    assert "こんにちは世界" in doc.text
    assert "привет" in doc.text.lower()


def test_elementtree_empty():
    parser = ElementTreeXmlParser()
    with pytest.raises(EmptyDocumentError):
        parser.parse(FIXTURES_DIR / "empty.xml")


def test_elementtree_malformed():
    parser = ElementTreeXmlParser()
    with pytest.raises(CorruptDocumentError):
        parser.parse(FIXTURES_DIR / "malformed.xml")
