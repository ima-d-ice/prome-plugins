from pathlib import Path
import pytest
from parsers.contracts import (
    BlockType,
    CorruptDocumentError,
    EmptyDocumentError,
    ParsedDocument,
)
from parsers.xml.lxml import LxmlXmlParser

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "parsers" / "xml" / "fixtures"


def test_lxml_catalog():
    parser = LxmlXmlParser()
    doc = parser.parse(FIXTURES_DIR / "catalog.xml")
    assert isinstance(doc, ParsedDocument)
    assert doc.parser_name == "lxml_xml"
    assert doc.metadata["root_tag"] == "catalog"
    assert "XML Developer's Guide" in doc.text

    xpaths = {b.metadata.get("xpath") for b in doc.blocks}
    assert "/catalog/book[1]/title" in xpaths or any("/catalog/book" in p for p in xpaths)


def test_lxml_namespaces():
    parser = LxmlXmlParser()
    doc = parser.parse(FIXTURES_DIR / "namespaces.xml")
    assert "Document With Namespaces" in doc.text


def test_lxml_cdata():
    parser = LxmlXmlParser()
    doc = parser.parse(FIXTURES_DIR / "cdata.xml")
    assert "function test()" in doc.text


def test_lxml_empty():
    parser = LxmlXmlParser()
    with pytest.raises(EmptyDocumentError):
        parser.parse(FIXTURES_DIR / "empty.xml")


def test_lxml_malformed():
    parser = LxmlXmlParser()
    with pytest.raises(CorruptDocumentError):
        parser.parse(FIXTURES_DIR / "malformed.xml")
