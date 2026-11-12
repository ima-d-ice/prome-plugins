import pytest

from parsers.contracts import (
    BlockType,
    BoundingBox,
    CorruptDocumentError,
    DocumentParseError,
    EmptyDocumentError,
    EncryptedDocumentError,
    ParsedBlock,
    ParsedDocument,
    ParsedPage,
    Parser,
    ParserDependencyError,
    ParserError,
)


def test_block_type_values():
    assert {t.value for t in BlockType} == {
        "text", "heading", "table", "image", "list", "code", "link",
        "record", "key_value", "attachment", "definition", "segment", "frame",
        "other",
    }


def test_bounding_box_as_dict():
    bbox = BoundingBox(x0=1, y0=2, x1=3, y1=4)
    assert bbox.as_dict() == {"x0": 1, "y0": 2, "x1": 3, "y1": 4}


def test_parsed_document_text_and_as_dict():
    block = ParsedBlock(
        block_type=BlockType.TEXT,
        content="hello",
        page_number=1,
        bbox=BoundingBox(0, 0, 10, 10),
        metadata={"k": "v"},
    )
    page = ParsedPage(page_number=1, width=612, height=792, blocks=[block])
    doc = ParsedDocument(
        source="sample.pdf",
        parser_name="dummy",
        parser_version="0.0.1",
        metadata={"title": "Sample"},
        pages=[page],
        warnings=["w"],
    )
    assert doc.page_count == 1
    assert doc.text == "hello"
    data = doc.as_dict()
    assert data["pages"][0]["blocks"][0]["bbox"] == {"x0": 0, "y0": 0, "x1": 10, "y1": 10}
    assert data["pages"][0]["blocks"][0]["block_type"] == "text"
    assert data["warnings"] == ["w"]


def test_exception_hierarchy():
    assert issubclass(ParserDependencyError, ParserError)
    assert issubclass(DocumentParseError, ParserError)
    assert issubclass(CorruptDocumentError, DocumentParseError)
    assert issubclass(EmptyDocumentError, DocumentParseError)
    assert issubclass(EncryptedDocumentError, DocumentParseError)


def test_parser_protocol_runtime_check():
    class DummyParser:
        name = "dummy"

        def parse(self, source):
            return ParsedDocument(source=str(source), parser_name=self.name, parser_version="0")

    assert isinstance(DummyParser(), Parser)

    class NotAParser:
        pass

    assert not isinstance(NotAParser(), Parser)
