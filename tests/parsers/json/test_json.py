from pathlib import Path
import pytest
from parsers.contracts import (
    BlockType,
    CorruptDocumentError,
    EmptyDocumentError,
    ParsedDocument,
)
from parsers.json import JsonParser

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "parsers" / "json" / "fixtures"


def test_json_nested_object():
    parser = JsonParser()
    doc = parser.parse(FIXTURES_DIR / "nested_object.json")
    assert isinstance(doc, ParsedDocument)
    assert doc.parser_name == "json_stdlib"
    assert doc.metadata["root_type"] == "dict"
    assert "Prome Technologies" in doc.text

    paths = {b.metadata.get("json_path") for b in doc.blocks if b.block_type == BlockType.KEY_VALUE}
    assert "$.company" in paths
    assert "$.headquarters.city" in paths
    assert "$.departments[0].name" in paths


def test_json_primitives():
    parser = JsonParser()
    doc = parser.parse(FIXTURES_DIR / "primitives.json")
    types = {b.metadata.get("data_type") for b in doc.blocks if b.block_type == BlockType.KEY_VALUE}
    assert "str" in types
    assert "int" in types
    assert "float" in types
    assert "bool" in types
    assert "null" in types
    assert "Café" in doc.text


def test_json_ndjson():
    parser = JsonParser()
    doc = parser.parse(FIXTURES_DIR / "stream.ndjson")
    assert doc.metadata["root_type"] == "ndjson"
    record_blocks = [b for b in doc.blocks if b.block_type == BlockType.RECORD]
    assert len(record_blocks) == 3
    assert "RAG parsers" in doc.text


def test_json_empty():
    parser = JsonParser()
    with pytest.raises(EmptyDocumentError):
        parser.parse(FIXTURES_DIR / "empty.json")


def test_json_malformed():
    parser = JsonParser()
    with pytest.raises(CorruptDocumentError):
        parser.parse(FIXTURES_DIR / "malformed.json")
