from pathlib import Path
import pytest
from parsers.contracts import (
    BlockType,
    CorruptDocumentError,
    EmptyDocumentError,
    ParsedDocument,
)
from parsers.code.tree_sitter import TreeSitterParser

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "parsers" / "code" / "fixtures"


@pytest.mark.parametrize(
    "fixture_file,expected_lang,expected_name",
    [
        ("sample.py", "python", "Calculator"),
        ("sample.js", "javascript", "TaskRunner"),
        ("sample.ts", "typescript", "UserService"),
        ("sample.java", "java", "DataPipeline"),
        ("sample.c", "c", "multiply"),
        ("sample.cpp", "cpp", "VectorBuffer"),
        ("sample.go", "go", "Worker"),
    ],
)
def test_tree_sitter_multilang(fixture_file, expected_lang, expected_name):
    parser = TreeSitterParser()
    doc = parser.parse(FIXTURES_DIR / fixture_file)
    assert isinstance(doc, ParsedDocument)
    assert doc.parser_name == "tree_sitter"
    assert doc.metadata["language"] == expected_lang
    names = {b.metadata.get("name") for b in doc.blocks if b.block_type == BlockType.DEFINITION}
    assert expected_name in names
    # Verify line positions
    for b in doc.blocks:
        if b.block_type == BlockType.DEFINITION:
            assert b.metadata.get("start_line", 0) > 0


def test_tree_sitter_empty():
    parser = TreeSitterParser()
    with pytest.raises(EmptyDocumentError):
        parser.parse(FIXTURES_DIR / "empty.py")
