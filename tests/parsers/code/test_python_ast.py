from pathlib import Path
import pytest
from parsers.contracts import (
    BlockType,
    CorruptDocumentError,
    EmptyDocumentError,
    ParsedDocument,
)
from parsers.code.python_ast import PythonAstParser

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "parsers" / "code" / "fixtures"


def test_python_ast_sample():
    parser = PythonAstParser()
    doc = parser.parse(FIXTURES_DIR / "sample.py")
    assert isinstance(doc, ParsedDocument)
    assert doc.parser_name == "python_ast"
    assert doc.metadata["classes_count"] == 1
    assert doc.metadata["functions_count"] == 1
    assert doc.metadata["methods_count"] == 2
    assert "Prome Code Extraction Module" in doc.metadata["module_docstring"]

    names = {b.metadata.get("name") for b in doc.blocks if b.block_type == BlockType.DEFINITION}
    assert "Calculator" in names
    assert "add" in names
    assert "compute_hypotenuse" in names

    # Check method parent class relationship
    add_block = [b for b in doc.blocks if b.metadata.get("name") == "add"][0]
    assert add_block.metadata["parent_class"] == "Calculator"
    assert add_block.metadata["start_line"] > 0


def test_python_ast_empty():
    parser = PythonAstParser()
    with pytest.raises(EmptyDocumentError):
        parser.parse(FIXTURES_DIR / "empty.py")


def test_python_ast_malformed():
    parser = PythonAstParser()
    with pytest.raises(CorruptDocumentError):
        parser.parse(FIXTURES_DIR / "malformed.py")
