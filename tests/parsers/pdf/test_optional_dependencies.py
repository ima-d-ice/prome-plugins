"""Optional dependency handling: missing extras must raise helpful errors,
and selecting one parser must never fall back to another."""

import sys

import pytest

from parsers.contracts import ParserDependencyError
from tests.parsers.pdf.conftest import make_parser


def _block_import(monkeypatch, *modules):
    for module in modules:
        monkeypatch.setitem(sys.modules, module, None)


def test_pymupdf_missing_dependency(monkeypatch, fixtures_dir):
    _block_import(monkeypatch, "pymupdf", "fitz")
    parser = make_parser("pymupdf")
    with pytest.raises(ParserDependencyError) as excinfo:
        parser.parse(fixtures_dir / "simple_digital.pdf")
    assert "pip install" in str(excinfo.value)


def test_pypdf_missing_dependency(monkeypatch, fixtures_dir):
    _block_import(monkeypatch, "pypdf")
    parser = make_parser("pypdf")
    with pytest.raises(ParserDependencyError) as excinfo:
        parser.parse(fixtures_dir / "simple_digital.pdf")
    assert "pip install" in str(excinfo.value)


def test_docling_missing_dependency(monkeypatch, fixtures_dir):
    _block_import(monkeypatch, "docling", "docling.document_converter", "docling.datamodel", "docling.datamodel.pipeline_options", "docling.datamodel.base_models")
    parser = make_parser("docling")
    with pytest.raises(ParserDependencyError) as excinfo:
        parser.parse(fixtures_dir / "simple_digital.pdf")
    assert "pip install" in str(excinfo.value)


@pytest.mark.parametrize("name", ["pymupdf", "pypdf", "docling"])
def test_dependency_error_does_not_fall_back(monkeypatch, fixtures_dir, name):
    if name == "pymupdf":
        _block_import(monkeypatch, "pymupdf", "fitz")
    elif name == "pypdf":
        _block_import(monkeypatch, "pypdf")
    else:
        _block_import(monkeypatch, "docling", "docling.document_converter")
    parser = make_parser(name)
    # Must raise, never silently produce a document via another backend.
    with pytest.raises(ParserDependencyError):
        parser.parse(fixtures_dir / "simple_digital.pdf")
