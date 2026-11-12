import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = ROOT / "benchmarks" / "parsers" / "pdf" / "fixtures"
EXPECTATIONS_PATH = ROOT / "benchmarks" / "parsers" / "pdf" / "expectations.json"

PARSER_NAMES = ["pymupdf", "pypdf", "docling"]


def parser_importable(name: str) -> bool:
    module = {"pymupdf": "pymupdf", "pypdf": "pypdf", "docling": "docling"}[name]
    try:
        __import__(module)
    except ImportError:
        return False
    return True


def make_parser(name: str):
    if name == "pymupdf":
        from parsers.pdf.pymupdf import PyMuPDFParser

        return PyMuPDFParser()
    if name == "pypdf":
        from parsers.pdf.pypdf import PyPDFParser

        return PyPDFParser()
    if name == "docling":
        from parsers.pdf.docling import DoclingParser

        return DoclingParser()
    raise ValueError(name)


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    if not FIXTURES_DIR.exists() or not any(FIXTURES_DIR.glob("*.pdf")):
        pytest.skip("PDF fixtures are missing; run benchmarks/parsers/pdf/generate_fixtures.py")
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def expectations() -> dict:
    if not EXPECTATIONS_PATH.exists():
        pytest.skip("expectations.json is missing")
    return json.loads(EXPECTATIONS_PATH.read_text())


def parser_params():
    return [
        pytest.param(name, marks=pytest.mark.skipif(not parser_importable(name), reason=f"{name} not installed"))
        for name in PARSER_NAMES
    ]


@pytest.fixture(params=parser_params(), scope="module")
def parser_name(request):
    return request.param


@pytest.fixture()
def parser(parser_name):
    return make_parser(parser_name)
