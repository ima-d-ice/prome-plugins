from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PDF_FIXTURES = ROOT / "benchmarks" / "parsers" / "pdf" / "fixtures"
HTML_FIXTURES = ROOT / "benchmarks" / "parsers" / "html" / "fixtures"
OFFICE_FIXTURES = ROOT / "benchmarks" / "parsers" / "office" / "fixtures"
SPREADSHEET_FIXTURES = ROOT / "benchmarks" / "parsers" / "spreadsheets" / "fixtures"


def _make_parser(family: str, name: str):
    if name == "pymupdf":
        from parsers.pdf.pymupdf import PyMuPDFParser

        return PyMuPDFParser()
    if name == "beautifulsoup":
        from parsers.html.beautifulsoup import BeautifulSoupParser

        return BeautifulSoupParser()
    if name == "python_docx":
        from parsers.office.docx.python_docx import PythonDocxParser

        return PythonDocxParser()
    if name == "python_pptx":
        from parsers.office.pptx.python_pptx import PythonPptxParser

        return PythonPptxParser()
    if name == "openpyxl":
        from parsers.spreadsheets.xlsx.openpyxl import OpenpyxlParser

        return OpenpyxlParser()
    if name == "csv":
        from parsers.spreadsheets.csv import CsvParser

        return CsvParser()
    raise ValueError(name)


def _lib_importable(name: str) -> bool:
    module = {
        "pymupdf": "pymupdf",
        "beautifulsoup": "bs4",
        "python_docx": "docx",
        "python_pptx": "pptx",
        "openpyxl": "openpyxl",
        "csv": None,
    }[name]
    if module is None:
        return True
    try:
        __import__(module)
        return True
    except ImportError:
        return False


# (family, parser_name, filename) — fixtures shipped with the parser benchmark suites.
FIXTURE_CASES = [
    ("pdf", "pymupdf", "simple_digital.pdf"),
    ("pdf", "pymupdf", "multipage_structured.pdf"),
    ("pdf", "pymupdf", "tables.pdf"),
    ("pdf", "pymupdf", "multipage_table.pdf"),
    ("html", "beautifulsoup", "documentation_headings.html"),
    ("html", "beautifulsoup", "nested_lists.html"),
    ("html", "beautifulsoup", "complex_table.html"),
    ("html", "beautifulsoup", "long_document.html"),
    ("html", "beautifulsoup", "unicode_text.html"),
    ("html", "beautifulsoup", "simple_article.html"),
    ("office", "python_docx", "headings.docx"),
    ("office", "python_docx", "tables.docx"),
    ("office", "python_docx", "nested_lists.docx"),
    ("office", "python_docx", "simple_text.docx"),
    ("office", "python_pptx", "multi_slide.pptx"),
    ("office", "python_pptx", "table.pptx"),
    ("office", "python_pptx", "shapes.pptx"),
    ("office", "python_pptx", "images_notes.pptx"),
    ("spreadsheets", "openpyxl", "merged_cells.xlsx"),
    ("spreadsheets", "openpyxl", "multi_sheet.xlsx"),
    ("spreadsheets", "openpyxl", "data_types.xlsx"),
    ("spreadsheets", "csv", "standard.csv"),
    ("spreadsheets", "csv", "semicolon.csv"),
]


def fixture_path(family: str, filename: str) -> Path:
    return {
        "pdf": PDF_FIXTURES,
        "html": HTML_FIXTURES,
        "office": OFFICE_FIXTURES,
        "spreadsheets": SPREADSHEET_FIXTURES,
    }[family] / filename


def fixture_params():
    params = []
    for family, name, filename in FIXTURE_CASES:
        available = _lib_importable(name) and fixture_path(family, filename).exists()
        params.append(
            pytest.param(
                (family, name, filename),
                marks=pytest.mark.skipif(not available, reason=f"{name} unavailable or fixture missing"),
                id=filename,
            )
        )
    return params


@pytest.fixture(scope="module", params=fixture_params())
def parsed_fixture(request):
    family, name, filename = request.param
    parser = _make_parser(family, name)
    return parser.parse(fixture_path(family, filename))


def doc_is_empty(doc) -> bool:
    """True when the document carries no non-whitespace text."""
    return not doc.text.strip()
