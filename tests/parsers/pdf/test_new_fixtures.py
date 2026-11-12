"""Tests for the challenging synthetic fixtures added to the benchmark corpus.

Ground-truth validation (fixtures vs expectations.json) is parser-independent
and uses PyMuPDF directly. Parser behaviour assertions are capability-aware.
"""

import json
from pathlib import Path

import pytest

from parsers.contracts import BlockType

ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = ROOT / "benchmarks" / "parsers" / "pdf" / "fixtures"
EXPECTATIONS_PATH = ROOT / "benchmarks" / "parsers" / "pdf" / "expectations.json"

NEW_FIXTURES = [
    "two_column.pdf",
    "headers_footers.pdf",
    "rotated.pdf",
    "multipage_table.pdf",
    "images_captions.pdf",
    "scanned.pdf",
    "mixed_sizes.pdf",
]

with EXPECTATIONS_PATH.open() as handle:
    EXPECTATIONS = json.load(handle)


def test_new_fixtures_exist():
    for name in NEW_FIXTURES:
        assert (FIXTURES_DIR / name).exists(), f"missing fixture {name}"


def test_new_fixtures_are_valid_pdfs():
    pymupdf = pytest.importorskip("pymupdf", reason="pymupdf not installed")
    for name in NEW_FIXTURES:
        doc = pymupdf.open(FIXTURES_DIR / name)
        assert doc.page_count == EXPECTATIONS[name]["expected_pages"], name
        assert doc.metadata.get("title"), name
        doc.close()


def test_new_fixtures_have_ground_truth():
    for name in NEW_FIXTURES:
        expected = EXPECTATIONS[name]
        for key in (
            "expected_pages",
            "expected_text",
            "expected_headings",
            "expected_tables",
            "blank_pages",
            "reading_order",
        ):
            assert key in expected, f"{name}: missing expectations key {key}"
        assert isinstance(expected["expected_text"], list)
        assert isinstance(expected["reading_order"], list)


def test_expected_text_present_in_fixture_pymupdf():
    """Ground truth must match the PDF itself, independent of the parsers."""
    pymupdf = pytest.importorskip("pymupdf", reason="pymupdf not installed")
    for name in NEW_FIXTURES:
        expected = EXPECTATIONS[name]
        if expected.get("requires_ocr"):
            continue  # image-only: no text layer by construction
        doc = pymupdf.open(FIXTURES_DIR / name)
        text = "\n".join(page.get_text() for page in doc).lower()
        doc.close()
        for snippet in expected["expected_text"]:
            assert snippet.lower() in text, f"{name}: snippet {snippet!r} not in fixture source text"


def test_expected_pages_match_fixture():
    pymupdf = pytest.importorskip("pymupdf", reason="pymupdf not installed")
    for name in NEW_FIXTURES:
        doc = pymupdf.open(FIXTURES_DIR / name)
        assert doc.page_count == EXPECTATIONS[name]["expected_pages"], name
        doc.close()


def test_table_page_markers_match_fixture():
    """Each multipage-table cell marker must appear on the expected page."""
    expected = EXPECTATIONS["multipage_table.pdf"]
    pymupdf = pytest.importorskip("pymupdf", reason="pymupdf not installed")
    doc = pymupdf.open(FIXTURES_DIR / "multipage_table.pdf")
    page_text = {str(i + 1): page.get_text() for i, page in enumerate(doc)}
    doc.close()
    for page_no, cells in expected["expected_table_pages"].items():
        assert page_no in page_text
        for cell in cells:
            assert cell in page_text[page_no], f"page {page_no}: {cell!r} not found"


def test_scanned_fixture_has_no_text_layer():
    pymupdf = pytest.importorskip("pymupdf", reason="pymupdf not installed")
    doc = pymupdf.open(FIXTURES_DIR / "scanned.pdf")
    for page in doc:
        assert page.get_text().strip() == "", "scanned fixture must not carry an extractable text layer"
        assert page.get_images(), "scanned fixture pages must contain rendered images"
    doc.close()


def test_rotated_fixture_has_rotated_page():
    pymupdf = pytest.importorskip("pymupdf", reason="pymupdf not installed")
    doc = pymupdf.open(FIXTURES_DIR / "rotated.pdf")
    rotations = [page.rotation for page in doc]
    doc.close()
    assert 90 in rotations or 270 in rotations


def test_two_column_fixture_two_block_streams():
    pymupdf = pytest.importorskip("pymupdf", reason="pymupdf not installed")
    doc = pymupdf.open(FIXTURES_DIR / "two_column.pdf")
    blocks = doc[0].get_text("blocks")
    doc.close()
    left = [b for b in blocks if b[0] < 200]
    right = [b for b in blocks if b[0] >= 300]
    assert left and right, "expected distinct left and right column content"


def test_headers_footers_fixture_repeats_boilerplate():
    pymupdf = pytest.importorskip("pymupdf", reason="pymupdf not installed")
    doc = pymupdf.open(FIXTURES_DIR / "headers_footers.pdf")
    texts = [page.get_text() for page in doc]
    doc.close()
    for text in texts:
        assert "PROME SYNTHETIC REPORT" in text
        assert "Synthetic benchmark fixture" in text
    assert [t for t in texts] != [texts[0]] * 3  # page numbers differ
    assert "PAGE-01" in texts[0] and "PAGE-02" in texts[1] and "PAGE-03" in texts[2]


# --- parser capability behaviour on the new fixtures ---

def test_two_column_reading_order_pymupdf(fixtures_dir):
    pytest.importorskip("pymupdf", reason="pymupdf not installed")
    from parsers.pdf.pymupdf import PyMuPDFParser

    doc = PyMuPDFParser().parse(fixtures_dir / "two_column.pdf")
    text = doc.text.upper()
    alpha = [text.find(f"ALPHA-MARKER-{n:02d}") for n in (1, 2, 3, 4, 5)]
    beta = [text.find(f"BETA-MARKER-{n:02d}") for n in (1, 2, 3, 4, 5)]
    assert all(p >= 0 for p in alpha + beta)
    assert max(alpha) < min(beta), "left column must be extracted fully before the right column"


def test_two_column_reading_order_pypdf(fixtures_dir):
    pytest.importorskip("pypdf", reason="pypdf not installed")
    from parsers.pdf.pypdf import PyPDFParser

    doc = PyPDFParser().parse(fixtures_dir / "two_column.pdf")
    text = doc.text.upper()
    alpha = [text.find(f"ALPHA-MARKER-{n:02d}") for n in (1, 2, 3)]
    beta = [text.find(f"BETA-MARKER-{n:02d}") for n in (1, 2, 3)]
    assert all(p >= 0 for p in alpha + beta)
    assert max(alpha) < min(beta)


def test_pymupdf_rotated_text_extracted(fixtures_dir):
    pytest.importorskip("pymupdf", reason="pymupdf not installed")
    from parsers.pdf.pymupdf import PyMuPDFParser

    doc = PyMuPDFParser().parse(fixtures_dir / "rotated.pdf")
    text = doc.text
    assert "ROTATED-MARKER-90" in text
    assert "ROTATED-PAGE-MARKER-2" in text


def test_scanned_fixture_extracts_no_text(fixtures_dir, parser_name):
    doc = None
    if parser_name == "pymupdf":
        pytest.importorskip("pymupdf", reason="pymupdf not installed")
        from parsers.pdf.pymupdf import PyMuPDFParser

        doc = PyMuPDFParser().parse(fixtures_dir / "scanned.pdf")
    elif parser_name == "pypdf":
        pytest.importorskip("pypdf", reason="pypdf not installed")
        from parsers.pdf.pypdf import PyPDFParser

        doc = PyPDFParser().parse(fixtures_dir / "scanned.pdf")
    else:
        pytest.importorskip("docling", reason="docling not installed")
        from parsers.pdf.docling import DoclingParser

        doc = DoclingParser().parse(fixtures_dir / "scanned.pdf")
    assert doc.page_count == 2
    assert doc.text.strip() == ""


def test_multipage_table_blocks_detected(fixtures_dir, parser_name):
    if parser_name == "pymupdf":
        pytest.importorskip("pymupdf", reason="pymupdf not installed")
        from parsers.pdf.pymupdf import PyMuPDFParser

        doc = PyMuPDFParser().parse(fixtures_dir / "multipage_table.pdf")
    elif parser_name == "docling":
        pytest.importorskip("docling", reason="docling not installed")
        from parsers.pdf.docling import DoclingParser

        doc = DoclingParser().parse(fixtures_dir / "multipage_table.pdf")
    else:
        pytest.skip("pypdf has no table detection capability")
    tables = [b for page in doc.pages for b in page.blocks if b.block_type == BlockType.TABLE]
    assert tables, "expected table blocks on the multipage table fixture"
    table_text = "\n".join(t.content for t in tables)
    for header_cell in ("SKU", "Name", "Qty", "Price"):
        assert header_cell in table_text, header_cell
    for page_no in ("1", "2"):
        per_page = "\n".join(
            b.content for page in doc.pages if str(page.page_number) == page_no
            for b in page.blocks if b.block_type == BlockType.TABLE
        )
        assert per_page, f"no table extracted on page {page_no}"
    assert "SKU-001" in table_text
    assert "SKU-060" in table_text


def test_images_detected(fixtures_dir, parser_name):
    if parser_name == "pypdf":
        pytest.skip("pypdf does not surface images as blocks")
    if parser_name == "pymupdf":
        pytest.importorskip("pymupdf", reason="pymupdf not installed")
        from parsers.pdf.pymupdf import PyMuPDFParser

        doc = PyMuPDFParser().parse(fixtures_dir / "images_captions.pdf")
    else:
        pytest.importorskip("docling", reason="docling not installed")
        from parsers.pdf.docling import DoclingParser

        doc = DoclingParser().parse(fixtures_dir / "images_captions.pdf")
    image_blocks = [b for page in doc.pages for b in page.blocks if b.block_type == BlockType.IMAGE]
    assert image_blocks, "expected at least one image block"
    captions = doc.text
    assert "Figure 1" in captions
    assert "Figure 2" in captions


def test_list_detection_capability(fixtures_dir, parser_name):
    if parser_name == "pypdf":
        pytest.skip("pypdf does not surface lists as blocks")
    if parser_name == "pymupdf":
        pytest.importorskip("pymupdf", reason="pymupdf not installed")
        from parsers.pdf.pymupdf import PyMuPDFParser

        doc = PyMuPDFParser().parse(fixtures_dir / "mixed_sizes.pdf")
        list_blocks = [b for page in doc.pages for b in page.blocks if b.block_type == BlockType.LIST]
        assert list_blocks == [], "pymupdf parser has no list detection; items arrive as text"
        assert "ITEM-ALPHA-1" in doc.text
    else:
        pytest.importorskip("docling", reason="docling not installed")
        from parsers.pdf.docling import DoclingParser

        doc = DoclingParser().parse(fixtures_dir / "mixed_sizes.pdf")
        list_blocks = [b for page in doc.pages for b in page.blocks if b.block_type == BlockType.LIST]
        list_text = "\n".join(b.content for b in list_blocks)
        assert "ITEM-ALPHA-1" in list_text
        assert "ITEM-ALPHA-2" in list_text
        assert "ITEM-ALPHA-3" in list_text


def test_mixed_sizes_headings(fixtures_dir, parser_name):
    if parser_name == "pypdf":
        pytest.skip("pypdf does not detect headings")
    if parser_name == "pymupdf":
        pytest.importorskip("pymupdf", reason="pymupdf not installed")
        from parsers.pdf.pymupdf import PyMuPDFParser

        doc = PyMuPDFParser().parse(fixtures_dir / "mixed_sizes.pdf")
    else:
        pytest.importorskip("docling", reason="docling not installed")
        from parsers.pdf.docling import DoclingParser

        doc = DoclingParser().parse(fixtures_dir / "mixed_sizes.pdf")
    headings = [b.content for page in doc.pages for b in page.blocks if b.block_type == BlockType.HEADING]
    joined = " ".join(headings)
    assert "Mixed Sizes Overview" in joined
    assert "Section Alpha" in joined
    assert "Section Beta" in joined


def test_headers_footers_body_extracted_pymupdf(fixtures_dir):
    pytest.importorskip("pymupdf", reason="pymupdf not installed")
    from parsers.pdf.pymupdf import PyMuPDFParser

    doc = PyMuPDFParser().parse(fixtures_dir / "headers_footers.pdf")
    for index, page in enumerate(doc.pages, start=1):
        assert "PROME SYNTHETIC REPORT" in page.text
        assert f"PAGE-{index:02d}" in page.text
        assert f"Synthetic benchmark fixture - page {index}" in page.text
    combined = doc.text
    assert "REPORT-ALPHA-MARKER" in combined
    assert "REPORT-GAMMA-MARKER" in combined
