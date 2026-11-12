"""Reading-order tests: two-column, headings+paragraphs, multi-page,
headers/footers and rotated content."""

import pytest

from parsers.contracts import BlockType, BoundingBox, ParsedBlock

pytest.importorskip("pymupdf", reason="pymupdf not installed")

from parsers.pdf.pymupdf.parser import _column_cut, _sort_reading_order


def _block(x0, y0, x1, y1, content, block_type=BlockType.TEXT):
    return ParsedBlock(
        block_type=block_type,
        content=content,
        page_number=1,
        bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1),
    )


def _contents(blocks):
    return [b.content for b in blocks]


def test_column_cut_detected_for_two_column_layout():
    blocks = [
        _block(54, 60, 160, 82, "Column Alpha", BlockType.HEADING),
        _block(324, 60, 430, 82, "Column Beta", BlockType.HEADING),
        _block(54, 110, 287, 255, "left body"),
        _block(324, 110, 558, 255, "right body"),
    ]
    cut = _column_cut(blocks, 612)
    assert cut is not None
    assert 250 < cut < 360


def test_column_cut_none_for_single_column():
    blocks = [
        _block(72, 60, 540, 90, "Heading spanning most of the page width", BlockType.HEADING),
        _block(72, 110, 540, 400, "body paragraph spanning most of the page width"),
    ]
    assert _column_cut(blocks, 612) is None


def test_two_column_sort_orders_columns_not_rows():
    blocks = [
        _block(54, 60, 160, 82, "heading alpha", BlockType.HEADING),
        _block(324, 60, 430, 82, "heading beta", BlockType.HEADING),
        _block(54, 110, 287, 255, "alpha first"),
        _block(324, 110, 558, 255, "beta first"),
        _block(54, 280, 287, 420, "alpha second"),
        _block(324, 280, 558, 420, "beta second"),
    ]
    _sort_reading_order(blocks, 612)
    assert _contents(blocks) == [
        "heading alpha",
        "alpha first",
        "alpha second",
        "heading beta",
        "beta first",
        "beta second",
    ]


def test_full_width_barrier_splits_bands():
    blocks = [
        _block(72, 30, 540, 44, "full width header"),
        _block(54, 110, 287, 255, "alpha body"),
        _block(324, 110, 558, 255, "beta body"),
        _block(72, 740, 540, 754, "full width footer"),
    ]
    _sort_reading_order(blocks, 612)
    assert _contents(blocks) == [
        "full width header",
        "alpha body",
        "beta body",
        "full width footer",
    ]


def test_sort_falls_back_to_y_then_x_without_columns():
    blocks = [
        _block(72, 200, 540, 260, "lower"),
        _block(72, 60, 540, 100, "upper"),
        _block(72, 130, 540, 180, "middle"),
    ]
    _sort_reading_order(blocks, 612)
    assert _contents(blocks) == ["upper", "middle", "lower"]


def test_blocks_without_bboxes_sort_last_and_stable():
    blocks = [
        _block(72, 60, 540, 100, "with bbox"),
        ParsedBlock(block_type=BlockType.TEXT, content="no bbox", page_number=1, bbox=None),
    ]
    _sort_reading_order(blocks, 612)
    assert _contents(blocks) == ["with bbox", "no bbox"]


def test_pymupdf_two_column_fixture_order(fixtures_dir):
    from parsers.pdf.pymupdf import PyMuPDFParser

    doc = PyMuPDFParser().parse(fixtures_dir / "two_column.pdf")
    text = doc.text.upper()
    markers = [
        "ALPHA-MARKER-01", "ALPHA-MARKER-02", "ALPHA-MARKER-03",
        "ALPHA-MARKER-04", "ALPHA-MARKER-05",
        "BETA-MARKER-01", "BETA-MARKER-02", "BETA-MARKER-03",
        "BETA-MARKER-04", "BETA-MARKER-05",
    ]
    positions = [text.find(m) for m in markers]
    assert all(p >= 0 for p in positions)
    assert positions == sorted(positions)


def test_headings_precede_their_paragraphs(fixtures_dir):
    from parsers.pdf.pymupdf import PyMuPDFParser

    doc = PyMuPDFParser().parse(fixtures_dir / "multipage_structured.pdf")
    for page in doc.pages:
        roles = [b.block_type for b in page.blocks]
        first_heading = next(i for i, t in enumerate(roles) if t == BlockType.HEADING)
        first_body = next(i for i, t in enumerate(roles) if t == BlockType.TEXT)
        assert first_heading < first_body


def test_multipage_content_order(fixtures_dir):
    from parsers.pdf.pymupdf import PyMuPDFParser

    doc = PyMuPDFParser().parse(fixtures_dir / "multi_section.pdf")
    text = doc.text
    assert 0 <= text.index("MARKER-01") < text.index("MARKER-03") < text.index("MARKER-05")


def test_headers_footers_order(fixtures_dir):
    from parsers.pdf.pymupdf import PyMuPDFParser

    doc = PyMuPDFParser().parse(fixtures_dir / "headers_footers.pdf")
    for page in doc.pages:
        text = page.text
        header_pos = text.index("PROME SYNTHETIC REPORT")
        body_pos = text.index("REPORT-")
        footer_pos = text.index("Synthetic benchmark fixture")
        assert header_pos < body_pos < footer_pos


def test_rotated_content_order(fixtures_dir):
    from parsers.pdf.pymupdf import PyMuPDFParser

    doc = PyMuPDFParser().parse(fixtures_dir / "rotated.pdf")
    text = doc.text
    assert text.index("NORMAL-MARKER-1") < text.index("Rotated Page Setup")
    assert "ROTATED-MARKER-90" in text
    assert "ROTATED-PAGE-MARKER-3" in text
