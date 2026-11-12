"""Unit tests for the benchmark scoring model: capability-aware N/A handling,
table evaluation, quality/performance scores, and report archiving."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from parsers.contracts import BlockType, BoundingBox, ParsedBlock, ParsedDocument, ParsedPage

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_PATH = ROOT / "benchmarks" / "parsers" / "pdf" / "benchmark.py"
FIXTURES_DIR = ROOT / "benchmarks" / "parsers" / "pdf" / "fixtures"
EXPECTATIONS_PATH = ROOT / "benchmarks" / "parsers" / "pdf" / "expectations.json"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("prome_pdf_benchmark", BENCHMARK_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = _load_benchmark()
EXPECTATIONS = json.loads(EXPECTATIONS_PATH.read_text())

CAPS_FULL = benchmark.PARSER_CAPABILITIES["docling"]      # text/structure/tables/images/lists, no OCR
CAPS_PYPDF = benchmark.PARSER_CAPABILITIES["pypdf"]       # text only
CAPS_PYMUPDF = benchmark.PARSER_CAPABILITIES["pymupdf"]   # no lists


def _block(block_type, content, page_number=1, bbox=None):
    return ParsedBlock(block_type=block_type, content=content, page_number=page_number, bbox=bbox)


def _doc(pages):
    return ParsedDocument(source="test.pdf", parser_name="test", parser_version="1.0", pages=pages)


def _page(blocks, page_number=1):
    return ParsedPage(page_number=page_number, width=612, height=792, blocks=blocks)


def _table_doc():
    return _doc([
        _page([
            _block(BlockType.HEADING, "Invoice Summary", 1),
            _block(BlockType.TABLE, "Item | Qty | Price\nWidget | 3 | 19.99\nGadget | 7 | 5.50\nGizmo | 2 | 42.00", 1),
            _block(BlockType.TEXT, "Totals include all applicable taxes and fees.", 1),
        ], 1)
    ])


TABLE_EXPECTATION = {
    "expected_pages": 1,
    "expected_text": ["Invoice Summary", "Widget"],
    "expected_headings": ["Invoice Summary"],
    "expected_tables": 1,
    "expected_table_header": ["Item", "Qty", "Price"],
    "expected_table_cells": ["Widget", "Gadget", "Gizmo"],
    "expected_table_pages": {"1": ["Widget", "Gizmo"]},
    "blank_pages": [],
    "reading_order": ["Invoice Summary", "Widget", "Totals include"],
}


def test_table_content_scored_against_ground_truth_not_block_type():
    doc = _table_doc()
    metrics = benchmark.score_document(doc, TABLE_EXPECTATION, CAPS_PYMUPDF)
    assert metrics["table_cell_coverage"] == 1.0
    assert metrics["table_header_coverage"] == 1.0
    assert metrics["multipage_table_continuity"] == 1.0
    assert metrics["table_blocks"] == 1


def test_table_content_partial_match():
    doc = _table_doc()
    doc.pages[0].blocks[1] = _block(BlockType.TABLE, "Item | Qty | Price\nWidget | 3 | 19.99", 1)
    metrics = benchmark.score_document(doc, TABLE_EXPECTATION, CAPS_PYMUPDF)
    assert metrics["table_cell_coverage"] == 0.3333
    assert metrics["multipage_table_continuity"] == 0.5


def test_table_block_type_alone_is_not_enough():
    doc = _table_doc()
    doc.pages[0].blocks[1] = _block(BlockType.TABLE, "Item | Qty | Price\nNothing expected here", 1)
    metrics = benchmark.score_document(doc, TABLE_EXPECTATION, CAPS_PYMUPDF)
    assert metrics["table_cell_coverage"] == 0.0
    assert metrics["structure_preservation"] == 0.5  # heading found, table cells not


def test_table_duplication_reported():
    expected = dict(TABLE_EXPECTATION)
    dup_doc = _doc([
        _page([
            _block(BlockType.TABLE, "Widget | Gadget | Gizmo", 1),
            _block(BlockType.TEXT, "Widget Gadget Gizmo also appear in plain text", 1),
        ], 1)
    ])
    metrics = benchmark.score_document(dup_doc, expected, CAPS_PYMUPDF)
    assert metrics["table_text_duplication"] == 1.0


def test_pypdf_table_capability_is_na_not_zero():
    doc = _table_doc()
    metrics = benchmark.score_document(doc, TABLE_EXPECTATION, CAPS_PYPDF)
    assert metrics["table_cell_coverage"] is None
    assert metrics["table_header_coverage"] is None
    assert metrics["multipage_table_continuity"] is None
    assert metrics["structure_preservation"] is None  # pypdf has no structure capability
    assert metrics["heading_preservation"] is None
    assert metrics["quality_score"] is not None  # text/reading-order still scored


def test_scanned_fixture_is_unsupported_not_failed():
    scanned = EXPECTATIONS["scanned.pdf"]
    assert scanned.get("requires_ocr") is True
    doc = _doc([_page([_block(BlockType.IMAGE, "", 1)], 1), _page([_block(BlockType.IMAGE, "", 2)], 2)])
    for name in ("pymupdf", "pypdf", "docling"):
        metrics = benchmark.score_document(doc, scanned, benchmark.PARSER_CAPABILITIES[name])
        assert metrics["status"] == "unsupported", name
        assert metrics["unsupported_capability"] == "ocr"
        assert metrics["quality_score"] is None
        assert metrics["text_coverage"] is None


def test_failed_metrics_zero_for_applicable_na_for_inapplicable():
    metrics = benchmark.failed_metrics(TABLE_EXPECTATION, CAPS_PYPDF)
    assert metrics["status"] == "failed"
    assert metrics["text_coverage"] == 0.0          # applicable, achieved nothing
    assert metrics["structure_preservation"] is None  # capability missing -> N/A
    assert metrics["table_cell_coverage"] is None     # capability missing -> N/A
    assert metrics["image_detection"] is None         # no images expected
    assert metrics["quality_score"] == 0.0


def test_quality_score_excludes_na_components():
    metrics = benchmark.score_document(_table_doc(), TABLE_EXPECTATION, CAPS_PYPDF)
    assert set(benchmark.QUALITY_COMPONENT_KEYS) >= {
        "text_coverage", "structure_preservation", "reading_order", "image_detection", "list_detection",
    }
    # pypdf on this fixture: text=1.0, reading_order=1.0, rest N/A
    assert metrics["quality_score"] == 1.0


def test_quality_score_mixed_components():
    expected = dict(TABLE_EXPECTATION)
    expected = {**expected, "expected_text": ["Invoice Summary", "absent-snippet-xyz"]}
    metrics = benchmark.score_document(_table_doc(), expected, CAPS_PYMUPDF)
    # text 0.5, structure (heading 1.0 + table cells 1.0) -> 1.0, reading order 1.0
    assert metrics["text_coverage"] == 0.5
    assert metrics["structure_preservation"] == 1.0
    assert metrics["quality_score"] == pytest.approx((0.5 + 1.0 + 1.0) / 3, abs=1e-4)


def test_unsupported_row_is_never_scored():
    scanned = EXPECTATIONS["scanned.pdf"]
    doc = _doc([_page([], 1), _page([], 2)])
    metrics = benchmark.score_document(doc, scanned, CAPS_PYMUPDF)
    assert metrics["status"] == "unsupported"
    assert metrics["quality_score"] is None


def test_reading_order_scoring():
    expected = {**TABLE_EXPECTATION, "reading_order": ["Invoice Summary", "Widget", "Totals include"]}
    metrics = benchmark.score_document(_table_doc(), expected, CAPS_PYMUPDF)
    assert metrics["reading_order"] == 1.0
    reversed_doc = _doc([
        _page([
            _block(BlockType.TEXT, "Totals include all applicable taxes and fees.", 1),
            _block(BlockType.TABLE, "Widget", 1),
            _block(BlockType.HEADING, "Invoice Summary", 1),
        ], 1)
    ])
    metrics = benchmark.score_document(reversed_doc, expected, CAPS_PYMUPDF)
    assert metrics["reading_order"] == 0.0


def test_image_and_list_detection():
    expected = {
        **TABLE_EXPECTATION,
        "expected_images": 1,
        "expected_list_items": ["Alpha", "Beta"],
    }
    doc = _doc([
        _page([
            _block(BlockType.HEADING, "Invoice Summary", 1),
            _block(BlockType.TABLE, "Item | Qty | Price\nWidget | 3 | 19.99", 1),
            _block(BlockType.TEXT, "Totals include all applicable taxes and fees. Widget", 1),
            _block(BlockType.IMAGE, "", 1),
            _block(BlockType.LIST, "Alpha", 1),
        ], 1)
    ])
    metrics = benchmark.score_document(doc, expected, CAPS_FULL)
    assert metrics["image_detection"] == 1.0
    assert metrics["list_detection"] == 0.5  # only "Alpha" in list blocks
    metrics_pymupdf = benchmark.score_document(doc, expected, CAPS_PYMUPDF)
    assert metrics_pymupdf["list_detection"] is None  # no list capability -> N/A
    metrics_pypdf = benchmark.score_document(doc, expected, CAPS_PYPDF)
    assert metrics_pypdf["image_detection"] is None


def test_archive_existing_reports(tmp_path):
    (tmp_path / "benchmark_report.json").write_text("{}")
    (tmp_path / "benchmark_results.csv").write_text("a,b\n")
    archived = benchmark.archive_existing_reports(tmp_path)
    assert len(archived) == 2
    for dst in archived:
        assert Path(dst).exists()
        assert Path(dst).parent == tmp_path / "archive"
    # originals still in place (archive is a copy, not a move)
    assert (tmp_path / "benchmark_report.json").exists()


def test_archive_no_existing_reports(tmp_path):
    assert benchmark.archive_existing_reports(tmp_path) == []
    assert not (tmp_path / "archive").exists()


def test_multipage_table_continuity_scored_per_page():
    expected = EXPECTATIONS["multipage_table.pdf"]
    doc = _doc([
        _page([_block(BlockType.TABLE, "SKU | Name\nSKU-001 | Component 001\nComponent 015\nSKU-030\nComponent 030", 1)], 1),
        _page([_block(BlockType.TABLE, "SKU | Name\nSKU-031\nComponent 045\nSKU-060\nComponent 060", 2)], 2),
    ])
    metrics = benchmark.score_document(doc, expected, CAPS_PYMUPDF)
    assert metrics["multipage_table_continuity"] == 1.0
    # second page's cells missing -> continuity drops
    doc.pages[1].blocks[0] = _block(BlockType.TABLE, "SKU | Name", 2)
    metrics = benchmark.score_document(doc, expected, CAPS_PYMUPDF)
    assert metrics["multipage_table_continuity"] < 1.0


def test_expectations_schema_keys():
    for filename, expected in EXPECTATIONS.items():
        assert "expected_pages" in expected, filename
        assert "expected_text" in expected, filename
        assert "expected_headings" in expected, filename
        assert "expected_tables" in expected, filename
        assert "reading_order" in expected, filename
        assert "blank_pages" in expected, filename
