"""Unit tests for the HTML benchmark scoring model: capability-aware N/A
handling, scope-aware boilerplate scoring, table evaluation, quality and
performance scores, and report archiving."""

import importlib.util
import json
from pathlib import Path

import pytest

from parsers.contracts import BlockType, ParsedBlock, ParsedDocument

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_PATH = ROOT / "benchmarks" / "parsers" / "html" / "benchmark.py"
EXPECTATIONS_PATH = ROOT / "benchmarks" / "parsers" / "html" / "expectations.json"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("prome_html_benchmark", BENCHMARK_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = _load_benchmark()
EXPECTATIONS = json.loads(EXPECTATIONS_PATH.read_text())

CAPS_BSOUP = benchmark.PARSER_CAPABILITIES["beautifulsoup"]
CAPS_TRAFILATURA = benchmark.PARSER_CAPABILITIES["trafilatura"]


def _block(block_type, content, metadata=None):
    return ParsedBlock(block_type=block_type, content=content, page_number=None, metadata=metadata or {})


def _doc(blocks, metadata=None):
    return ParsedDocument(
        source="test.html",
        parser_name="test",
        parser_version="1.0",
        metadata=metadata or {"title": "T"},
        pages=[],
        blocks=blocks,
    )


def _simple_doc():
    return _doc([
        _block(BlockType.HEADING, "Getting Started with Widgets", {"heading_level": 1, "dom_order": 0}),
        _block(BlockType.TEXT, "Widgets are the fundamental building blocks of the example system.", {"dom_order": 1}),
        _block(BlockType.HEADING, "Installing Widgets", {"heading_level": 2, "dom_order": 2}),
        _block(BlockType.TEXT, "To install the widget library, add the package to your project dependencies.", {"dom_order": 3}),
    ])


TEXT_EXPECTATION = {
    "expected_text": ["Getting Started with Widgets", "fundamental building blocks"],
    "expected_headings": ["Getting Started with Widgets", "Installing Widgets"],
    "expected_heading_levels": {"Getting Started with Widgets": 1, "Installing Widgets": 2},
    "expected_tables": 0,
    "expected_list_items": [],
    "expected_link_hrefs": [],
    "expected_images": [],
    "reading_order": ["Getting Started with Widgets", "Installing Widgets"],
    "expected_metadata": ["title"],
    "boilerplate_text": [],
    "forbidden_text": [],
}


def test_text_and_structure_scored():
    doc = _simple_doc()
    metrics = benchmark.score_document(doc, TEXT_EXPECTATION, CAPS_BSOUP)
    assert metrics["text_coverage"] == 1.0
    assert metrics["heading_preservation"] == 1.0
    assert metrics["heading_level_accuracy"] == 1.0
    assert metrics["structure_preservation"] == 1.0
    assert metrics["reading_order"] == 1.0
    assert metrics["metadata_completeness"] == 1.0
    assert metrics["quality_score"] == 1.0


def test_wrong_heading_level_detected():
    doc = _doc([
        _block(BlockType.HEADING, "Getting Started with Widgets", {"heading_level": 3, "dom_order": 0}),
        _block(BlockType.HEADING, "Installing Widgets", {"heading_level": 2, "dom_order": 1}),
    ])
    metrics = benchmark.score_document(doc, TEXT_EXPECTATION, CAPS_BSOUP)
    assert metrics["heading_level_accuracy"] == 0.5


def test_table_content_scored():
    doc = _doc([
        _block(BlockType.TABLE, "Item | Qty\nWidget | 3\nGadget | 7", {"headers": ["Item", "Qty"], "dom_order": 0}),
    ])
    expected = {
        "expected_text": ["Widget"],
        "expected_headings": [],
        "expected_tables": 1,
        "expected_table_cells": ["Widget", "Gadget"],
        "expected_table_header": ["Item", "Qty"],
        "expected_list_items": [],
        "expected_link_hrefs": [],
        "expected_images": [],
        "reading_order": [],
        "expected_metadata": [],
        "boilerplate_text": [],
        "forbidden_text": [],
    }
    metrics = benchmark.score_document(doc, expected, CAPS_BSOUP)
    assert metrics["table_cell_coverage"] == 1.0
    assert metrics["table_header_coverage"] == 1.0
    assert metrics["structure_preservation"] == 1.0


def test_link_preservation_scored():
    doc = _doc([
        _block(BlockType.TEXT, "see the docs at reference example", {"dom_order": 0}),
        _block(BlockType.LINK, "reference example", {"href": "https://reference.example.com/overview", "dom_order": 1}),
    ])
    expected = {
        "expected_text": ["reference example"],
        "expected_headings": [],
        "expected_tables": 0,
        "expected_list_items": [],
        "expected_link_hrefs": ["https://reference.example.com/overview", "https://missing.example.com/nope"],
        "expected_images": [],
        "reading_order": [],
        "expected_metadata": [],
        "boilerplate_text": [],
        "forbidden_text": [],
    }
    metrics = benchmark.score_document(doc, expected, CAPS_BSOUP)
    assert metrics["link_preservation"] == 0.5


def test_images_na_for_trafilatura():
    doc = _doc([_block(BlockType.IMAGE, "diagram", {"alt": "diagram", "dom_order": 0})])
    expected = {
        "expected_text": ["diagram"],
        "expected_headings": [],
        "expected_tables": 0,
        "expected_list_items": [],
        "expected_link_hrefs": [],
        "expected_images": ["diagram"],
        "reading_order": [],
        "expected_metadata": [],
        "boilerplate_text": [],
        "forbidden_text": [],
    }
    metrics = benchmark.score_document(doc, expected, CAPS_TRAFILATURA)
    assert metrics["image_detection"] is None  # unsupported -> N/A, not 0
    metrics = benchmark.score_document(doc, expected, CAPS_BSOUP)
    assert metrics["image_detection"] == 1.0


def test_boilerplate_removal_credited_for_main_content_scope():
    doc = _doc([_block(BlockType.TEXT, "A well-designed navigation layout guides readers", {"dom_order": 0})])
    expected = {
        "expected_text": ["navigation layout"],
        "expected_headings": [],
        "expected_tables": 0,
        "expected_list_items": [],
        "expected_link_hrefs": [],
        "expected_images": [],
        "reading_order": [],
        "expected_metadata": [],
        "boilerplate_text": ["MUSTER ANALYTICS", "Copyright 2026 Muster Analytics"],
        "forbidden_text": [],
    }
    metrics = benchmark.score_document(doc, expected, CAPS_TRAFILATURA)
    assert metrics["boilerplate_retention"] == 0.0
    assert metrics["boilerplate_removal"] == 1.0
    assert metrics["quality_score"] is not None


def test_boilerplate_retention_not_penalized_for_full_dom_scope():
    doc = _doc([
        _block(BlockType.TEXT, "MUSTER ANALYTICS", {"dom_order": 0}),
        _block(BlockType.TEXT, "A well-designed navigation layout guides readers", {"dom_order": 1}),
        _block(BlockType.TEXT, "Copyright 2026 Muster Analytics", {"dom_order": 2}),
    ])
    expected = {
        "expected_text": ["navigation layout"],
        "expected_headings": [],
        "expected_tables": 0,
        "expected_list_items": [],
        "expected_link_hrefs": [],
        "expected_images": [],
        "reading_order": [],
        "expected_metadata": [],
        "boilerplate_text": ["MUSTER ANALYTICS", "Copyright 2026 Muster Analytics"],
        "forbidden_text": [],
    }
    metrics = benchmark.score_document(doc, expected, CAPS_BSOUP)
    # full_dom scope: retention is reported, removal is N/A, quality excludes it
    assert metrics["boilerplate_retention"] == 1.0
    assert metrics["boilerplate_removal"] is None


def test_boilerplate_removal_counts_when_boilerplate_remains():
    doc = _doc([
        _block(BlockType.TEXT, "MUSTER ANALYTICS", {"dom_order": 0}),
        _block(BlockType.TEXT, "A well-designed navigation layout guides readers", {"dom_order": 1}),
    ])
    expected = {
        "expected_text": ["navigation layout"],
        "expected_headings": [],
        "expected_tables": 0,
        "expected_list_items": [],
        "expected_link_hrefs": [],
        "expected_images": [],
        "reading_order": [],
        "expected_metadata": [],
        "boilerplate_text": ["MUSTER ANALYTICS", "Copyright 2026 Muster Analytics"],
        "forbidden_text": [],
    }
    metrics = benchmark.score_document(doc, expected, CAPS_TRAFILATURA)
    assert metrics["boilerplate_retention"] == 0.5
    assert metrics["boilerplate_removal"] == 0.5


def test_forbidden_text_isolation():
    doc = _doc([_block(BlockType.TEXT, "visible prose only", {"dom_order": 0})])
    expected = {
        "expected_text": ["visible prose"],
        "expected_headings": [],
        "expected_tables": 0,
        "expected_list_items": [],
        "expected_link_hrefs": [],
        "expected_images": [],
        "reading_order": [],
        "expected_metadata": [],
        "boilerplate_text": [],
        "forbidden_text": ["SECRET_TOKEN", "window.track"],
    }
    metrics = benchmark.score_document(doc, expected, CAPS_BSOUP)
    assert metrics["forbidden_text_isolation"] == 1.0
    leaky = _doc([_block(BlockType.TEXT, "visible prose with SECRET_TOKEN inside", {"dom_order": 0})])
    metrics = benchmark.score_document(leaky, expected, CAPS_BSOUP)
    assert metrics["forbidden_text_isolation"] == 0.5


def test_failed_metrics_zero_for_applicable_na_for_inapplicable():
    metrics = benchmark.failed_metrics(TEXT_EXPECTATION, CAPS_TRAFILATURA)
    assert metrics["status"] == "failed"
    assert metrics["text_coverage"] == 0.0
    assert metrics["reading_order"] == 0.0
    assert metrics["image_detection"] is None      # capability missing -> N/A
    assert metrics["quality_score"] == 0.0


def test_quality_score_excludes_na_components():
    metrics = benchmark.score_document(_simple_doc(), TEXT_EXPECTATION, CAPS_TRAFILATURA)
    # text=1.0, structure=1.0, reading_order=1.0, metadata=1.0 (title) -> 1.0
    assert metrics["quality_score"] == 1.0


def test_reading_order_scoring():
    reversed_doc = _doc([
        _block(BlockType.HEADING, "Installing Widgets", {"heading_level": 2, "dom_order": 2}),
        _block(BlockType.TEXT, "Widgets are the fundamental building blocks of the example system.", {"dom_order": 1}),
        _block(BlockType.HEADING, "Getting Started with Widgets", {"heading_level": 1, "dom_order": 0}),
    ])
    metrics = benchmark.score_document(reversed_doc, TEXT_EXPECTATION, CAPS_BSOUP)
    assert metrics["reading_order"] == 0.0


def test_archive_existing_reports(tmp_path):
    (tmp_path / "benchmark_report.json").write_text("{}")
    (tmp_path / "benchmark_results.csv").write_text("a,b\n")
    archived = benchmark.archive_existing_reports(tmp_path)
    assert len(archived) == 2
    for dst in archived:
        assert Path(dst).exists()
        assert Path(dst).parent == tmp_path / "archive"
    assert (tmp_path / "benchmark_report.json").exists()


def test_archive_no_existing_reports(tmp_path):
    assert benchmark.archive_existing_reports(tmp_path) == []
    assert not (tmp_path / "archive").exists()


def test_expectations_schema_keys():
    required = {
        "expected_text",
        "expected_headings",
        "expected_heading_levels",
        "expected_tables",
        "expected_list_items",
        "expected_link_hrefs",
        "expected_images",
        "reading_order",
        "expected_metadata",
        "boilerplate_text",
        "forbidden_text",
    }
    for filename, expected in EXPECTATIONS.items():
        assert required <= set(expected), filename
