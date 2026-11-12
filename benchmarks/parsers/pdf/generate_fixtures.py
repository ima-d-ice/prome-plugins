"""Generate synthetic PDF fixtures for parser tests and benchmarks.

Fixture bytes are deterministic (fixed metadata creation/mod dates and
`reproducible=True, no_new_id=True`), so SHA-256 hashes are stable across
regeneration runs. `encrypted.pdf` is the only exception: PDF encryption uses
a random file key on every save.

Usage: python benchmarks/parsers/pdf/generate_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf

FIXTURES_DIR = Path(__file__).parent / "fixtures"
EXPECTATIONS_PATH = Path(__file__).parent / "expectations.json"


def _new_doc() -> pymupdf.Document:
    return pymupdf.open()


def _insert_heading(page, x: float, y: float, text: str, size: float) -> None:
    page.insert_text((x, y), text, fontsize=size, fontname="hebo")


def _insert_body(page, rect, text: str, size: float = 11) -> None:
    page.insert_textbox(rect, text, fontsize=size, fontname="helv")


def simple_digital(path: Path) -> None:
    doc = _new_doc()
    page = doc.new_page(width=612, height=792)
    _insert_heading(page, 72, 96, "Quarterly Report Overview", 20)
    _insert_body(
        page,
        pymupdf.Rect(72, 130, 540, 400),
        "This report summarizes quarterly performance across all business units. "
        "Revenue showed steady growth with moderate operating variance. "
        "All figures in this document are synthetic and provided for testing purposes only.",
    )
    doc.set_metadata({"title": "Quarterly Report Overview", "author": "Prome Test Suite", "creationDate": "D:20200101000000Z", "modDate": "D:20200101000000Z"})
    doc.save(path, reproducible=True, no_new_id=True)
    doc.close()


def multipage_structured(path: Path) -> None:
    doc = _new_doc()
    sections = [
        ("Introduction", "The introduction outlines scope, goals and audience for this document. " * 3),
        ("Methods", "The methods section describes data collection and evaluation procedures. " * 3),
        ("Results", "The results section presents findings from the evaluation campaign. " * 3),
    ]
    for heading, body in sections:
        page = doc.new_page(width=612, height=792)
        _insert_heading(page, 72, 96, heading, 16)
        _insert_body(page, pymupdf.Rect(72, 130, 540, 700), body)
    doc.set_metadata({"title": "Structured Multi-Page Document", "author": "Prome Test Suite", "creationDate": "D:20200101000000Z", "modDate": "D:20200101000000Z"})
    doc.save(path, reproducible=True, no_new_id=True)
    doc.close()


def tables(path: Path) -> None:
    doc = _new_doc()
    page = doc.new_page(width=612, height=792)
    _insert_heading(page, 72, 72, "Invoice Summary", 16)

    rows = [
        ["Item", "Qty", "Price"],
        ["Widget", "3", "19.99"],
        ["Gadget", "7", "5.50"],
        ["Gizmo", "2", "42.00"],
    ]
    x0, y0 = 72, 140
    col_widths = [220, 100, 140]
    row_height = 28
    table_width = sum(col_widths)
    n_rows = len(rows)

    shape = page.new_shape()
    for i in range(n_rows + 1):
        y = y0 + i * row_height
        shape.draw_line(pymupdf.Point(x0, y), pymupdf.Point(x0 + table_width, y))
    x = x0
    shape.draw_line(pymupdf.Point(x0, y0), pymupdf.Point(x0, y0 + n_rows * row_height))
    for width in col_widths:
        x += width
        shape.draw_line(pymupdf.Point(x, y0), pymupdf.Point(x, y0 + n_rows * row_height))
    shape.finish(width=1)
    shape.commit()

    for r, row in enumerate(rows):
        x = x0
        for c, cell in enumerate(row):
            page.insert_text((x + 6, y0 + r * row_height + 19), cell, fontsize=11, fontname="helv")
            x += col_widths[c]

    _insert_body(
        page,
        pymupdf.Rect(72, y0 + n_rows * row_height + 30, 540, y0 + n_rows * row_height + 90),
        "Totals include all applicable taxes and fees.",
    )
    doc.set_metadata({"title": "Invoice Summary", "author": "Prome Test Suite", "creationDate": "D:20200101000000Z", "modDate": "D:20200101000000Z"})
    doc.save(path, reproducible=True, no_new_id=True)
    doc.close()


def blank_page(path: Path) -> None:
    doc = _new_doc()
    page = doc.new_page(width=612, height=792)
    _insert_heading(page, 72, 96, "Content Before Blank Page", 16)
    _insert_body(
        page,
        pymupdf.Rect(72, 130, 540, 400),
        "This page contains text. The following page intentionally left blank for layout testing.",
    )
    doc.new_page(width=612, height=792)  # blank page 2
    doc.set_metadata({"title": "Document With Blank Page", "author": "Prome Test Suite", "creationDate": "D:20200101000000Z", "modDate": "D:20200101000000Z"})
    doc.save(path, reproducible=True, no_new_id=True)
    doc.close()


def multi_section(path: Path) -> None:
    doc = _new_doc()
    pages_content = [
        ("Section One", "Alpha overview. MARKER-01 begins the first section. " * 4),
        ("Section Two", "Beta analysis. MARKER-03 continues the narrative. " * 4),
        ("Section Three", "Gamma conclusion. MARKER-05 closes the document. " * 4),
    ]
    for heading, body in pages_content:
        page = doc.new_page(width=612, height=792)
        _insert_heading(page, 72, 96, heading, 16)
        _insert_body(page, pymupdf.Rect(72, 130, 540, 700), body)
    doc.set_metadata({"title": "Multi Section Document", "author": "Prome Test Suite", "creationDate": "D:20200101000000Z", "modDate": "D:20200101000000Z"})
    doc.save(path, reproducible=True, no_new_id=True)
    doc.close()


def two_column(path: Path) -> None:
    doc = _new_doc()
    page = doc.new_page(width=612, height=792)
    left = pymupdf.Rect(54, 60, 288, 720)
    right = pymupdf.Rect(324, 60, 558, 720)
    # textboxes keep the two column headings in separate blocks instead of
    # one merged same-line block
    page.insert_textbox(pymupdf.Rect(left.x0, 60, left.x1, 92), "Column Alpha", fontsize=16, fontname="hebo")
    page.insert_textbox(pymupdf.Rect(right.x0, 60, right.x1, 92), "Column Beta", fontsize=16, fontname="hebo")
    _insert_body(
        page,
        pymupdf.Rect(left.x0, 110, left.x1, 300),
        "ALPHA-MARKER-01 opens the left column with introductory material. "
        "ALPHA-MARKER-02 continues the left column with supporting detail. "
        "ALPHA-MARKER-03 closes the left column before the gutter. " * 2,
    )
    _insert_body(
        page,
        pymupdf.Rect(left.x0, 320, left.x1, 520),
        "ALPHA-MARKER-04 second paragraph of the left column keeps going. "
        "ALPHA-MARKER-05 wraps up the left column narrative. " * 2,
    )
    _insert_body(
        page,
        pymupdf.Rect(right.x0, 110, right.x1, 300),
        "BETA-MARKER-01 opens the right column with its own opening statement. "
        "BETA-MARKER-02 continues the right column with supporting detail. "
        "BETA-MARKER-03 closes the right column opening. " * 2,
    )
    _insert_body(
        page,
        pymupdf.Rect(right.x0, 320, right.x1, 520),
        "BETA-MARKER-04 second paragraph of the right column keeps going. "
        "BETA-MARKER-05 wraps up the right column narrative. " * 2,
    )
    doc.set_metadata({"title": "Two Column Layout", "author": "Prome Test Suite", "creationDate": "D:20200101000000Z", "modDate": "D:20200101000000Z"})
    doc.save(path, reproducible=True, no_new_id=True)
    doc.close()


def headers_footers(path: Path) -> None:
    doc = _new_doc()
    sections = [
        ("Report Section Alpha", "REPORT-ALPHA-MARKER body text for the first report section. "),
        ("Report Section Beta", "REPORT-BETA-MARKER body text for the second report section. "),
        ("Report Section Gamma", "REPORT-GAMMA-MARKER body text for the third report section. "),
    ]
    for index, (heading, body) in enumerate(sections, start=1):
        page = doc.new_page(width=612, height=792)
        shape = page.new_shape()
        shape.draw_line(pymupdf.Point(54, 48), pymupdf.Point(558, 48))
        shape.draw_line(pymupdf.Point(54, 744), pymupdf.Point(558, 744))
        shape.finish(width=0.5)
        shape.commit()
        page.insert_text(
            (54, 38), "PROME SYNTHETIC REPORT", fontsize=9, fontname="hebo", color=(0.3, 0.3, 0.3)
        )
        page.insert_text(
            (558 - 130, 38), f"INTERNAL-{index:02d}", fontsize=9, fontname="helv", color=(0.3, 0.3, 0.3)
        )
        page.insert_text(
            (54, 762),
            f"Synthetic benchmark fixture - page {index}",
            fontsize=9,
            fontname="helv",
            color=(0.3, 0.3, 0.3),
        )
        page.insert_text(
            (558 - 60, 762), f"PAGE-{index:02d}", fontsize=9, fontname="helv", color=(0.3, 0.3, 0.3)
        )
        _insert_heading(page, 54, 108, heading, 16)
        _insert_body(page, pymupdf.Rect(54, 140, 558, 700), body * 12)
    doc.set_metadata({"title": "Headers Footers And Page Numbers", "author": "Prome Test Suite", "creationDate": "D:20200101000000Z", "modDate": "D:20200101000000Z"})
    doc.save(path, reproducible=True, no_new_id=True)
    doc.close()


def rotated(path: Path) -> None:
    doc = _new_doc()
    page = doc.new_page(width=612, height=792)
    _insert_heading(page, 72, 96, "Rotated Content Showcase", 16)
    _insert_body(
        page,
        pymupdf.Rect(72, 130, 540, 300),
        "NORMAL-MARKER-1 introduces the page before any rotated material. "
        "NORMAL-MARKER-2 keeps the horizontal flow easy to follow. " * 2,
    )
    page.insert_text(
        (580, 500),
        "ROTATED-MARKER-90 vertical label reading bottom to top",
        fontsize=10,
        rotate=90,
        fontname="helv",
    )
    rotated_page = doc.new_page(width=612, height=792)
    _insert_heading(rotated_page, 72, 96, "Rotated Page Setup", 16)
    _insert_body(
        rotated_page,
        pymupdf.Rect(72, 130, 540, 300),
        "ROTATED-PAGE-MARKER-2 lives on a page whose /Rotate attribute is set to 90 degrees. "
        "ROTATED-PAGE-MARKER-3 continues on the same rotated page. " * 2,
    )
    rotated_page.set_rotation(90)
    doc.set_metadata({"title": "Rotated Text And Pages", "author": "Prome Test Suite", "creationDate": "D:20200101000000Z", "modDate": "D:20200101000000Z"})
    doc.save(path, reproducible=True, no_new_id=True)
    doc.close()


def multipage_table(path: Path) -> None:
    doc = _new_doc()
    headers = ["SKU", "Name", "Qty", "Price"]
    col_widths = [80, 270, 60, 100]
    table_width = sum(col_widths)
    x0, header_y, row_height = 54, 120, 18
    rows_per_page = 30
    total_rows = 60

    for page_index in range(total_rows // rows_per_page):
        page = doc.new_page(width=612, height=792)
        if page_index == 0:
            _insert_heading(page, 54, 84, "Component Catalog", 16)
            note = "Catalog of synthetic components. Table continues on the next page."
        else:
            _insert_heading(page, 54, 84, "Component Catalog Continued", 16)
            note = "Component Catalog continued. Header row repeated for readability."
        _insert_body(page, pymupdf.Rect(54, 96, 558, 112), note, size=8)

        first, last = page_index * rows_per_page + 1, (page_index + 1) * rows_per_page
        page_rows = [headers] + [
            [f"SKU-{n:03d}", f"Component {n:03d}", str((n % 9) + 1), f"{(n * 7 % 80) + 4}.99"]
            for n in range(first, last + 1)
        ]
        table_height = len(page_rows) * row_height

        shape = page.new_shape()
        for i in range(len(page_rows) + 1):
            y = header_y + i * row_height
            shape.draw_line(pymupdf.Point(x0, y), pymupdf.Point(x0 + table_width, y))
        shape.draw_line(pymupdf.Point(x0, header_y), pymupdf.Point(x0, header_y + table_height))
        x = x0
        for width in col_widths:
            x += width
            shape.draw_line(pymupdf.Point(x, header_y), pymupdf.Point(x, header_y + table_height))
        shape.finish(width=0.75)
        shape.commit()

        for r, row in enumerate(page_rows):
            x = x0
            font = "hebo" if r == 0 else "helv"
            for c, cell in enumerate(row):
                page.insert_text(
                    (x + 5, header_y + r * row_height + 13), cell, fontsize=8, fontname=font
                )
                x += col_widths[c]

    doc.set_metadata({"title": "Component Catalog", "author": "Prome Test Suite", "creationDate": "D:20200101000000Z", "modDate": "D:20200101000000Z"})
    doc.save(path, reproducible=True, no_new_id=True)
    doc.close()


def images_captions(path: Path) -> None:
    doc = _new_doc()
    page = doc.new_page(width=612, height=792)
    _insert_heading(page, 54, 84, "Visual Evidence", 16)

    boxes = [("NODE-ALPHA", 54), ("NODE-BETA", 234), ("NODE-GAMMA", 414)]
    shape = page.new_shape()
    for label, x in boxes:
        shape.draw_rect(pymupdf.Rect(x, 140, x + 128, 220))
    shape.draw_line(pymupdf.Point(182, 180), pymupdf.Point(234, 180))
    shape.draw_line(pymupdf.Point(362, 180), pymupdf.Point(414, 180))
    shape.finish(width=1.25)
    shape.commit()
    for label, x in boxes:
        page.insert_text((x + 12, 184), label, fontsize=10, fontname="hebo")
    _insert_body(
        page,
        pymupdf.Rect(54, 236, 558, 292),
        "Diagram walk-through: NODE-ALPHA feeds NODE-BETA, which emits NODE-GAMMA. " * 2,
    )
    page.insert_text(
        (54, 316), "Figure 1: synthetic node diagram generated for caption testing.", fontsize=9, fontname="heit"
    )

    scratch = _new_doc()
    scratch_page = scratch.new_page(width=240, height=140)
    scratch_shape = scratch_page.new_shape()
    scratch_shape.draw_rect(pymupdf.Rect(0, 0, 240, 140))
    scratch_shape.draw_line(pymupdf.Point(0, 70), pymupdf.Point(240, 70))
    scratch_shape.draw_line(pymupdf.Point(120, 0), pymupdf.Point(120, 140))
    scratch_shape.finish(width=6, color=(0.15, 0.2, 0.55))
    scratch_shape.commit()
    pix = scratch_page.get_pixmap(dpi=110)
    scratch.close()
    page.insert_image(pymupdf.Rect(54, 340, 294, 480), pixmap=pix)
    page.insert_text(
        (54, 502), "Figure 2: synthetic rendered bitmap inserted as an image block.", fontsize=9, fontname="heit"
    )
    _insert_body(
        page,
        pymupdf.Rect(54, 528, 558, 640),
        "CAPTION-MARKER-1 closes the visual evidence page. CAPTION-MARKER-2 keeps the order stable. " * 2,
    )
    doc.set_metadata({"title": "Visual Evidence", "author": "Prome Test Suite", "creationDate": "D:20200101000000Z", "modDate": "D:20200101000000Z"})
    doc.save(path, reproducible=True, no_new_id=True)
    doc.close()


def scanned(path: Path) -> None:
    scratch = _new_doc()
    marker_pages = [
        ("SCANNED-MARKER-01 opens the scanned page one narrative.", "SCANNED-MARKER-02 keeps page one going."),
        ("SCANNED-MARKER-03 opens the scanned page two narrative.", "SCANNED-MARKER-04 keeps page two going."),
    ]
    pixmaps = []
    for first, second in marker_pages:
        scratch_page = scratch.new_page(width=612, height=792)
        _insert_heading(scratch_page, 72, 96, "Scanned Document", 18)
        _insert_body(scratch_page, pymupdf.Rect(72, 140, 540, 400), f"{first} {second} " * 4)
        pixmaps.append(scratch_page.get_pixmap(dpi=110))
    scratch.close()

    doc = _new_doc()
    for pix in pixmaps:
        page = doc.new_page(width=612, height=792)
        page.insert_image(page.rect, pixmap=pix)
    doc.set_metadata({"title": "Scanned Image Only", "author": "Prome Test Suite", "creationDate": "D:20200101000000Z", "modDate": "D:20200101000000Z"})
    doc.save(path, reproducible=True, no_new_id=True)
    doc.close()


def mixed_sizes(path: Path) -> None:
    doc = _new_doc()
    page = doc.new_page(width=612, height=792)
    page.insert_text((54, 96), "Mixed Sizes Overview", fontsize=22, fontname="hebo")
    page.insert_text((54, 150), "Section Alpha", fontsize=15, fontname="hebo")
    _insert_body(
        page,
        pymupdf.Rect(54, 165, 558, 260),
        "MIXED-ALPHA-MARKER opens the alpha section at body size. Advance sizing cues down the page. " * 3,
        size=10,
    )
    bullets = [
        "ITEM-ALPHA-1 first bullet of the alpha section",
        "ITEM-ALPHA-2 second bullet of the alpha section",
        "ITEM-ALPHA-3 third bullet of the alpha section",
    ]
    y = 275
    for bullet in bullets:
        page.insert_text((72, y), "•", fontsize=10, fontname="helv")
        page.insert_text((92, y), bullet, fontsize=10, fontname="helv")
        y += 18
    page.insert_text((54, 380), "Section Beta", fontsize=15, fontname="hebo")
    _insert_body(
        page,
        pymupdf.Rect(54, 395, 558, 500),
        "MIXED-BETA-MARKER opens the beta section at body size. Finish the sizing tour. " * 3,
        size=10,
    )
    page.insert_text((54, 560), "Fine print: MIXED-FINE-MARKER drops to the smallest size on the page.", fontsize=7, fontname="heit")
    doc.set_metadata({"title": "Mixed Sizes Headings And Lists", "author": "Prome Test Suite", "creationDate": "D:20200101000000Z", "modDate": "D:20200101000000Z"})
    doc.save(path, reproducible=True, no_new_id=True)
    doc.close()


def encrypted(path: Path) -> None:
    doc = _new_doc()
    page = doc.new_page(width=612, height=792)
    _insert_body(page, pymupdf.Rect(72, 96, 540, 400), "This document is encrypted for testing.")
    doc.save(
        path,
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw="prome-test-password",
        owner_pw="prome-test-password",
    )
    doc.close()


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    simple_digital(FIXTURES_DIR / "simple_digital.pdf")
    multipage_structured(FIXTURES_DIR / "multipage_structured.pdf")
    tables(FIXTURES_DIR / "tables.pdf")
    blank_page(FIXTURES_DIR / "blank_page.pdf")
    multi_section(FIXTURES_DIR / "multi_section.pdf")
    two_column(FIXTURES_DIR / "two_column.pdf")
    headers_footers(FIXTURES_DIR / "headers_footers.pdf")
    rotated(FIXTURES_DIR / "rotated.pdf")
    multipage_table(FIXTURES_DIR / "multipage_table.pdf")
    images_captions(FIXTURES_DIR / "images_captions.pdf")
    scanned(FIXTURES_DIR / "scanned.pdf")
    mixed_sizes(FIXTURES_DIR / "mixed_sizes.pdf")
    encrypted(FIXTURES_DIR / "encrypted.pdf")

    expectations = {
        "simple_digital.pdf": {
            "expected_pages": 1,
            "expected_text": ["Quarterly Report Overview", "steady growth"],
            "expected_headings": ["Quarterly Report Overview"],
            "expected_tables": 0,
            "blank_pages": [],
            "reading_order": ["Quarterly Report Overview", "steady growth"],
        },
        "multipage_structured.pdf": {
            "expected_pages": 3,
            "expected_text": ["Introduction", "Methods", "Results", "data collection"],
            "expected_headings": ["Introduction", "Methods", "Results"],
            "expected_tables": 0,
            "blank_pages": [],
            "reading_order": ["Introduction", "Methods", "Results"],
        },
        "tables.pdf": {
            "expected_pages": 1,
            "expected_text": ["Invoice Summary", "Widget", "Gadget", "Gizmo"],
            "expected_headings": ["Invoice Summary"],
            "expected_tables": 1,
            "expected_table_header": ["Item", "Qty", "Price"],
            "expected_table_cells": ["Widget", "Gadget", "Gizmo", "19.99", "5.50", "42.00"],
            "expected_table_pages": {"1": ["Widget", "Gadget", "Gizmo"]},
            "blank_pages": [],
            "reading_order": ["Invoice Summary", "Widget", "Totals include"],
        },
        "blank_page.pdf": {
            "expected_pages": 2,
            "expected_text": ["Content Before Blank Page"],
            "expected_headings": ["Content Before Blank Page"],
            "expected_tables": 0,
            "blank_pages": [2],
            "reading_order": ["Content Before Blank Page"],
        },
        "multi_section.pdf": {
            "expected_pages": 3,
            "expected_text": ["Section One", "Section Two", "Section Three", "MARKER-01", "MARKER-03", "MARKER-05"],
            "expected_headings": ["Section One", "Section Two", "Section Three"],
            "expected_tables": 0,
            "blank_pages": [],
            "reading_order": ["MARKER-01", "MARKER-03", "MARKER-05"],
        },
        "two_column.pdf": {
            "expected_pages": 1,
            "expected_text": [
                "Column Alpha", "Column Beta",
                "ALPHA-MARKER-01", "ALPHA-MARKER-02", "ALPHA-MARKER-03",
                "ALPHA-MARKER-04", "ALPHA-MARKER-05",
                "BETA-MARKER-01", "BETA-MARKER-02", "BETA-MARKER-03",
                "BETA-MARKER-04", "BETA-MARKER-05",
            ],
            "expected_headings": ["Column Alpha", "Column Beta"],
            "expected_tables": 0,
            "blank_pages": [],
            "reading_order": [
                "ALPHA-MARKER-01", "ALPHA-MARKER-02", "ALPHA-MARKER-03",
                "ALPHA-MARKER-04", "ALPHA-MARKER-05",
                "BETA-MARKER-01", "BETA-MARKER-02", "BETA-MARKER-03",
                "BETA-MARKER-04", "BETA-MARKER-05",
            ],
        },
        "headers_footers.pdf": {
            "expected_pages": 3,
            "expected_text": [
                "PROME SYNTHETIC REPORT",
                "INTERNAL-01", "INTERNAL-02", "INTERNAL-03",
                "Synthetic benchmark fixture - page 1",
                "Synthetic benchmark fixture - page 2",
                "Synthetic benchmark fixture - page 3",
                "PAGE-01", "PAGE-02", "PAGE-03",
                "Report Section Alpha", "Report Section Beta", "Report Section Gamma",
                "REPORT-ALPHA-MARKER", "REPORT-BETA-MARKER", "REPORT-GAMMA-MARKER",
            ],
            "expected_headings": ["Report Section Alpha", "Report Section Beta", "Report Section Gamma"],
            "expected_tables": 0,
            "blank_pages": [],
            "reading_order": ["Report Section Alpha", "Report Section Beta", "Report Section Gamma"],
            # Documented, measured limitation (see benchmark report): docling parses
            # running headers/footers as page boilerplate and excludes them from the
            # body flow. Gaps only inform the test suite; the benchmark still counts
            # these snippets as missed.
            "known_extraction_gaps": {
                "docling": [
                    "PROME SYNTHETIC REPORT",
                    "INTERNAL-01", "INTERNAL-02", "INTERNAL-03",
                    "Synthetic benchmark fixture - page 1",
                    "Synthetic benchmark fixture - page 2",
                    "Synthetic benchmark fixture - page 3",
                    "PAGE-01", "PAGE-02", "PAGE-03",
                ],
            },
        },
        "rotated.pdf": {
            "expected_pages": 2,
            "expected_text": [
                "Rotated Content Showcase", "NORMAL-MARKER-1", "NORMAL-MARKER-2",
                "ROTATED-MARKER-90",
                "Rotated Page Setup", "ROTATED-PAGE-MARKER-2", "ROTATED-PAGE-MARKER-3",
            ],
            "expected_headings": ["Rotated Content Showcase", "Rotated Page Setup"],
            "expected_tables": 0,
            "blank_pages": [],
            "reading_order": [
                "Rotated Content Showcase", "NORMAL-MARKER-1",
                "Rotated Page Setup", "ROTATED-PAGE-MARKER-2",
            ],
            # Documented, measured limitation (see benchmark report): with OCR
            # disabled, docling drops the vertical rotated label and misinterprets
            # the rotated page's text as a table, breaking marker order there.
            "known_extraction_gaps": {
                "docling": ["ROTATED-MARKER-90", "ROTATED-PAGE-MARKER-2"],
            },
        },
        "multipage_table.pdf": {
            "expected_pages": 2,
            "expected_text": [
                "Component Catalog", "Component Catalog Continued",
                "SKU-001", "SKU-030", "SKU-031", "SKU-060",
            ],
            "expected_headings": ["Component Catalog", "Component Catalog Continued"],
            "expected_tables": 2,
            "expected_table_header": ["SKU", "Name", "Qty", "Price"],
            "expected_table_cells": [
                "SKU-001", "Component 001", "SKU-030", "Component 030",
                "SKU-031", "Component 031", "SKU-060", "Component 060",
            ],
            "expected_table_pages": {
                "1": ["SKU-001", "Component 015", "SKU-030", "Component 030"],
                "2": ["SKU-031", "Component 045", "SKU-060", "Component 060"],
            },
            "blank_pages": [],
            "reading_order": ["Component Catalog", "SKU-001", "Component Catalog Continued", "SKU-060"],
        },
        "images_captions.pdf": {
            "expected_pages": 1,
            "expected_text": [
                "Visual Evidence", "NODE-ALPHA", "NODE-BETA", "NODE-GAMMA",
                "Figure 1", "Figure 2", "CAPTION-MARKER-1", "CAPTION-MARKER-2",
            ],
            "expected_headings": ["Visual Evidence"],
            "expected_tables": 0,
            "expected_images": 1,
            "blank_pages": [],
            "reading_order": ["Visual Evidence", "Figure 1", "Figure 2", "CAPTION-MARKER-1"],
        },
        "scanned.pdf": {
            "expected_pages": 2,
            "expected_text": [],
            "expected_headings": [],
            "expected_tables": 0,
            "blank_pages": [],
            "reading_order": [],
            "requires_ocr": True,
        },
        "mixed_sizes.pdf": {
            "expected_pages": 1,
            "expected_text": [
                "Mixed Sizes Overview", "Section Alpha", "MIXED-ALPHA-MARKER",
                "ITEM-ALPHA-1", "ITEM-ALPHA-2", "ITEM-ALPHA-3",
                "Section Beta", "MIXED-BETA-MARKER", "MIXED-FINE-MARKER",
            ],
            "expected_headings": ["Mixed Sizes Overview", "Section Alpha", "Section Beta"],
            "expected_tables": 0,
            "expected_list_items": ["ITEM-ALPHA-1", "ITEM-ALPHA-2", "ITEM-ALPHA-3"],
            "blank_pages": [],
            "reading_order": [
                "Mixed Sizes Overview", "Section Alpha", "ITEM-ALPHA-1",
                "Section Beta", "MIXED-FINE-MARKER",
            ],
        },
    }
    EXPECTATIONS_PATH.write_text(json.dumps(expectations, indent=2) + "\n")
    print(f"Fixtures written to {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
