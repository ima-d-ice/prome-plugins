"""Reproducible PDF parser benchmark for Prome RAG Plugins.

Parses the synthetic fixture corpus (see generate_fixtures.py) with each
available parser and reports extraction quality and speed separately:

  Quality:  page coverage, text coverage, structure preservation
            (headings + table cells), reading order, image/list detection
  Speed:    cold-start time, first-document time, steady-state time per
            fixture, pages-per-second throughput

Capability model
----------------
Not every parser supports every capability, so unsupported capabilities are
reported as N/A (null in JSON, empty in CSV) and are excluded from every
average. They are never treated as ordinary extraction failures and never
scored as success. A fixture may also require a capability globally disabled
(e.g. OCR for the scanned image-only fixture); those rows are status =
"unsupported" and are excluded from scoring, not failed.

Statuses per (parser, fixture) row:
  ok          parsed and scored normally
  failed      parsing raised an exception (ordinary extraction failure,
              scored as 0 for the achieved metrics)
  unsupported a required capability is missing; metrics are N/A

Quality score: mean of the applicable, non-None quality components
(text coverage, structure preservation, reading order, image detection,
list detection). Components that do not apply to the fixture/parser pair
are excluded, not scored as 1 or 0.

Performance score: steady-state pages/second relative to the fastest
non-skipped parser (1.0 = fastest). Documented; not an absolute grade.

Cold vs steady state
--------------------
Each parser gets a fresh instance. The first pass over the fixture corpus is
the cold pass (includes any lazy initialization such as Docling's converter
setup); subsequent repeats are the steady-state pass. The report separates
both. Parsers whose one-time cost lives in construction measure it in
cold_start_seconds via construction + first fixture.

Existing reports are never silently destroyed: before overwriting, previous
report files are copied into reports/archive/ with a UTC timestamp.

Unavailable parsers are recorded as "skipped" -- never as fake numbers.

Usage:
  python benchmarks/parsers/pdf/benchmark.py
  python benchmarks/parsers/pdf/benchmark.py --parsers pymupdf pypdf
  python benchmarks/parsers/pdf/benchmark.py --repeats 3 --output-dir reports
  python benchmarks/parsers/pdf/benchmark.py --no-archive
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

FIXTURES_DIR = Path(__file__).parent / "fixtures"
EXPECTATIONS_PATH = Path(__file__).parent / "expectations.json"
DEFAULT_REPORTS_DIR = Path(__file__).parent / "reports"

REPORT_VERSION = 2

PARSER_BUILDERS = {
    "pymupdf": ("parsers.pdf.pymupdf", "PyMuPDFParser", ("pymupdf", "fitz")),
    "pypdf": ("parsers.pdf.pypdf", "PyPDFParser", ("pypdf",)),
    "docling": ("parsers.pdf.docling", "DoclingParser", ("docling",)),
}

# Capabilities each parser implementation actually provides. These are
# reviewed against the implementations in parsers/pdf/*/; a capability set to
# False means the parser has no implementation of it, not that it did badly.
# OCR is disabled in all three parsers (Docling runs with do_ocr=False).
PARSER_CAPABILITIES = {
    "pymupdf": {
        "text_extraction": True,
        "structure": True,      # font-size heading heuristic + table detection
        "tables": True,         # page.find_tables()
        "images": True,         # image blocks via get_text("dict")
        "lists": False,         # no list detection
        "ocr": False,
    },
    "pypdf": {
        "text_extraction": True,
        "structure": False,     # plain text only; no heading/table detection
        "tables": False,        # no table detection
        "images": False,        # images not surfaced as blocks
        "lists": False,
        "ocr": False,
    },
    "docling": {
        "text_extraction": True,
        "structure": True,      # layout model: title/section_header/table/list labels
        "tables": True,         # TableFormer; exported as markdown
        "images": True,         # picture/chart items
        "lists": True,          # list_item label
        "ocr": False,           # do_ocr=False; scanned fixtures are out of scope
    },
}

QUALITY_COMPONENT_KEYS = (
    "text_coverage",
    "structure_preservation",
    "reading_order",
    "image_detection",
    "list_detection",
)


def load_expectations() -> dict:
    if not EXPECTATIONS_PATH.exists():
        sys.exit(f"expectations.json not found at {EXPECTATIONS_PATH}; run generate_fixtures.py first")
    return json.loads(EXPECTATIONS_PATH.read_text())


def available_parsers(names: list[str]) -> dict[str, tuple[object, str | None]]:
    """Return {name: (parser_instance, None)} for available, else (None, reason)."""
    result = {}
    import importlib

    for name in names:
        module_name, class_name, lib_modules = PARSER_BUILDERS[name]
        missing = None
        for lib in lib_modules:
            try:
                importlib.import_module(lib)
                missing = None
                break
            except Exception as exc:
                missing = exc
        else:
            result[name] = (None, f"missing optional dependency for {name}: {missing}".replace("\n", " "))
            continue
        try:
            module = importlib.import_module(module_name)
            start = time.perf_counter()
            parser = getattr(module, class_name)()
            parser.construction_seconds = time.perf_counter() - start
            result[name] = (parser, None)
        except Exception as exc:  # ParserDependencyError or import failure
            result[name] = (None, f"{type(exc).__name__}: {exc}".replace("\n", " "))
    return result


def library_versions() -> dict[str, str]:
    import importlib.metadata

    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for dist in ("pymupdf", "pypdf", "docling", "docling-core"):
        try:
            versions[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            versions[dist] = "not installed"
    return versions


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fraction(hits: int, total: int) -> float | None:
    return round(hits / total, 4) if total else None


def _content_hits(text_lower: str, snippets: list[str]) -> int:
    return sum(1 for snippet in snippets if snippet.lower() in text_lower)


def _blocks_of_type(doc, block_type: str) -> list:
    return [b for page in doc.pages for b in page.blocks if b.block_type.value == block_type]


def unsupported_metrics(expected: dict, capability: str) -> dict:
    return {
        "status": "unsupported",
        "unsupported_capability": capability,
        "page_coverage": None,
        "text_coverage": None,
        "structure_preservation": None,
        "reading_order": None,
        "heading_preservation": None,
        "heading_hits": None,
        "table_blocks": None,
        "expected_tables": expected.get("expected_tables"),
        "table_cell_coverage": None,
        "table_header_coverage": None,
        "multipage_table_continuity": None,
        "table_text_duplication": None,
        "image_detection": None,
        "list_detection": None,
        "quality_score": None,
    }


def _component_applicable(expected: dict, capabilities: dict[str, bool]) -> dict[str, bool]:
    """Which quality components apply to this fixture/parser pair."""
    return {
        "text_coverage": bool(expected.get("expected_text")),
        "structure_preservation": capabilities.get("structure", False)
        and bool(
            expected.get("expected_headings")
            or (capabilities.get("tables") and expected.get("expected_table_cells"))
        ),
        "reading_order": bool(expected.get("reading_order")) and capabilities.get("text_extraction", False),
        "image_detection": bool(expected.get("expected_images")) and capabilities.get("images", False),
        "list_detection": bool(expected.get("expected_list_items")) and capabilities.get("lists", False),
    }


def score_document(doc, expected: dict, capabilities: dict[str, bool]) -> dict:
    """Score a parsed document against ground truth.

    Returns metrics where every component is either a float in [0, 1] or None
    (not applicable / capability unsupported -- never scored, never fudged).
    """
    text = doc.text.lower()

    if expected.get("requires_ocr") and not capabilities.get("ocr"):
        return unsupported_metrics(expected, "ocr")

    expected_text = expected.get("expected_text", [])
    text_coverage = round(_content_hits(text, expected_text) / len(expected_text), 4) if expected_text else None

    page_coverage = min(1.0, doc.page_count / expected["expected_pages"]) if expected["expected_pages"] else None

    # --- headings / structure ---
    expected_headings = expected.get("expected_headings", [])
    heading_preservation = None
    heading_hits = None
    if capabilities.get("structure"):
        headings = [b.content.lower() for b in _blocks_of_type(doc, "heading")]
        heading_hits = sum(
            1 for eh in expected_headings if any(eh.lower() in h for h in headings)
        )
        heading_preservation = _fraction(heading_hits, len(expected_headings))

    # --- tables ---
    table_blocks = _blocks_of_type(doc, "table")
    expected_tables = expected.get("expected_tables", 0)
    expected_cells = expected.get("expected_table_cells", [])
    expected_header = expected.get("expected_table_header", [])
    expected_table_pages = expected.get("expected_table_pages", {})
    table_cell_coverage = None
    table_header_coverage = None
    multipage_table_continuity = None
    table_text_duplication = None
    if capabilities.get("tables"):
        table_content = " ".join(b.content.lower() for b in table_blocks)
        if expected_cells:
            table_cell_coverage = round(_content_hits(table_content, expected_cells) / len(expected_cells), 4)
            non_table_text = " ".join(
                b.content.lower() for page in doc.pages for b in page.blocks if b.block_type.value != "table"
            )
            table_text_duplication = round(
                _content_hits(non_table_text, expected_cells) / len(expected_cells), 4
            )
        if expected_header:
            table_header_coverage = round(_content_hits(table_content, expected_header) / len(expected_header), 4)
        if expected_table_pages:
            per_page_total = sum(len(v) for v in expected_table_pages.values())
            per_page_hits = 0
            page_table_text = {
                str(page.page_number): " ".join(
                    b.content.lower() for b in page.blocks if b.block_type.value == "table"
                )
                for page in doc.pages
            }
            for page_no, cells in expected_table_pages.items():
                per_page_hits += _content_hits(page_table_text.get(str(page_no), ""), cells)
            multipage_table_continuity = round(per_page_hits / per_page_total, 4) if per_page_total else None

    # --- structure preservation: mean of applicable structural subscores ---
    structure_subscores = [s for s in (heading_preservation, table_cell_coverage) if s is not None]
    if not capabilities.get("structure"):
        structure_preservation = None
    elif structure_subscores:
        structure_preservation = round(sum(structure_subscores) / len(structure_subscores), 4)
    else:
        structure_preservation = None  # nothing structural expected in this fixture

    # --- reading order ---
    markers = expected.get("reading_order", [])
    reading_order = None
    if markers and capabilities.get("text_extraction"):
        positions = [text.find(marker.lower()) for marker in markers]
        reading_order = float(all(pos >= 0 for pos in positions) and positions == sorted(positions))

    # --- images ---
    expected_images = expected.get("expected_images", 0)
    image_detection = None
    if expected_images:
        if capabilities.get("images"):
            image_blocks = len(_blocks_of_type(doc, "image"))
            image_detection = round(min(image_blocks, expected_images) / expected_images, 4)

    # --- lists ---
    expected_list_items = expected.get("expected_list_items", [])
    list_detection = None
    if expected_list_items:
        if capabilities.get("lists"):
            list_content = " ".join(b.content.lower() for b in _blocks_of_type(doc, "list"))
            list_detection = round(_content_hits(list_content, expected_list_items) / len(expected_list_items), 4)

    components = {
        "text_coverage": text_coverage,
        "structure_preservation": structure_preservation,
        "reading_order": reading_order,
        "image_detection": image_detection,
        "list_detection": list_detection,
    }
    applicable = [v for v in components.values() if v is not None]
    quality_score = round(sum(applicable) / len(applicable), 4) if applicable else None

    return {
        "status": "ok",
        "unsupported_capability": None,
        "page_coverage": round(page_coverage, 4) if page_coverage is not None else None,
        "text_coverage": text_coverage,
        "structure_preservation": structure_preservation,
        "reading_order": reading_order,
        "heading_preservation": heading_preservation,
        "heading_hits": heading_hits,
        "table_blocks": len(table_blocks),
        "expected_tables": expected_tables,
        "table_cell_coverage": table_cell_coverage,
        "table_header_coverage": table_header_coverage,
        "multipage_table_continuity": multipage_table_continuity,
        "table_text_duplication": table_text_duplication,
        "image_detection": image_detection,
        "list_detection": list_detection,
        "quality_score": quality_score,
    }


def failed_metrics(expected: dict, capabilities: dict[str, bool]) -> dict:
    """Metrics for a fixture that raised despite being parseable.

    Zero for every applicable component (the parser achieved nothing),
    None for components that do not apply -- never a fake score.
    """
    applicable = _component_applicable(expected, capabilities)
    tables_applicable = capabilities.get("tables", False) and bool(
        expected.get("expected_table_cells") or expected.get("expected_table_header") or expected.get("expected_table_pages")
    )
    table_cell = 0.0 if tables_applicable and expected.get("expected_table_cells") else None
    table_header = 0.0 if tables_applicable and expected.get("expected_table_header") else None
    continuity = 0.0 if tables_applicable and expected.get("expected_table_pages") else None
    zeros = {key: 0.0 for key, applies in applicable.items() if applies}
    components = list(zeros.values())
    metrics = {
        "status": "failed",
        "unsupported_capability": None,
        "page_coverage": 0.0 if expected.get("expected_pages") else None,
        "text_coverage": zeros.get("text_coverage"),
        "structure_preservation": zeros.get("structure_preservation"),
        "reading_order": zeros.get("reading_order"),
        "heading_preservation": 0.0
        if capabilities.get("structure") and expected.get("expected_headings")
        else None,
        "heading_hits": 0 if capabilities.get("structure") and expected.get("expected_headings") else None,
        "table_blocks": 0,
        "expected_tables": expected.get("expected_tables", 0),
        "table_cell_coverage": table_cell,
        "table_header_coverage": table_header,
        "multipage_table_continuity": continuity,
        "table_text_duplication": None,
        "image_detection": zeros.get("image_detection"),
        "list_detection": zeros.get("list_detection"),
        "quality_score": round(sum(components) / len(components), 4) if components else None,
    }
    return metrics


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def evaluate_parser(name: str, parser, fixtures: dict, repeats: int) -> dict:
    capabilities = PARSER_CAPABILITIES[name]
    rows = []

    # Fresh instance: cold pass (first pass over the corpus, includes any
    # lazy one-time initialization), then the steady-state pass.
    for filename, expected in fixtures.items():
        path = FIXTURES_DIR / filename
        fixture_info = {
            "fixture": filename,
            "bytes": path.stat().st_size,
            "sha256": sha256_of(path),
        }

        cold_start = time.perf_counter()
        try:
            cold_doc = parser.parse(path)
            cold_time = time.perf_counter() - cold_start
            cold_metrics = score_document(cold_doc, expected, capabilities)
            base_error = None
        except Exception as exc:
            cold_doc = None
            cold_time = time.perf_counter() - cold_start
            cold_metrics = failed_metrics(expected, capabilities)
            base_error = f"{type(exc).__name__}: {exc}".replace("\n", " ")

        times = []
        last_doc = cold_doc
        for _ in range(repeats):
            start = time.perf_counter()
            try:
                last_doc = parser.parse(path)
                times.append(time.perf_counter() - start)
            except Exception as exc:
                times.append(time.perf_counter() - start)
                if base_error is None:
                    base_error = f"{type(exc).__name__}: {exc}".replace("\n", " ")
                last_doc = None

        if base_error is not None:
            metrics = failed_metrics(expected, capabilities)
            success = False
            pages_parsed = 0
            parser_version = None
            warnings: list[str] = []
        else:
            assert last_doc is not None
            if repeats:
                metrics = score_document(last_doc, expected, capabilities)
            else:
                metrics = cold_metrics
            success = True
            pages_parsed = last_doc.page_count
            parser_version = last_doc.parser_version
            warnings = list(last_doc.warnings)

        if metrics["status"] == "unsupported":
            warnings = [
                *warnings,
                f"capability '{metrics['unsupported_capability']}' is not supported by {name}; "
                "fixture not scored (N/A)",
            ]

        best = min(times) if times else cold_time
        mean = sum(times) / len(times) if times else cold_time
        rows.append({
            **fixture_info,
            "success": success,
            "status": metrics["status"],
            "unsupported_capability": metrics["unsupported_capability"],
            "error": base_error,
            "parser_version": parser_version,
            "pages_parsed": pages_parsed,
            "warnings": warnings,
            **metrics,
            "time_seconds_cold": round(cold_time, 4),
            "time_seconds_best": round(best, 4),
            "time_seconds_mean": round(mean, 4),
            "pages_per_second": round(pages_parsed / mean, 3) if mean > 0 else 0.0,
        })

    ok_rows = [r for r in rows if r["status"] == "ok"]
    scored_rows = ok_rows + [r for r in rows if r["status"] == "failed"]
    unsupported_rows = [r for r in rows if r["status"] == "unsupported"]
    evaluable = len(rows) - len(unsupported_rows)

    def avg(rows_: list[dict], key: str) -> float | None:
        values = [r[key] for r in rows_ if r[key] is not None]
        return round(sum(values) / len(values), 4) if values else None

    cold_total = sum(r["time_seconds_cold"] for r in rows)
    warm_total = sum(r["time_seconds_mean"] for r in rows)
    construction_seconds = getattr(parser, "construction_seconds", None)
    first_fixture_cold = rows[0]["time_seconds_cold"] if rows else 0.0
    converter_init = getattr(parser, "converter_init_seconds", None)

    summary = {
        "parser": name,
        "skipped": False,
        "parser_version": next((r["parser_version"] for r in rows if r["parser_version"]), None),
        "fixtures_total": len(rows),
        "fixtures_succeeded": len(ok_rows),
        "fixtures_failed": sum(1 for r in rows if r["status"] == "failed"),
        "fixtures_unsupported": len(unsupported_rows),
        "fixtures_scored": len(ok_rows),
        "extraction_success_rate": round(len(ok_rows) / evaluable, 4) if evaluable else None,
        "failure_rate": round((evaluable - len(ok_rows)) / evaluable, 4) if evaluable else None,
        "page_coverage": avg(scored_rows, "page_coverage"),
        "text_coverage": avg(scored_rows, "text_coverage"),
        "structure_preservation": avg(scored_rows, "structure_preservation"),
        "reading_order_rate": avg(scored_rows, "reading_order"),
        "heading_preservation": avg(scored_rows, "heading_preservation"),
        "table_cell_coverage": avg(scored_rows, "table_cell_coverage"),
        "table_header_coverage": avg(scored_rows, "table_header_coverage"),
        "multipage_table_continuity": avg(scored_rows, "multipage_table_continuity"),
        "table_text_duplication": avg(scored_rows, "table_text_duplication"),
        "image_detection": avg(scored_rows, "image_detection"),
        "list_detection": avg(scored_rows, "list_detection"),
        "quality_score": avg(scored_rows, "quality_score"),
        "parser_construction_seconds": round(construction_seconds, 4) if construction_seconds is not None else None,
        "converter_init_seconds": round(converter_init, 4) if converter_init is not None else None,
        "cold_start_seconds": round((construction_seconds or 0.0) + first_fixture_cold, 4),
        "first_document_seconds": round(first_fixture_cold, 4),
        "cold_total_seconds": round(cold_total, 4),
        "subsequent_document_mean_seconds": avg(rows[1:], "time_seconds_mean"),
        "total_time_seconds": round((construction_seconds or 0.0) + cold_total + warm_total, 4),
        "throughput_pages_per_second": (
            round(sum(r["pages_parsed"] for r in rows) / warm_total, 3) if warm_total > 0 else 0.0
        ),
        "capabilities_supported": [k for k, v in capabilities.items() if v],
        "unsupported_capabilities": [k for k, v in capabilities.items() if not v],
    }
    return {"rows": rows, "summary": summary}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def archive_existing_reports(output_dir: Path) -> list[str]:
    archive_dir = output_dir / "archive"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = []
    for name in ("benchmark_report.json", "benchmark_results.csv", "benchmark_summary.csv"):
        target = output_dir / name
        if not target.exists():
            continue
        archive_dir.mkdir(parents=True, exist_ok=True)
        destination = archive_dir / f"{target.stem}.{stamp}{target.suffix}"
        shutil.copy2(target, destination)
        archived.append(str(destination))
    return archived


def per_fixture_comparison(all_rows: list[dict]) -> dict:
    comparison: dict[str, dict] = {}
    for row in all_rows:
        comparison.setdefault(row["fixture"], {})[row["parser"]] = {
            "status": row["status"],
            "success": row["success"],
            "pages_parsed": row["pages_parsed"],
            "text_coverage": row["text_coverage"],
            "structure_preservation": row["structure_preservation"],
            "reading_order": row["reading_order"],
            "quality_score": row["quality_score"],
            "time_seconds_cold": row["time_seconds_cold"],
            "time_seconds_mean": row["time_seconds_mean"],
        }
    return comparison


def capability_comparison(all_rows: list[dict]) -> dict:
    table: dict[str, dict] = {}
    for capability_key, row_keys in (
        ("text_extraction", ["text_coverage"]),
        ("structure", ["structure_preservation", "heading_preservation"]),
        ("tables", ["table_cell_coverage", "table_header_coverage", "multipage_table_continuity"]),
        ("images", ["image_detection"]),
        ("lists", ["list_detection"]),
    ):
        per_parser: dict[str, dict] = {}
        for name in PARSER_CAPABILITIES:
            parser_rows = [r for r in all_rows if r["parser"] == name]
            if not parser_rows:
                per_parser[name] = {"supported": PARSER_CAPABILITIES[name][capability_key], "score": None}
                continue
            supported = PARSER_CAPABILITIES[name][capability_key]
            if not supported:
                per_parser[name] = {"supported": False, "score": None}
                continue
            values = [
                r[row_keys[0]]
                for r in parser_rows
                if r["status"] == "ok" and r[row_keys[0]] is not None
            ]
            per_parser[name] = {
                "supported": True,
                "score": round(sum(values) / len(values), 4) if values else None,
            }
        table[capability_key] = per_parser
    table["ocr"] = {
        name: {"supported": PARSER_CAPABILITIES[name]["ocr"], "score": None}
        for name in PARSER_CAPABILITIES
    }
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parsers", nargs="+", choices=list(PARSER_BUILDERS), default=list(PARSER_BUILDERS))
    parser.add_argument("--repeats", type=int, default=1, help="steady-state parse repeats per fixture (default: 1)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--no-archive", action="store_true", help="do not archive existing report files before overwriting")
    args = parser.parse_args()

    fixtures = load_expectations()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archived = [] if args.no_archive else archive_existing_reports(args.output_dir)

    availability = available_parsers(args.parsers)
    all_rows: list[dict] = []
    summaries: list[dict] = []
    skipped: list[dict] = []

    for name, (instance, reason) in availability.items():
        if instance is None:
            skipped.append({"parser": name, "reason": reason})
            summaries.append(
                {
                    "parser": name,
                    "skipped": True,
                    "parser_version": None,
                    "fixtures_total": len(fixtures),
                    "fixtures_succeeded": 0,
                    "fixtures_failed": 0,
                    "fixtures_unsupported": 0,
                    "fixtures_scored": 0,
                    "extraction_success_rate": None,
                    "failure_rate": None,
                    "page_coverage": None,
                    "text_coverage": None,
                    "structure_preservation": None,
                    "reading_order_rate": None,
                    "heading_preservation": None,
                    "table_cell_coverage": None,
                    "table_header_coverage": None,
                    "multipage_table_continuity": None,
                    "table_text_duplication": None,
                    "image_detection": None,
                    "list_detection": None,
                    "quality_score": None,
                    "parser_construction_seconds": None,
                    "converter_init_seconds": None,
                    "cold_start_seconds": None,
                    "first_document_seconds": None,
                    "cold_total_seconds": None,
                    "subsequent_document_mean_seconds": None,
                    "total_time_seconds": None,
                    "throughput_pages_per_second": None,
                    "capabilities_supported": [],
                    "unsupported_capabilities": list(PARSER_CAPABILITIES[name]),
                }
            )
            continue
        print(f"Benchmarking {name} ...", flush=True)
        result = evaluate_parser(name, instance, fixtures, args.repeats)
        all_rows.extend({"parser": name, **row} for row in result["rows"])
        summaries.append(result["summary"])

    throughputs = [s["throughput_pages_per_second"] for s in summaries if not s["skipped"]]
    best_throughput = max(throughputs) if throughputs else None
    for summary in summaries:
        if summary["skipped"] or best_throughput in (None, 0.0):
            summary["performance_score"] = None
        else:
            summary["performance_score"] = round(summary["throughput_pages_per_second"] / best_throughput, 4)

    report = {
        "benchmark": "prome-pdf-parsers",
        "report_version": REPORT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "parsers": args.parsers,
            "repeats": args.repeats,
            "fixtures": sorted(fixtures),
            "output_dir": str(args.output_dir),
            "capabilities": PARSER_CAPABILITIES,
            "cold_start": "fresh parser instance; cold pass covers one parse of every fixture in corpus order, before the steady-state repeats",
            "quality_score": "mean of applicable non-None components: text_coverage, structure_preservation, reading_order, image_detection, list_detection",
            "performance_score": "steady-state pages/second relative to the fastest non-skipped parser (1.0 = fastest)",
            "note": "Quality and speed are reported separately. Unsupported capabilities are null (N/A), never 0. Skipped parsers have no measurements.",
        },
        "environment": library_versions(),
        "results": all_rows,
        "per_fixture": per_fixture_comparison(all_rows),
        "capabilities": capability_comparison(all_rows),
        "skipped": skipped,
        "summary": summaries,
        "archived_previous": archived,
    }

    json_path = args.output_dir / "benchmark_report.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n")

    row_fields = [
        "parser", "fixture", "success", "status", "unsupported_capability", "error", "parser_version",
        "bytes", "sha256", "pages_parsed",
        "time_seconds_cold", "time_seconds_best", "time_seconds_mean", "pages_per_second",
        "page_coverage", "text_coverage", "structure_preservation", "reading_order",
        "heading_preservation", "heading_hits", "table_blocks", "expected_tables",
        "table_cell_coverage", "table_header_coverage", "multipage_table_continuity", "table_text_duplication",
        "image_detection", "list_detection", "quality_score", "warnings",
    ]
    results_rows = [
        {**row, "warnings": "; ".join(row["warnings"]) if isinstance(row.get("warnings"), list) else row.get("warnings", "")}
        for row in all_rows
    ]
    write_csv(args.output_dir / "benchmark_results.csv", results_rows, row_fields)

    summary_fields = [
        "parser", "skipped", "parser_version", "fixtures_total", "fixtures_succeeded", "fixtures_failed",
        "fixtures_unsupported", "fixtures_scored",
        "extraction_success_rate", "failure_rate",
        "page_coverage", "text_coverage", "structure_preservation", "reading_order_rate",
        "heading_preservation", "table_cell_coverage", "table_header_coverage",
        "multipage_table_continuity", "table_text_duplication", "image_detection", "list_detection",
        "quality_score", "performance_score",
        "parser_construction_seconds", "converter_init_seconds", "cold_start_seconds",
        "first_document_seconds", "cold_total_seconds", "subsequent_document_mean_seconds",
        "total_time_seconds", "throughput_pages_per_second",
        "capabilities_supported", "unsupported_capabilities",
    ]
    summary_rows = [
        {
            **s,
            "capabilities_supported": "; ".join(s.get("capabilities_supported") or []),
            "unsupported_capabilities": "; ".join(s.get("unsupported_capabilities") or []),
        }
        for s in summaries
    ]
    write_csv(args.output_dir / "benchmark_summary.csv", summary_rows, summary_fields)

    print(f"\nWrote {json_path}")
    print(f"Wrote {args.output_dir / 'benchmark_results.csv'}")
    print(f"Wrote {args.output_dir / 'benchmark_summary.csv'}")
    if archived:
        print(f"Archived previous reports: {', '.join(archived)}")
    for summary in summaries:
        if summary["skipped"]:
            print(f"  {summary['parser']}: SKIPPED")
        else:
            print(
                f"  {summary['parser']}: success={summary['extraction_success_rate']} "
                f"quality={summary['quality_score']} "
                f"text={summary['text_coverage']} structure={summary['structure_preservation']} "
                f"cold_start={summary['cold_start_seconds']}s "
                f"throughput={summary['throughput_pages_per_second']} pages/s"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
