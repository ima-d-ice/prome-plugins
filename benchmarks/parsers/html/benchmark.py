"""Reproducible HTML parser benchmark for Prome RAG Plugins.

Parses the synthetic fixture corpus (fixtures/) with each available parser
and reports extraction quality and speed separately:

  Quality:  text coverage, structure preservation (headings, heading levels,
            table cells), list preservation, link preservation, image
            detection, metadata completeness, DOM ordering, boilerplate
            handling, forbidden-content isolation (scripts/styles)
  Speed:    cold-start time, first-document time, steady-state time per
            fixture, documents-per-second throughput, blocks-per-second

Extraction scopes
-----------------
The two parsers optimize for different things, so the same ground truth is
scored through two extraction scopes:

  full_dom      (beautifulsoup): the parser is expected to preserve the
                                 document as it is, including navigation,
                                 header, sidebar and footer. Boilerplate
                                 retention is reported informationally and is
                                 never penalized.
  main_content  (trafilatura):   the parser is expected to isolate the main
                                 article/document. Boilerplate removal earns
                                 credit (boilerplate_removal = 1 - retention)
                                 and counts toward the quality score.

Capability model
----------------
Unsupported capabilities are reported as N/A (null in JSON, empty in CSV) and
are excluded from every average. They are never treated as ordinary
extraction failures and never scored as success or invented. Notably,
Trafilatura does not surface images, so image detection is N/A for it.

Statuses per (parser, fixture) row:
  ok       parsed and scored normally
  failed   parsing raised an exception (ordinary extraction failure,
           scored as 0 for the achieved metrics)

Quality score: mean of the applicable, non-None quality components
(text coverage, structure preservation, reading order, metadata completeness,
list detection, link preservation, image detection, boilerplate removal,
forbidden-text isolation). Components that do not apply to the
fixture/parser pair are excluded, not scored as 1 or 0.

Performance score: steady-state documents/second relative to the fastest
non-skipped parser (1.0 = fastest). Documented; not an absolute grade.

Cold vs steady state
--------------------
Each parser gets a fresh instance. The first pass over the fixture corpus is
the cold pass; subsequent repeats are the steady-state pass. The report
separates both.

Existing reports are never silently destroyed: before overwriting, previous
report files are copied into reports/archive/ with a UTC timestamp.

Unavailable parsers are recorded as "skipped" -- never as fake numbers.

Usage:
  python benchmarks/parsers/html/benchmark.py
  python benchmarks/parsers/html/benchmark.py --parsers beautifulsoup
  python benchmarks/parsers/html/benchmark.py --repeats 3 --output-dir reports
  python benchmarks/parsers/html/benchmark.py --no-archive
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

REPORT_VERSION = 1

PARSER_BUILDERS = {
    "beautifulsoup": ("parsers.html.beautifulsoup", "BeautifulSoupParser", ("bs4",)),
    "trafilatura": ("parsers.html.trafilatura", "TrafilaturaParser", ("trafilatura",)),
}

# Capabilities each parser implementation actually provides, reviewed against
# parsers/html/*/. full_dom=True means "expected to keep boilerplate";
# full_dom=False means "expected to isolate main content".
PARSER_CAPABILITIES = {
    "beautifulsoup": {
        "text_extraction": True,
        "structure": True,      # tag dispatch + heading levels from h1-h6
        "tables": True,         # thead/tbody/th/td + colspan expansion
        "lists": True,          # ul/ol/dl, nested
        "images": True,         # <img> -> IMAGE block with alt text
        "links": True,          # <a href> -> LINK block
        "full_dom": True,       # scope: keeps nav/header/footer (default)
        "metadata": True,       # title/meta/og/canonical/JSON-LD
        "code": True,           # <pre>/<code> -> CODE block
    },
    "trafilatura": {
        "text_extraction": True,
        "structure": True,      # headings/lists/tables via lxml XML output
        "tables": True,         # <table> kept when include_tables=True (default)
        "lists": True,          # <list> elements kept
        "images": False,        # images are not surfaced as blocks
        "links": True,          # <ref target> -> LINK block; include_links=True (default)
        "full_dom": False,      # scope: isolates main content (default)
        "metadata": True,       # title/author/date/description/... via bare_extraction
        "code": True,           # <code> blocks kept
    },
}

QUALITY_COMPONENT_KEYS = (
    "text_coverage",
    "structure_preservation",
    "reading_order",
    "metadata_completeness",
    "list_detection",
    "link_preservation",
    "image_detection",
    "boilerplate_removal",
    "forbidden_text_isolation",
)


def load_expectations() -> dict:
    if not EXPECTATIONS_PATH.exists():
        sys.exit(f"expectations.json not found at {EXPECTATIONS_PATH}")
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
    for dist in ("beautifulsoup4", "soupsieve", "trafilatura", "lxml", "htmldate", "courlan", "justext"):
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
    return [b for b in doc.blocks if b.block_type.value == block_type]


def score_document(doc, expected: dict, capabilities: dict[str, bool]) -> dict:
    """Score a parsed document against ground truth.

    Returns metrics where every component is either a float in [0, 1] or None
    (not applicable / capability unsupported -- never scored, never fudged).
    """
    text = doc.text.lower()

    expected_text = expected.get("expected_text", [])
    text_coverage = round(_content_hits(text, expected_text) / len(expected_text), 4) if expected_text else None

    # --- headings / structure ---
    expected_headings = expected.get("expected_headings", [])
    expected_levels = expected.get("expected_heading_levels", {})
    heading_preservation = None
    heading_level_accuracy = None
    heading_hits = None
    heading_blocks = _blocks_of_type(doc, "heading")
    heading_texts = [b.content.lower() for b in heading_blocks]
    if capabilities.get("structure"):
        heading_hits = sum(
            1 for eh in expected_headings if any(eh.lower() in h for h in heading_texts)
        )
        heading_preservation = _fraction(heading_hits, len(expected_headings))
        if expected_levels:
            correct = 0
            for heading, level in expected_levels.items():
                match = next((b for b in heading_blocks if heading.lower() in b.content.lower()), None)
                if match is not None and match.metadata.get("heading_level") == level:
                    correct += 1
            heading_level_accuracy = round(correct / len(expected_levels), 4)

    # --- tables ---
    table_blocks = _blocks_of_type(doc, "table")
    expected_tables = expected.get("expected_tables", 0)
    expected_cells = expected.get("expected_table_cells", [])
    expected_header = expected.get("expected_table_header", [])
    table_cell_coverage = None
    table_header_coverage = None
    if capabilities.get("tables"):
        table_content = " ".join(b.content.lower() for b in table_blocks)
        if expected_cells:
            table_cell_coverage = round(_content_hits(table_content, expected_cells) / len(expected_cells), 4)
        if expected_header:
            table_header_coverage = round(_content_hits(table_content, expected_header) / len(expected_header), 4)

    # --- structure preservation: mean of applicable structural subscores ---
    structure_subscores = [s for s in (heading_preservation, table_cell_coverage) if s is not None]
    if not capabilities.get("structure"):
        structure_preservation = None
    elif structure_subscores:
        structure_preservation = round(sum(structure_subscores) / len(structure_subscores), 4)
    else:
        structure_preservation = None  # nothing structural expected in this fixture

    # --- reading order (DOM ordering proxy) ---
    markers = expected.get("reading_order", [])
    reading_order = None
    if markers and capabilities.get("text_extraction"):
        positions = [text.find(marker.lower()) for marker in markers]
        reading_order = float(all(pos >= 0 for pos in positions) and positions == sorted(positions))

    # --- lists ---
    expected_list_items = expected.get("expected_list_items")
    list_detection = None
    if expected_list_items:
        if capabilities.get("lists"):
            list_content = " ".join(b.content.lower() for b in _blocks_of_type(doc, "list"))
            list_detection = round(_content_hits(list_content, expected_list_items) / len(expected_list_items), 4)

    # --- links ---
    expected_hrefs = expected.get("expected_link_hrefs")
    link_preservation = None
    link_blocks = _blocks_of_type(doc, "link")
    if expected_hrefs:
        if capabilities.get("links"):
            found_hrefs = " ".join(str(b.metadata.get("href", "")).lower() for b in link_blocks)
            link_preservation = round(sum(1 for h in expected_hrefs if h.lower() in found_hrefs) / len(expected_hrefs), 4)

    # --- images ---
    expected_alts = expected.get("expected_images")
    image_detection = None
    if expected_alts:
        if capabilities.get("images"):
            image_content = " ".join(
                (b.content + " " + str(b.metadata.get("alt", ""))).lower() for b in _blocks_of_type(doc, "image")
            )
            image_detection = round(_content_hits(image_content, expected_alts) / len(expected_alts), 4)

    # --- metadata completeness ---
    expected_metadata = expected.get("expected_metadata", [])
    metadata_completeness = None
    if expected_metadata:
        if capabilities.get("metadata"):
            def _present(key: str) -> bool:
                value = doc.metadata.get(key)
                return value not in (None, "", [], {})
            metadata_completeness = round(sum(1 for k in expected_metadata if _present(k)) / len(expected_metadata), 4)

    # --- boilerplate handling ---
    boilerplate = expected.get("boilerplate_text", [])
    boilerplate_retention = None
    boilerplate_removal = None
    if boilerplate:
        boilerplate_hits = _content_hits(text, boilerplate)
        boilerplate_retention = round(boilerplate_hits / len(boilerplate), 4)
        if not capabilities.get("full_dom"):
            # main-content extraction: removing boilerplate earns credit
            boilerplate_removal = round(1.0 - boilerplate_retention, 4)

    # --- forbidden content isolation (scripts/styles/secrets) ---
    forbidden = expected.get("forbidden_text", [])
    forbidden_text_isolation = None
    if forbidden:
        present = _content_hits(text, forbidden)
        forbidden_text_isolation = round(1.0 - present / len(forbidden), 4)

    components = {
        "text_coverage": text_coverage,
        "structure_preservation": structure_preservation,
        "reading_order": reading_order,
        "metadata_completeness": metadata_completeness,
        "list_detection": list_detection,
        "link_preservation": link_preservation,
        "image_detection": image_detection,
        "boilerplate_removal": boilerplate_removal,
        "forbidden_text_isolation": forbidden_text_isolation,
    }
    applicable = [v for v in components.values() if v is not None]
    quality_score = round(sum(applicable) / len(applicable), 4) if applicable else None

    return {
        "status": "ok",
        "text_coverage": text_coverage,
        "structure_preservation": structure_preservation,
        "reading_order": reading_order,
        "heading_preservation": heading_preservation,
        "heading_level_accuracy": heading_level_accuracy,
        "heading_hits": heading_hits,
        "table_blocks": len(table_blocks),
        "expected_tables": expected_tables,
        "table_cell_coverage": table_cell_coverage,
        "table_header_coverage": table_header_coverage,
        "list_detection": list_detection,
        "link_preservation": link_preservation,
        "image_detection": image_detection,
        "metadata_completeness": metadata_completeness,
        "boilerplate_retention": boilerplate_retention,
        "boilerplate_removal": boilerplate_removal,
        "forbidden_text_isolation": forbidden_text_isolation,
        "quality_score": quality_score,
    }


def _component_applicable(expected: dict, capabilities: dict[str, bool]) -> dict[str, bool]:
    """Which quality components apply to this fixture/parser pair."""
    boilerplate = bool(expected.get("boilerplate_text")) and not capabilities.get("full_dom")
    return {
        "text_coverage": bool(expected.get("expected_text")),
        "structure_preservation": capabilities.get("structure", False)
        and bool(
            expected.get("expected_headings")
            or (capabilities.get("tables") and expected.get("expected_table_cells"))
        ),
        "reading_order": bool(expected.get("reading_order")) and capabilities.get("text_extraction", False),
        "metadata_completeness": bool(expected.get("expected_metadata")) and capabilities.get("metadata", False),
        "list_detection": bool(expected.get("expected_list_items")) and capabilities.get("lists", False),
        "link_preservation": bool(expected.get("expected_link_hrefs")) and capabilities.get("links", False),
        "image_detection": bool(expected.get("expected_images")) and capabilities.get("images", False),
        "boilerplate_removal": boilerplate,
        "forbidden_text_isolation": bool(expected.get("forbidden_text")),
    }


def failed_metrics(expected: dict, capabilities: dict[str, bool]) -> dict:
    """Metrics for a fixture that raised despite being parseable.

    Zero for every applicable component (the parser achieved nothing),
    None for components that do not apply -- never a fake score.
    """
    applicable = _component_applicable(expected, capabilities)
    zeros = {key: 0.0 for key, applies in applicable.items() if applies}
    table_applicable = capabilities.get("tables", False) and bool(
        expected.get("expected_table_cells") or expected.get("expected_table_header")
    )
    metrics = {
        "status": "failed",
        "text_coverage": zeros.get("text_coverage"),
        "structure_preservation": zeros.get("structure_preservation"),
        "reading_order": zeros.get("reading_order"),
        "heading_preservation": 0.0
        if capabilities.get("structure") and expected.get("expected_headings")
        else None,
        "heading_level_accuracy": 0.0
        if capabilities.get("structure") and expected.get("expected_heading_levels")
        else None,
        "heading_hits": 0 if capabilities.get("structure") and expected.get("expected_headings") else None,
        "table_blocks": 0,
        "expected_tables": expected.get("expected_tables", 0),
        "table_cell_coverage": 0.0 if table_applicable and expected.get("expected_table_cells") else None,
        "table_header_coverage": 0.0 if table_applicable and expected.get("expected_table_header") else None,
        "list_detection": zeros.get("list_detection"),
        "link_preservation": zeros.get("link_preservation"),
        "image_detection": zeros.get("image_detection"),
        "metadata_completeness": zeros.get("metadata_completeness"),
        "boilerplate_retention": None,
        "boilerplate_removal": zeros.get("boilerplate_removal"),
        "forbidden_text_isolation": zeros.get("forbidden_text_isolation"),
        "quality_score": round(sum(zeros.values()) / len(zeros), 4) if zeros else None,
    }
    return metrics


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def evaluate_parser(name: str, parser, fixtures: dict, repeats: int) -> dict:
    capabilities = PARSER_CAPABILITIES[name]
    rows = []

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
            blocks_parsed = 0
            parser_version = None
            warnings: list[str] = []
        else:
            assert last_doc is not None
            if repeats:
                metrics = score_document(last_doc, expected, capabilities)
            else:
                metrics = cold_metrics
            success = True
            blocks_parsed = last_doc.block_count
            parser_version = last_doc.parser_version
            warnings = list(last_doc.warnings)

        best = min(times) if times else cold_time
        mean = sum(times) / len(times) if times else cold_time
        rows.append({
            **fixture_info,
            "success": success,
            "error": base_error,
            "parser_version": parser_version,
            "blocks_parsed": blocks_parsed,
            "warnings": warnings,
            **metrics,
            "time_seconds_cold": round(cold_time, 4),
            "time_seconds_best": round(best, 4),
            "time_seconds_mean": round(mean, 4),
            "documents_per_second": round(1.0 / mean, 3) if mean > 0 else 0.0,
            "blocks_per_second": round(blocks_parsed / mean, 3) if mean > 0 else 0.0,
            "bytes_per_second": round(fixture_info["bytes"] / mean, 1) if mean > 0 else 0.0,
        })

    ok_rows = [r for r in rows if r["status"] == "ok"]

    def avg(rows_: list[dict], key: str) -> float | None:
        values = [r[key] for r in rows_ if r[key] is not None]
        return round(sum(values) / len(values), 4) if values else None

    cold_total = sum(r["time_seconds_cold"] for r in rows)
    warm_total = sum(r["time_seconds_mean"] for r in rows)
    construction_seconds = getattr(parser, "construction_seconds", None)
    first_fixture_cold = rows[0]["time_seconds_cold"] if rows else 0.0

    summary = {
        "parser": name,
        "skipped": False,
        "parser_version": next((r["parser_version"] for r in rows if r["parser_version"]), None),
        "fixtures_total": len(rows),
        "fixtures_succeeded": len(ok_rows),
        "fixtures_failed": sum(1 for r in rows if r["status"] == "failed"),
        "extraction_success_rate": round(len(ok_rows) / len(rows), 4) if rows else None,
        "failure_rate": round((len(rows) - len(ok_rows)) / len(rows), 4) if rows else None,
        "text_coverage": avg(rows, "text_coverage"),
        "structure_preservation": avg(rows, "structure_preservation"),
        "reading_order_rate": avg(rows, "reading_order"),
        "heading_preservation": avg(rows, "heading_preservation"),
        "heading_level_accuracy": avg(rows, "heading_level_accuracy"),
        "table_cell_coverage": avg(rows, "table_cell_coverage"),
        "table_header_coverage": avg(rows, "table_header_coverage"),
        "list_detection": avg(rows, "list_detection"),
        "link_preservation": avg(rows, "link_preservation"),
        "image_detection": avg(rows, "image_detection"),
        "metadata_completeness": avg(rows, "metadata_completeness"),
        "boilerplate_retention": avg(rows, "boilerplate_retention"),
        "boilerplate_removal": avg(rows, "boilerplate_removal"),
        "forbidden_text_isolation": avg(rows, "forbidden_text_isolation"),
        "quality_score": avg(rows, "quality_score"),
        "parser_construction_seconds": round(construction_seconds, 4) if construction_seconds is not None else None,
        "cold_start_seconds": round((construction_seconds or 0.0) + first_fixture_cold, 4),
        "first_document_seconds": round(first_fixture_cold, 4),
        "cold_total_seconds": round(cold_total, 4),
        "subsequent_document_mean_seconds": avg(rows[1:], "time_seconds_mean"),
        "total_time_seconds": round((construction_seconds or 0.0) + cold_total + warm_total, 4),
        "throughput_documents_per_second": (
            round(len(rows) / warm_total, 3) if warm_total > 0 else 0.0
        ),
        "throughput_blocks_per_second": (
            round(sum(r["blocks_parsed"] for r in rows) / warm_total, 3) if warm_total > 0 else 0.0
        ),
        "capabilities_supported": [k for k, v in capabilities.items() if v],
        "unsupported_capabilities": [k for k, v in capabilities.items() if not v],
        "extraction_scope": "full_dom" if capabilities.get("full_dom") else "main_content",
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
            "blocks_parsed": row["blocks_parsed"],
            "text_coverage": row["text_coverage"],
            "structure_preservation": row["structure_preservation"],
            "reading_order": row["reading_order"],
            "metadata_completeness": row["metadata_completeness"],
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
        ("tables", ["table_cell_coverage", "table_header_coverage"]),
        ("lists", ["list_detection"]),
        ("links", ["link_preservation"]),
        ("images", ["image_detection"]),
        ("metadata", ["metadata_completeness"]),
    ):
        per_parser: dict[str, dict] = {}
        for name in PARSER_CAPABILITIES:
            parser_rows = [r for r in all_rows if r["parser"] == name]
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
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parsers", nargs="+", choices=list(PARSER_BUILDERS), default=list(PARSER_BUILDERS))
    parser.add_argument("--repeats", type=int, default=3, help="steady-state parse repeats per fixture (default: 3)")
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
                    "extraction_success_rate": None,
                    "failure_rate": None,
                    "text_coverage": None,
                    "structure_preservation": None,
                    "reading_order_rate": None,
                    "heading_preservation": None,
                    "heading_level_accuracy": None,
                    "table_cell_coverage": None,
                    "table_header_coverage": None,
                    "list_detection": None,
                    "link_preservation": None,
                    "image_detection": None,
                    "metadata_completeness": None,
                    "boilerplate_retention": None,
                    "boilerplate_removal": None,
                    "forbidden_text_isolation": None,
                    "quality_score": None,
                    "parser_construction_seconds": None,
                    "cold_start_seconds": None,
                    "first_document_seconds": None,
                    "cold_total_seconds": None,
                    "subsequent_document_mean_seconds": None,
                    "total_time_seconds": None,
                    "throughput_documents_per_second": None,
                    "throughput_blocks_per_second": None,
                    "capabilities_supported": [],
                    "unsupported_capabilities": list(PARSER_CAPABILITIES[name]),
                    "extraction_scope": "full_dom" if PARSER_CAPABILITIES[name].get("full_dom") else "main_content",
                }
            )
            continue
        print(f"Benchmarking {name} ...", flush=True)
        result = evaluate_parser(name, instance, fixtures, args.repeats)
        all_rows.extend({"parser": name, **row} for row in result["rows"])
        summaries.append(result["summary"])

    throughputs = [s["throughput_documents_per_second"] for s in summaries if not s["skipped"]]
    best_throughput = max(throughputs) if throughputs else None
    for summary in summaries:
        if summary["skipped"] or best_throughput in (None, 0.0):
            summary["performance_score"] = None
        else:
            summary["performance_score"] = round(summary["throughput_documents_per_second"] / best_throughput, 4)

    report = {
        "benchmark": "prome-html-parsers",
        "report_version": REPORT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "parsers": args.parsers,
            "repeats": args.repeats,
            "fixtures": sorted(fixtures),
            "output_dir": str(args.output_dir),
            "capabilities": PARSER_CAPABILITIES,
            "extraction_scopes": {
                "beautifulsoup": "full_dom (preserve nav/header/footer as authored; retention reported, never penalized)",
                "trafilatura": "main_content (isolate main article; boilerplate_removal earns credit and counts toward quality)",
            },
            "cold_start": "fresh parser instance; cold pass covers one parse of every fixture in corpus order, before the steady-state repeats",
            "quality_score": "mean of applicable non-None components: text_coverage, structure_preservation, reading_order, metadata_completeness, list_detection, link_preservation, image_detection, boilerplate_removal, forbidden_text_isolation",
            "performance_score": "steady-state documents/second relative to the fastest non-skipped parser (1.0 = fastest)",
            "note": "Quality and speed are reported separately. Unsupported capabilities are null (N/A), never 0. Skipped parsers have no measurements. Metrics are computed, never invented.",
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
        "parser", "fixture", "success", "status", "error", "parser_version",
        "bytes", "sha256", "blocks_parsed",
        "time_seconds_cold", "time_seconds_best", "time_seconds_mean",
        "documents_per_second", "blocks_per_second", "bytes_per_second",
        "text_coverage", "structure_preservation", "reading_order",
        "heading_preservation", "heading_level_accuracy", "heading_hits",
        "table_blocks", "expected_tables", "table_cell_coverage", "table_header_coverage",
        "list_detection", "link_preservation", "image_detection", "metadata_completeness",
        "boilerplate_retention", "boilerplate_removal", "forbidden_text_isolation",
        "quality_score", "warnings",
    ]
    results_rows = [
        {**row, "warnings": "; ".join(row["warnings"]) if isinstance(row.get("warnings"), list) else row.get("warnings", "")}
        for row in all_rows
    ]
    write_csv(args.output_dir / "benchmark_results.csv", results_rows, row_fields)

    summary_fields = [
        "parser", "skipped", "parser_version", "fixtures_total", "fixtures_succeeded", "fixtures_failed",
        "extraction_success_rate", "failure_rate",
        "text_coverage", "structure_preservation", "reading_order_rate",
        "heading_preservation", "heading_level_accuracy",
        "table_cell_coverage", "table_header_coverage",
        "list_detection", "link_preservation", "image_detection", "metadata_completeness",
        "boilerplate_retention", "boilerplate_removal", "forbidden_text_isolation",
        "quality_score", "performance_score",
        "parser_construction_seconds", "cold_start_seconds",
        "first_document_seconds", "cold_total_seconds", "subsequent_document_mean_seconds",
        "total_time_seconds", "throughput_documents_per_second", "throughput_blocks_per_second",
        "capabilities_supported", "unsupported_capabilities", "extraction_scope",
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
                f"throughput={summary['throughput_documents_per_second']} docs/s"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
