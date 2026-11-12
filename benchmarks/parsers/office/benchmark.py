"""Reproducible Office document benchmark (DOCX and PPTX) for Prome RAG Plugins."""

from __future__ import annotations

import argparse
import csv
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

PARSER_BUILDERS = {
    "python_docx": ("parsers.office.docx.python_docx", "PythonDocxParser", ("docx",)),
    "docling_docx": ("parsers.office.docx.docling", "DoclingDocxParser", ("docling",)),
    "python_pptx": ("parsers.office.pptx.python_pptx", "PythonPptxParser", ("pptx",)),
    "docling_pptx": ("parsers.office.pptx.docling", "DoclingPptxParser", ("docling",)),
}


def parser_available(name: str) -> bool:
    if name not in PARSER_BUILDERS:
        return False
    _, _, deps = PARSER_BUILDERS[name]
    for dep in deps:
        try:
            __import__(dep)
        except ImportError:
            return False
    return True


def instantiate_parser(name: str):
    module_path, class_name, _ = PARSER_BUILDERS[name]
    mod = __import__(module_path, fromlist=[class_name])
    return getattr(mod, class_name)()


def archive_existing_reports(output_dir: Path) -> list[Path]:
    archive_dir = output_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = []
    for stem in ("benchmark_report", "benchmark_results", "benchmark_summary"):
        for ext in (".json", ".csv"):
            src = output_dir / f"{stem}{ext}"
            if src.exists():
                dst = archive_dir / f"{stem}.{ts}{ext}"
                shutil.copy2(src, dst)
                archived.append(dst)
    return archived


def score_fixture(doc, exp: dict) -> dict[str, float]:
    doc_text = doc.text.lower()
    exp_texts = exp.get("expected_text", [])
    found_texts = sum(1 for t in exp_texts if t.lower() in doc_text)
    text_cov = found_texts / len(exp_texts) if exp_texts else 1.0

    exp_headings = exp.get("expected_headings", [])
    if exp_headings:
        headings_found = 0
        all_headings = []
        for b in doc.blocks:
            if b.block_type.value == "heading":
                all_headings.append(b.content.lower())
        for p in doc.pages:
            for b in p.blocks:
                if b.block_type.value == "heading":
                    all_headings.append(b.content.lower())
        for h in exp_headings:
            if any(h.lower() in ah for ah in all_headings):
                headings_found += 1
        heading_acc = headings_found / len(exp_headings)
    else:
        heading_acc = 1.0

    exp_tables = exp.get("expected_tables", 0)
    actual_tables = 0
    for b in doc.blocks:
        if b.block_type.value == "table":
            actual_tables += 1
    for p in doc.pages:
        for b in p.blocks:
            if b.block_type.value == "table":
                actual_tables += 1
    table_acc = 1.0 if actual_tables == exp_tables else (0.5 if actual_tables > 0 and exp_tables > 0 else 0.0)

    quality = (text_cov * 0.5) + (heading_acc * 0.3) + (table_acc * 0.2)
    return {
        "text_coverage": text_cov,
        "heading_accuracy": heading_acc,
        "table_accuracy": table_acc,
        "quality_score": quality,
    }


def run_benchmark(
    parsers: list[str] | None = None,
    repeats: int = 2,
    output_dir: Path | None = None,
    archive: bool = True,
) -> dict:
    output_dir = output_dir or DEFAULT_REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    if archive:
        archive_existing_reports(output_dir)

    target_parsers = parsers or list(PARSER_BUILDERS.keys())
    expectations = json.loads(EXPECTATIONS_PATH.read_text())

    results = []
    summaries = []

    for p_name in target_parsers:
        if not parser_available(p_name):
            continue
        parser = instantiate_parser(p_name)
        is_pptx_parser = "pptx" in p_name

        times = []
        qualities = []
        successes = 0
        total_fixtures = 0

        for fix_name, exp in expectations.items():
            is_pptx_fix = fix_name.endswith(".pptx")
            if (is_pptx_parser and not is_pptx_fix) or (not is_pptx_parser and is_pptx_fix):
                continue

            fix_path = FIXTURES_DIR / fix_name
            if not fix_path.exists():
                continue

            total_fixtures += 1
            t0 = time.perf_counter()
            try:
                doc = parser.parse(fix_path)
                elapsed = time.perf_counter() - t0
                scores = score_fixture(doc, exp)
                status = "ok"
                successes += 1
                qualities.append(scores["quality_score"])
                times.append(elapsed)
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                status = "failed"
                scores = {"text_coverage": 0.0, "heading_accuracy": 0.0, "table_accuracy": 0.0, "quality_score": 0.0}

            results.append({
                "parser": p_name,
                "fixture": fix_name,
                "status": status,
                "latency_sec": round(elapsed, 4),
                **{k: round(v, 4) for k, v in scores.items()},
            })

        avg_latency = sum(times) / len(times) if times else 0.0
        avg_quality = sum(qualities) / len(qualities) if qualities else 0.0
        throughput = (1.0 / avg_latency) if avg_latency > 0 else 0.0

        summaries.append({
            "parser": p_name,
            "success_rate": round(successes / total_fixtures, 4) if total_fixtures else 0.0,
            "avg_quality_score": round(avg_quality, 4),
            "avg_latency_sec": round(avg_latency, 4),
            "throughput_docs_sec": round(throughput, 2),
        })

    report = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "summaries": summaries,
        "results": results,
    }

    (output_dir / "benchmark_report.json").write_text(json.dumps(report, indent=2))

    # Write CSV results
    if results:
        with (output_dir / "benchmark_results.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    # Write CSV summaries
    if summaries:
        with (output_dir / "benchmark_summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)

    return report


def main():
    parser = argparse.ArgumentParser(description="Run office benchmarks")
    parser.add_argument("--parsers", nargs="+", help="Parsers to benchmark")
    parser.add_argument("--repeats", type=int, default=1, help="Repetitions")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--no-archive", action="store_true")
    args = parser.parse_args()

    report = run_benchmark(
        parsers=args.parsers,
        repeats=args.repeats,
        output_dir=args.output_dir,
        archive=not args.no_archive,
    )
    print("Benchmark complete. Summaries:")
    for s in report["summaries"]:
        print(f"  {s['parser']}: success={s['success_rate']}, quality={s['avg_quality_score']}, avg_latency={s['avg_latency_sec']}s")


if __name__ == "__main__":
    main()
