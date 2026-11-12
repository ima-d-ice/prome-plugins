"""Reproducible Email benchmark (EML and MBOX) for Prome RAG Plugins."""

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
    "eml_stdlib": ("parsers.email.eml", "EmlParser", ()),
    "mbox_stdlib": ("parsers.email.mbox", "MboxParser", ()),
}


def parser_available(name: str) -> bool:
    return name in PARSER_BUILDERS


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

    exp_subj = exp.get("expected_subject")
    if exp_subj:
        subj_acc = 1.0 if exp_subj.lower() in doc.metadata.get("subject", "").lower() else 0.0
    else:
        subj_acc = 1.0

    quality = (text_cov * 0.7) + (subj_acc * 0.3)
    return {
        "text_coverage": text_cov,
        "subject_accuracy": subj_acc,
        "quality_score": quality,
    }


def run_benchmark(
    parsers: list[str] | None = None,
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
        is_mbox = p_name == "mbox_stdlib"

        times = []
        qualities = []
        successes = 0
        total_fixtures = 0

        for fix_name, exp in expectations.items():
            is_mbox_fix = fix_name.endswith(".mbox")
            if (is_mbox and not is_mbox_fix) or (not is_mbox and is_mbox_fix):
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
            except Exception:
                elapsed = time.perf_counter() - t0
                status = "failed"
                scores = {"text_coverage": 0.0, "subject_accuracy": 0.0, "quality_score": 0.0}

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
    if results:
        with (output_dir / "benchmark_results.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
    if summaries:
        with (output_dir / "benchmark_summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)

    return report


def main():
    parser = argparse.ArgumentParser(description="Run email benchmarks")
    parser.add_argument("--parsers", nargs="+", help="Parsers to benchmark")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--no-archive", action="store_true")
    args = parser.parse_args()

    report = run_benchmark(
        parsers=args.parsers,
        output_dir=args.output_dir,
        archive=not args.no_archive,
    )
    print("Benchmark complete. Summaries:")
    for s in report["summaries"]:
        print(f"  {s['parser']}: success={s['success_rate']}, quality={s['avg_quality_score']}, avg_latency={s['avg_latency_sec']}s")


if __name__ == "__main__":
    main()
