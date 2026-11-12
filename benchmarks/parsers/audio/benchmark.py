"""Reproducible faster-whisper audio transcription benchmark for Prome RAG Plugins.

Metrics per fixture:
  text_coverage      fraction of expected_text substrings present in output (1.0 if none expected)
  segment_coverage   1.0 if at least min_blocks segments extracted, else proportional
  language_accuracy  1.0 if detected language matches expected_language (or none expected)
  wer                word error rate against expected_transcript, when provided (else N/A)
  quality_score      0.6*text_coverage + 0.2*segment_coverage + 0.2*language_accuracy

Model weights are NOT downloaded by this benchmark implicitly beyond what
faster-whisper does on first model load (default model: tiny). Use
--model-size to select a different model; larger models are downloaded to the
Hugging Face cache on first use.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
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
    "faster_whisper": ("parsers.audio.faster_whisper", "FasterWhisperParser", ("faster_whisper", "av", "numpy")),
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


def instantiate_parser(name: str, model_size: str | None = None):
    module_path, class_name, _ = PARSER_BUILDERS[name]
    mod = __import__(module_path, fromlist=[class_name])
    cls = getattr(mod, class_name)
    return cls(model_size=model_size) if model_size else cls()


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


_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = _words(reference), _words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    # word-level Levenshtein
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        curr = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + (r != h))
        prev = curr
    return round(prev[-1] / len(ref), 4)


def score_fixture(doc, exp: dict) -> dict:
    doc_text = doc.text.lower()
    exp_texts = exp.get("expected_text", [])
    found = sum(1 for t in exp_texts if t.lower() in doc_text)
    text_cov = found / len(exp_texts) if exp_texts else 1.0

    actual_blocks = len(doc.blocks)
    min_b = exp.get("min_blocks", 1)
    seg_cov = 1.0 if actual_blocks >= min_b else (actual_blocks / min_b if min_b else 1.0)

    expected_lang = exp.get("expected_language")
    if expected_lang:
        lang_acc = 1.0 if doc.metadata.get("language") == expected_lang else 0.0
    else:
        lang_acc = 1.0

    scores: dict[str, float | None] = {
        "text_coverage": round(text_cov, 4),
        "segment_coverage": round(seg_cov, 4),
        "language_accuracy": round(lang_acc, 4),
        "wer": word_error_rate(exp["expected_transcript"], doc.text) if exp.get("expected_transcript") else None,
    }
    scores["quality_score"] = round(0.6 * text_cov + 0.2 * seg_cov + 0.2 * lang_acc, 4)
    return scores


def run_benchmark(
    parsers: list[str] | None = None,
    output_dir: Path | None = None,
    archive: bool = True,
    model_size: str | None = None,
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
            summaries.append({
                "parser": p_name,
                "status": "skipped",
                "success_rate": 0.0,
                "avg_quality_score": 0.0,
                "avg_latency_sec": 0.0,
                "throughput_docs_sec": 0.0,
                "notes": "Dependencies not installed (pip install 'prome-rag-plugins[audio]')",
            })
            continue

        try:
            parser = instantiate_parser(p_name, model_size=model_size)
            # Warm-up / availability check: triggers model load (and download on first use).
            first_fixture = next(iter(expectations))
            parser.parse(FIXTURES_DIR / first_fixture)  # warm-up: triggers model load
        except Exception as exc:
            summaries.append({
                "parser": p_name,
                "status": "skipped",
                "success_rate": 0.0,
                "avg_quality_score": 0.0,
                "avg_latency_sec": 0.0,
                "throughput_docs_sec": 0.0,
                "notes": f"Model unavailable: {exc}",
            })
            continue

        times: list[float] = []
        qualities: list[float] = []
        successes = 0
        total_fixtures = 0

        for fix_name, exp in expectations.items():
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
                scores = {"text_coverage": 0.0, "segment_coverage": 0.0, "language_accuracy": 0.0, "wer": None, "quality_score": 0.0}

            results.append({
                "parser": p_name,
                "fixture": fix_name,
                "status": status,
                "latency_sec": round(elapsed, 4),
                **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in scores.items()},
            })

        avg_latency = sum(times) / len(times) if times else 0.0
        avg_quality = sum(qualities) / len(qualities) if qualities else 0.0
        throughput = (1.0 / avg_latency) if avg_latency > 0 else 0.0

        summaries.append({
            "parser": p_name,
            "status": "available",
            "success_rate": round(successes / total_fixtures, 4) if total_fixtures else 0.0,
            "avg_quality_score": round(avg_quality, 4),
            "avg_latency_sec": round(avg_latency, 4),
            "throughput_docs_sec": round(throughput, 2),
            "notes": f"model={parser.model_size}",
        })

    report = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "model_size": model_size or "tiny (default)",
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
    parser = argparse.ArgumentParser(description="Run audio transcription benchmarks")
    parser.add_argument("--parsers", nargs="+", help="Parsers to benchmark")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--no-archive", action="store_true")
    parser.add_argument("--model-size", default=None, help="faster-whisper model size (default: tiny)")
    args = parser.parse_args()

    report = run_benchmark(
        parsers=args.parsers,
        output_dir=args.output_dir,
        archive=not args.no_archive,
        model_size=args.model_size,
    )
    print("Benchmark complete. Summaries:")
    for s in report["summaries"]:
        print(f"  {s['parser']}: {s}")


if __name__ == "__main__":
    main()
