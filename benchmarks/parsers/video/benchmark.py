"""Reproducible PyAV/ffmpeg video parser benchmark for Prome RAG Plugins.

Metrics per fixture (interval strategy, interval=1.0s, max_frames=10):
  metadata_accuracy  fraction of expected container/stream fields matched (duration ±0.2s, fps ±0.5)
  frames_extracted   number of frames selected
  quality_score      metadata_accuracy (the primary signal for video)

For scene_cut.mp4, also verifies the red->blue hard cut at ~1.5s is detected
with the scene strategy (reported as separate fixture entry).
"""

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
    "ffmpeg_video": ("parsers.video.ffmpeg", "FFmpegVideoParser", ("av", "numpy")),
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


def instantiate_parser(name: str, **kwargs):
    module_path, class_name, _ = PARSER_BUILDERS[name]
    mod = __import__(module_path, fromlist=[class_name])
    return getattr(mod, class_name)(**kwargs)


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


def score_fixture(doc, exp: dict, strategy: str) -> dict[str, float]:
    meta = doc.metadata
    checks = []

    def check(cond: bool) -> None:
        checks.append(1.0 if cond else 0.0)

    if exp.get("expected_codec") is not None:
        check(meta.get("video_codec") == exp["expected_codec"])
    if exp.get("expected_width") is not None:
        check(meta.get("width") == exp["expected_width"])
    if exp.get("expected_height") is not None:
        check(meta.get("height") == exp["expected_height"])
    if exp.get("expected_fps") is not None:
        check(abs((meta.get("frame_rate") or 0) - exp["expected_fps"]) <= 0.5)
    if exp.get("expected_duration") is not None:
        check(abs((meta.get("duration_seconds") or 0) - exp["expected_duration"]) <= 0.25)
    if exp.get("min_audio_streams") is not None:
        check(meta.get("audio_stream_count", 0) >= exp["min_audio_streams"])

    metadata_accuracy = sum(checks) / len(checks) if checks else 1.0

    if strategy == "interval":
        expected = exp.get("expected_frames_interval")
        frames_ok = len(doc.blocks) == expected if expected is not None else 1.0
    else:  # scene
        cut_ts = exp.get("scene_cut_timestamp")
        frames_ok = any(abs(b.metadata["timestamp"] - cut_ts) < 0.15 for b in doc.blocks) if cut_ts is not None else 1.0

    quality = round(0.7 * metadata_accuracy + 0.3 * (1.0 if frames_ok else 0.0), 4)
    return {
        "metadata_accuracy": round(metadata_accuracy, 4),
        "frames_extracted": float(len(doc.blocks)),
        "frames_ok": 1.0 if frames_ok else 0.0,
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
            summaries.append({
                "parser": p_name,
                "status": "skipped",
                "success_rate": 0.0,
                "avg_quality_score": 0.0,
                "avg_latency_sec": 0.0,
                "throughput_docs_sec": 0.0,
                "notes": "Dependencies not installed (pip install 'prome-rag-plugins[video]')",
            })
            continue

        times: list[float] = []
        qualities: list[float] = []
        successes = 0
        total_fixtures = 0

        # Each fixture is parsed with its relevant strategy. testsrc uses the
        # interval strategy; scene_cut is evaluated in both modes.
        jobs = [("testsrc.mp4", "interval"), ("scene_cut.mp4", "interval"), ("scene_cut.mp4", "scene")]

        for fix_name, strategy in jobs:
            exp = expectations.get(fix_name)
            fix_path = FIXTURES_DIR / fix_name
            if exp is None or not fix_path.exists():
                continue

            kwargs = {"frame_strategy": strategy, "interval_seconds": 1.0, "max_frames": 10}
            if strategy == "scene":
                kwargs["scene_threshold"] = 0.08
            parser = instantiate_parser(p_name, **kwargs)

            total_fixtures += 1
            label = fix_name if strategy == "interval" else f"{fix_name}[scene]"
            t0 = time.perf_counter()
            try:
                doc = parser.parse(fix_path)
                elapsed = time.perf_counter() - t0
                scores = score_fixture(doc, exp, strategy)
                status = "ok"
                successes += 1
                qualities.append(scores["quality_score"])
                times.append(elapsed)
            except Exception:
                elapsed = time.perf_counter() - t0
                status = "failed"
                scores = {"metadata_accuracy": 0.0, "frames_extracted": 0.0, "frames_ok": 0.0, "quality_score": 0.0}

            results.append({
                "parser": p_name,
                "fixture": label,
                "status": status,
                "latency_sec": round(elapsed, 4),
                **{k: round(v, 4) for k, v in scores.items()},
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
            "notes": "interval(1s)+scene strategies",
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
    parser = argparse.ArgumentParser(description="Run video parser benchmarks")
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
        print(f"  {s['parser']}: {s}")


if __name__ == "__main__":
    main()
