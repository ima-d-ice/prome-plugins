"""Reproducible loader benchmark for Prome RAG Plugins.

Measures wall-clock throughput of the loader ecosystem on a synthetic
corpus. Performance only; no quality score is produced and no loader is
reported as universally better.

Metrics (per loader, per fixture):
  files_per_second      loaded files / mean wall-clock seconds
  bytes_per_second      loaded bytes / mean wall-clock seconds
  time_seconds_best     wall-clock of the fastest run
  time_seconds_mean     wall-clock of the mean run

Usage:
  python benchmarks/loaders/benchmark.py
  python benchmarks/loaders/benchmark.py --repeats 3
  python benchmarks/loaders/benchmark.py --output-dir reports
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from loaders.filesystem import FileLoader, DirectoryLoader
from loaders.uploads import UploadManager, UploadedFile, UploadedFileStatus


def make_fixture_tree(root: Path, num_files: int = 30, file_bytes: int = 4096) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = bytes((i * 37 + 11) % 251 for i in range(file_bytes))
    for i in range(num_files):
        (root / f"fixture-{i:04d}.txt").write_bytes(payload)


def bench_file_loader(corpus: Path, repeats: int) -> dict:
    loader = FileLoader(permitted_roots=[corpus])
    files = sorted(corpus.glob("*.txt"))
    timings: list[float] = []
    total_bytes = 0
    for _ in range(repeats):
        start = time.perf_counter()
        for f in files:
            loader.load(f)
        timings.append(time.perf_counter() - start)
        total_bytes = sum(f.stat().st_size for f in files)
    mean = sum(timings) / len(timings)
    return {
        "loader": "file_loader",
        "files": len(files),
        "bytes": total_bytes,
        "time_seconds_best": min(timings),
        "time_seconds_mean": mean,
        "files_per_second": len(files) / mean if mean else 0.0,
        "bytes_per_second": total_bytes / mean if mean else 0.0,
    }


def bench_directory_loader(corpus: Path, repeats: int) -> dict:
    loader = DirectoryLoader(permitted_roots=[corpus], recursive=False)
    timings: list[float] = []
    total_bytes = 0
    loaded = 0
    for _ in range(repeats):
        start = time.perf_counter()
        result = loader.load(corpus)
        timings.append(time.perf_counter() - start)
        loaded = result.success_count
        total_bytes = sum(r.file_size for r in result.resources)
    mean = sum(timings) / len(timings)
    return {
        "loader": "directory_loader",
        "files": loaded,
        "bytes": total_bytes,
        "time_seconds_best": min(timings),
        "time_seconds_mean": mean,
        "files_per_second": loaded / mean if mean else 0.0,
        "bytes_per_second": total_bytes / mean if mean else 0.0,
    }


def bench_upload_storage(tmpdir: Path, repeats: int) -> dict:
    store_root = Path(tempfile.mkdtemp(prefix="loader-bench-", dir=tmpdir))
    manager = UploadManager(storage_root=store_root / "store", session_root=store_root / "sessions")
    payload = b"benchmark payload " * 256
    timings: list[float] = []
    total_bytes = 0
    loaded = 0
    for run in range(repeats):
        start = time.perf_counter()
        files = []
        for i in range(10):
            content = payload + str(run).encode() + str(i).encode()
            files.append(
                UploadedFile(
                    file_id=str(uuid.uuid4()),
                    original_filename=f"bench-{run}-{i}.txt",
                    sanitized_filename=f"bench-{run}-{i}.txt",
                    content_type="text/plain",
                    file_size=len(content),
                    content_hash=hashlib.sha256(content).hexdigest(),
                    upload_timestamp=datetime.now(timezone.utc),
                    storage_reference="",
                    status=UploadedFileStatus.RECEIVED,
                    content=content,
                )
            )
        result = manager.upload_files(conversation_id=f"bench-conv-{run}", files=files)
        timings.append(time.perf_counter() - start)
        loaded = len(result.successful)
        total_bytes = sum(f.file_size for f in result.successful)
    mean = sum(timings) / len(timings)
    return {
        "loader": "upload_storage",
        "files": loaded,
        "bytes": total_bytes,
        "time_seconds_best": min(timings),
        "time_seconds_mean": mean,
        "files_per_second": loaded / mean if mean else 0.0,
        "bytes_per_second": total_bytes / mean if mean else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Loader benchmark")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--files", type=int, default=30)
    parser.add_argument("--file-bytes", type=int, default=4096)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    tmpdir = Path(tempfile.mkdtemp(prefix="loader-bench-corpus-"))
    corpus = tmpdir / "corpus"
    make_fixture_tree(corpus, num_files=args.files, file_bytes=args.file_bytes)

    results = [
        bench_file_loader(corpus, args.repeats),
        bench_directory_loader(corpus, args.repeats),
        bench_upload_storage(tmpdir, args.repeats),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "loader_benchmark_report.json"
    report_path.write_text(json.dumps(results, indent=2))
    for row in results:
        print(
            f"{row['loader']:16s} files={row['files']:4d} "
            f"bps={row['bytes_per_second']:.0f} fps={row['files_per_second']:.1f} "
            f"mean={row['time_seconds_mean']:.4f}s"
        )
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
