"""Reproducible vector-store benchmark for Prome RAG Plugins.

Measures the vector-store adapters on a deterministic synthetic corpus.  No
embedding model is downloaded and no inference is run: the vectors come from
a seeded pseudo random generator, so two runs on the same machine produce
the same corpus - and the *same* corpus for every backend, which makes the
numbers comparable.

Metrics (reported separately, never combined into a single score):

  * search latency for the dense exact path (``params={"exact": True}``),
    the dense approximate path (the native ANN index of the backend), the
    sparse path (when the backend has one) and the filtered dense path:
    mean, p50 and p95 per query, in milliseconds;
  * recall@k of the approximate path against the exact baseline (the
    fraction of the exact top-k that the approximate top-k contains);
  * batch upsert throughput (records per second), the index build time and
    the on-disk footprint of the collection.

Backends are *not* interchangeable:

  * ``qdrant``   - embedded local, in-memory or a running server;
  * ``lancedb``  - embedded local (a directory) or an ephemeral directory;
  * ``faiss``    - embedded local: a FAISS index plus a SQLite database;
  * ``pgvector`` - a PostgreSQL server (needs ``PROMERAG_PGVECTOR_TEST_DSN``).

A scenario that a backend does not support is reported as ``null`` with a
reason, never as a fake zero.

Usage:
  python benchmarks/vector_stores/benchmark.py --backend qdrant --mode memory
  python benchmarks/vector_stores/benchmark.py --backend lancedb --records 5000
  python benchmarks/vector_stores/benchmark.py --backend faiss --records 5000
  python benchmarks/vector_stores/benchmark.py --backend pgvector \\
      --dsn postgresql://postgres:prome@localhost:55432/prome
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import shutil
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from chunkers.contracts.models import BlockReference, Chunk
from embeddings.contracts.models import (
    EmbeddingSpaceDescriptor,
    Normalization,
    RepresentationType,
)
from vector_stores.contracts import (
    CollectionConfig,
    DistanceMetric,
    IndexConfig,
    IndexKind,
    PayloadFieldType,
    PayloadIndexSpec,
    SearchRequest,
    SpaceVectors,
    StoredRecord,
    VectorSpaceConfig,
    eq,
)

MODEL_ID = "benchmark/synthetic-embedding"
MODEL_REVISION = "bench-1"
DIMENSIONS = 128
SPARSE_TERMS = 24
MULTI_ROWS = 8
TENANTS = ("acme", "globex", "initech")
CATEGORIES = ("docs", "code", "logs", "papers")


# ----------------------------------------------------------------------
# deterministic corpus (identical for every backend)
# ----------------------------------------------------------------------
def unit_vector(rng: random.Random) -> list[float]:
    values = [rng.gauss(0.0, 1.0) for _ in range(DIMENSIONS)]
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def sparse_vector(rng: random.Random) -> tuple[list[int], list[float]]:
    indices = sorted(rng.sample(range(4096), SPARSE_TERMS))
    values = [round(rng.uniform(0.05, 1.0), 4) for _ in indices]
    return indices, values


def multi_matrix(rng: random.Random) -> list[list[float]]:
    return [unit_vector(rng) for _ in range(MULTI_ROWS)]


def build_records(count: int, seed: int) -> list[StoredRecord]:
    """Build ``count`` deterministic records."""
    rng = random.Random(seed)
    dense_space = EmbeddingSpaceDescriptor(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        dimensions=DIMENSIONS,
        representation_type=RepresentationType.DENSE,
        normalization=Normalization.L2,
    )
    sparse_desc = EmbeddingSpaceDescriptor(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        dimensions=None,
        representation_type=RepresentationType.SPARSE,
        normalization=Normalization.L2,
    )
    multi_desc = EmbeddingSpaceDescriptor(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        dimensions=DIMENSIONS,
        representation_type=RepresentationType.MULTI_VECTOR,
        normalization=Normalization.L2,
    )
    records: list[StoredRecord] = []
    for index in range(count):
        doc_id = f"doc-{index // 8:05d}.txt"
        chunk_id = f"bench{seed:04d}-fixed-{index:06d}"
        text = f"synthetic chunk {index} of {count}"
        chunk = Chunk(
            id=chunk_id,
            doc_id=doc_id,
            text=text,
            index=index % 8,
            count=8,
            length=len(text),
            length_unit="characters",
            block_refs=[BlockReference(index % 4, 1 + index % 20, "text", 0, len(text))],
            heading=f"section {index % 4}",
            metadata={
                "source": doc_id,
                "pages": [1 + index % 20],
                "strategy": "fixed",
            },
        )
        sparse_indices, sparse_values = sparse_vector(rng)
        records.append(
            StoredRecord(
                chunk=chunk,
                vectors={
                    "dense": SpaceVectors(
                        space="dense",
                        embedding_space=dense_space,
                        dense=unit_vector(rng),
                    ),
                    "sparse": SpaceVectors(
                        space="sparse",
                        embedding_space=sparse_desc,
                        sparse_indices=sparse_indices,
                        sparse_values=sparse_values,
                    ),
                    "late": SpaceVectors(
                        space="late",
                        embedding_space=multi_desc,
                        multi_vector=multi_matrix(rng),
                    ),
                },
                metadata={
                    "tenant": TENANTS[index % len(TENANTS)],
                    "category": CATEGORIES[index % len(CATEGORIES)],
                    "score": float(index % 1000) / 10.0,
                    "index": index,
                },
            )
        )
    return records


def restrict_records(records: list[StoredRecord], spaces: dict[str, str]) -> list[StoredRecord]:
    """Keep only the vectors of the spaces the backend supports."""
    supported = set(spaces)
    for record in records:
        record.vectors = {
            name: vectors
            for name, vectors in record.vectors.items()
            if name in supported
        }
    return records


def build_config(name: str, spaces: dict[str, str]) -> CollectionConfig:
    """Build the collection configuration of the requested vector spaces.

    ``spaces`` maps a space name to its kind (``dense``, ``sparse`` or
    ``multi-vector``).
    """
    vector_spaces: dict[str, VectorSpaceConfig] = {}
    for space_name, kind in spaces.items():
        if kind == "dense":
            vector_spaces[space_name] = VectorSpaceConfig(
                name=space_name,
                vector_type=RepresentationType.DENSE,
                embedding_space=EmbeddingSpaceDescriptor(
                    model_id=MODEL_ID,
                    model_revision=MODEL_REVISION,
                    dimensions=DIMENSIONS,
                    representation_type=RepresentationType.DENSE,
                    normalization=Normalization.L2,
                ),
                dimensions=DIMENSIONS,
                distance=DistanceMetric.COSINE,
            )
        elif kind == "sparse":
            vector_spaces[space_name] = VectorSpaceConfig(
                name=space_name,
                vector_type=RepresentationType.SPARSE,
                embedding_space=EmbeddingSpaceDescriptor(
                    model_id=MODEL_ID,
                    model_revision=MODEL_REVISION,
                    dimensions=None,
                    representation_type=RepresentationType.SPARSE,
                    normalization=Normalization.L2,
                ),
            )
        else:
            vector_spaces[space_name] = VectorSpaceConfig(
                name=space_name,
                vector_type=RepresentationType.MULTI_VECTOR,
                embedding_space=EmbeddingSpaceDescriptor(
                    model_id=MODEL_ID,
                    model_revision=MODEL_REVISION,
                    dimensions=DIMENSIONS,
                    representation_type=RepresentationType.MULTI_VECTOR,
                    normalization=Normalization.L2,
                ),
                dimensions=DIMENSIONS,
                distance=DistanceMetric.COSINE,
            )
    return CollectionConfig(
        name=name,
        description="synthetic vector-store benchmark collection",
        vector_spaces=vector_spaces,
        payload_indexes=(
            PayloadIndexSpec("tenant", PayloadFieldType.KEYWORD),
            PayloadIndexSpec("category", PayloadFieldType.KEYWORD),
            PayloadIndexSpec("score", PayloadFieldType.FLOAT),
            PayloadIndexSpec("index", PayloadFieldType.INTEGER),
        ),
    )


# ----------------------------------------------------------------------
# measurement helpers
# ----------------------------------------------------------------------
def percentile(values: list[float], fraction: float) -> float:
    """Return the ``fraction`` percentile of ``values`` (nearest rank)."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def timed(fn, repeats: int) -> tuple[list[float], list]:
    """Run ``fn`` ``repeats`` times and return the durations and the last result."""
    durations: list[float] = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        durations.append((time.perf_counter() - start) * 1000.0)
    return durations, result


def summarize(name: str, durations: list[float], extra: dict | None = None) -> dict:
    row = {
        "scenario": name,
        "queries": len(durations),
        "latency_mean_ms": round(statistics.fmean(durations), 4) if durations else None,
        "latency_p50_ms": round(percentile(durations, 0.50), 4) if durations else None,
        "latency_p95_ms": round(percentile(durations, 0.95), 4) if durations else None,
        "latency_min_ms": round(min(durations), 4) if durations else None,
        "latency_max_ms": round(max(durations), 4) if durations else None,
    }
    if extra:
        row.update(extra)
    return row


def recall_at_k(exact: list[str], approximate: list[str]) -> float:
    """Return the fraction of the exact top-k present in the approximate top-k."""
    if not exact:
        return float("nan")
    found = len(set(exact) & set(approximate))
    return found / len(exact)


def directory_size(path: Path) -> int:
    """Return the size of a directory in bytes."""
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


# ----------------------------------------------------------------------
# backends
# ----------------------------------------------------------------------
class Backend:
    """One benchmarkable backend."""

    name = "backend"
    mode = "local"

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.store = None
        self.data_path: Path | None = None

    def open(self) -> None:  # pragma: no cover - implemented by the subclasses
        raise NotImplementedError

    def close(self) -> None:
        if self.store is not None:
            self.store.close()
            self.store = None

    def spaces(self) -> dict[str, str]:
        """Return the vector spaces the backend supports."""
        capabilities = self.store.capabilities
        spaces = {"dense": "dense"}
        if capabilities.sparse_search:
            spaces["sparse"] = "sparse"
        if capabilities.multi_vector_search:
            spaces["late"] = "multi-vector"
        return spaces

    def build_ann_index(self, collection: str) -> tuple[bool, float, str]:
        """Build the native approximate index. Returns (built, seconds, note)."""
        capabilities = self.store.capabilities
        if not capabilities.supports(IndexKind.HNSW):
            return False, 0.0, "the backend has no approximate dense index"
        start = time.perf_counter()
        try:
            self.store.create_index(
                collection,
                IndexConfig(
                    kind=IndexKind.HNSW,
                    target="dense",
                    build_params={"num_partitions": 16, "m": 16, "ef_construction": 100},
                ),
            )
        except Exception as exc:  # pragma: no cover - depends on the backend
            return False, 0.0, f"the index could not be built: {exc}"
        return True, time.perf_counter() - start, ""

    def footprint_bytes(self, collection: str) -> int | None:
        """Return the on-disk footprint of the collection in bytes."""
        if self.data_path is None:
            return None
        return directory_size(self.data_path)


class QdrantBackend(Backend):
    name = "qdrant"

    def open(self) -> None:
        from vector_stores.qdrant import (
            QdrantMode,
            QdrantStoreConfig,
            QdrantVectorStore,
        )

        if self.args.mode == "server":
            config = QdrantStoreConfig(mode=QdrantMode.SERVER, url=self.args.url)
        elif self.args.mode == "local":
            self.data_path = Path(self.args.path)
            config = QdrantStoreConfig(mode=QdrantMode.LOCAL, path=str(self.data_path))
        else:
            config = QdrantStoreConfig.memory()
        self.store = QdrantVectorStore(config)


class LanceDBBackend(Backend):
    name = "lancedb"

    def open(self) -> None:
        from vector_stores.lancedb import LanceDBStoreConfig, LanceDBVectorStore

        if self.args.mode == "memory":
            config = LanceDBStoreConfig.memory()
        else:
            self.data_path = Path(self.args.path)
            config = LanceDBStoreConfig.local(str(self.data_path))
        self.store = LanceDBVectorStore(config)


class FaissBackend(Backend):
    name = "faiss"

    def open(self) -> None:
        from vector_stores.faiss import FaissStoreConfig, FaissVectorStore

        if self.args.mode == "memory":
            config = FaissStoreConfig.memory()
        else:
            self.data_path = Path(self.args.path)
            config = FaissStoreConfig.local(str(self.data_path))
        self.store = FaissVectorStore(config)


class PgVectorBackend(Backend):
    name = "pgvector"
    mode = "server"

    def open(self) -> None:
        from vector_stores.pgvector import PgVectorStoreConfig, PgVectorVectorStore

        self.store = PgVectorVectorStore(PgVectorStoreConfig(dsn=self.args.dsn))

    def footprint_bytes(self, collection: str) -> int | None:
        try:
            with self.store._connection.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_total_relation_size(%s::regclass)",
                    (collection,),
                )
                row = cursor.fetchone()
            self.store._connection.commit()
            return int(row[0]) if row else None
        except Exception:
            return None


BACKENDS = {
    "qdrant": QdrantBackend,
    "lancedb": LanceDBBackend,
    "faiss": FaissBackend,
    "pgvector": PgVectorBackend,
}


# ----------------------------------------------------------------------
# benchmark
# ----------------------------------------------------------------------
def run(args: argparse.Namespace) -> dict:
    backend = BACKENDS[args.backend](args)
    collection = f"bench_{args.backend.replace('-', '_')}_{args.records}"
    skipped: list[str] = []
    notes: list[str] = [
        "All vectors are synthetic and deterministic: no model is downloaded and no inference is run.",
        "The same corpus is used for every backend, so the numbers are comparable.",
        "Latency and recall are reported separately; there is no combined score.",
        "Numbers describe this synthetic corpus on this machine; they are not production baselines.",
    ]
    backend.open()
    try:
        spaces = backend.spaces()
        store = backend.store
        store.delete_collection(collection)
        config = build_config(collection, spaces)
        store.create_collection(config)

        records = restrict_records(build_records(args.records, args.seed), spaces)
        batch = args.batch_size
        upsert_seconds = 0.0
        written = 0
        for start in range(0, len(records), batch):
            chunk = records[start : start + batch]
            begin = time.perf_counter()
            store.upsert(collection, chunk)
            upsert_seconds += time.perf_counter() - begin
            written += len(chunk)

        index_built, index_seconds, index_note = backend.build_ann_index(collection)
        if not index_built and index_note:
            notes.append(f"approximate index: {index_note}")

        rng = random.Random(args.seed + 1)
        queries = [unit_vector(rng) for _ in range(args.queries)]
        sparse_queries = [sparse_vector(rng) for _ in range(args.queries)]
        multi_queries = [multi_matrix(rng) for _ in range(args.queries)]

        rows: list[dict] = []

        # 1. exact dense search (baseline)
        exact_latencies: list[float] = []
        exact_results: list[list[str]] = []
        for vector in queries:
            durations, hits = timed(
                lambda v=vector: store.search(
                    SearchRequest(
                        collection=collection,
                        space="dense",
                        dense=v,
                        limit=args.top_k,
                        params={"exact": True},
                    )
                ),
                args.repeats,
            )
            exact_latencies.extend(durations)
            exact_results.append([hit.chunk_id for hit in hits])
        rows.append(
            summarize(
                "dense_exact",
                exact_latencies,
                {"metric": "cosine", "score_direction": "higher-is-better"},
            )
        )

        # 2. approximate dense search (the native ANN index of the backend)
        approximate_latencies: list[float] = []
        recalls: list[float] = []
        for index, vector in enumerate(queries):
            durations, hits = timed(
                lambda v=vector: store.search(
                    SearchRequest(
                        collection=collection,
                        space="dense",
                        dense=v,
                        limit=args.top_k,
                        params={"hnsw_ef": args.hnsw_ef},
                    )
                ),
                args.repeats,
            )
            approximate_latencies.extend(durations)
            recalls.append(recall_at_k(exact_results[index], [hit.chunk_id for hit in hits]))
        rows.append(
            summarize(
                "dense_approximate",
                approximate_latencies,
                {
                    "recall_at_k": round(statistics.fmean(recalls), 4) if recalls else None,
                    "recall_at_k_min": round(min(recalls), 4) if recalls else None,
                    "top_k": args.top_k,
                    "index_built": index_built,
                },
            )
        )

        # 3. sparse search (only when the backend has one)
        if "sparse" in spaces:
            sparse_latencies: list[float] = []
            for indices, values in sparse_queries:
                durations, _ = timed(
                    lambda i=indices, v=values: store.search(
                        SearchRequest(
                            collection=collection,
                            space="sparse",
                            sparse_indices=i,
                            sparse_values=v,
                            limit=args.top_k,
                        )
                    ),
                    args.repeats,
                )
                sparse_latencies.extend(durations)
            rows.append(summarize("sparse", sparse_latencies))
        else:
            rows.append(summarize("sparse", [], {"unsupported": "no sparse search"}))
            skipped.append("sparse")

        # 4. multi-vector (late interaction) search
        if "late" in spaces:
            multi_latencies: list[float] = []
            for matrix in multi_queries:
                durations, _ = timed(
                    lambda m=matrix: store.search(
                        SearchRequest(
                            collection=collection,
                            space="late",
                            multi_vector=m,
                            limit=args.top_k,
                        )
                    ),
                    args.repeats,
                )
                multi_latencies.extend(durations)
            rows.append(summarize("multi_vector", multi_latencies))
        else:
            rows.append(
                summarize("multi_vector", [], {"unsupported": "no multi-vector search"})
            )
            skipped.append("multi_vector")

        # 5. dense search with a metadata filter
        filtered_latencies: list[float] = []
        for vector in queries:
            durations, _ = timed(
                lambda v=vector: store.search(
                    SearchRequest(
                        collection=collection,
                        space="dense",
                        dense=v,
                        limit=args.top_k,
                        filter=eq("tenant", TENANTS[0]),
                    )
                ),
                args.repeats,
            )
            filtered_latencies.extend(durations)
        rows.append(summarize("dense_filtered", filtered_latencies))

        # 6. upsert throughput, index build time and footprint
        rows.append(
            {
                "scenario": "batch_upsert",
                "queries": None,
                "records": written,
                "batch_size": batch,
                "seconds": round(upsert_seconds, 4),
                "records_per_second": round(written / upsert_seconds, 2)
                if upsert_seconds > 0
                else None,
                "latency_mean_ms": None,
                "latency_p50_ms": None,
                "latency_p95_ms": None,
                "latency_min_ms": None,
                "latency_max_ms": None,
            }
        )
        rows.append(
            {
                "scenario": "index_build",
                "queries": None,
                "records": None,
                "batch_size": None,
                "seconds": round(index_seconds, 4) if index_built else None,
                "records_per_second": None,
                "latency_mean_ms": None,
                "latency_p50_ms": None,
                "latency_p95_ms": None,
                "latency_min_ms": None,
                "latency_max_ms": None,
                "unsupported": None if index_built else index_note,
            }
        )
        footprint = backend.footprint_bytes(collection)
        rows.append(
            {
                "scenario": "storage_footprint",
                "queries": None,
                "records": None,
                "batch_size": None,
                "seconds": None,
                "records_per_second": None,
                "latency_mean_ms": None,
                "latency_p50_ms": None,
                "latency_p95_ms": None,
                "latency_min_ms": None,
                "latency_max_ms": None,
                "bytes": footprint,
                "bytes_per_vector": round(footprint / written, 2)
                if footprint and written
                else None,
            }
        )

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "backend": backend.name,
            "mode": backend.mode if backend.name == "pgvector" else args.mode,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "configuration": {
                "records": args.records,
                "queries": args.queries,
                "repeats": args.repeats,
                "top_k": args.top_k,
                "dimensions": DIMENSIONS,
                "sparse_terms": SPARSE_TERMS,
                "multi_rows": MULTI_ROWS,
                "batch_size": args.batch_size,
                "hnsw_ef": args.hnsw_ef,
                "seed": args.seed,
                "vector_spaces": spaces,
            },
            "notes": notes,
            "skipped": skipped,
            "results": rows,
        }
        return report
    finally:
        backend.close()


RESULT_FIELDS = [
    "scenario",
    "queries",
    "records",
    "batch_size",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_min_ms",
    "latency_max_ms",
    "recall_at_k",
    "recall_at_k_min",
    "top_k",
    "seconds",
    "records_per_second",
    "bytes",
    "bytes_per_vector",
    "unsupported",
]


def archive_previous(output_dir: Path) -> list[str]:
    """Move the previous reports into ``reports/archive``."""
    archive = output_dir / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived: list[str] = []
    for path in sorted(output_dir.glob("benchmark_*")):
        if path.is_dir():
            continue
        target = archive / f"{path.stem}-{stamp}{path.suffix}"
        shutil.copy2(path, target)
        archived.append(str(target.relative_to(output_dir)))
    return archived


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--backend",
        choices=sorted(BACKENDS),
        default="qdrant",
        help="the backend to measure",
    )
    parser.add_argument(
        "--mode",
        choices=("memory", "local", "server"),
        default="memory",
        help="how the backend stores the data (default: memory)",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:6333",
        help="URL of the Qdrant server (--backend qdrant --mode server)",
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get(
            "PROMERAG_PGVECTOR_TEST_DSN", "postgresql://postgres:prome@localhost:55432/prome"
        ),
        help="DSN of the PostgreSQL server (--backend pgvector)",
    )
    parser.add_argument(
        "--path",
        default="benchmarks/vector_stores/.data",
        help="directory of the local mode",
    )
    parser.add_argument("--records", type=int, default=2000)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hnsw-ef", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20240517)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="directory of the report (default: benchmarks/vector_stores/reports/<backend>)",
    )
    parser.add_argument("--no-archive", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(
        args.output_dir
        or Path(__file__).resolve().parent / "reports" / args.backend
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    report = run(args)
    archived: list[str] = []
    if not args.no_archive:
        archived = archive_previous(output_dir)
    report["archived_previous"] = archived

    json_path = output_dir / "benchmark_report.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    write_csv(output_dir / "benchmark_results.csv", report["results"], RESULT_FIELDS)

    print(
        f"\nBackend: {report['backend']} | mode: {report['mode']} | "
        f"records: {args.records} | queries: {args.queries}"
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {output_dir / 'benchmark_results.csv'}")
    if archived:
        print(f"Archived previous reports: {', '.join(archived)}")
    for row in report["results"]:
        if row["scenario"] == "batch_upsert":
            print(
                f"  {row['scenario']}: {row['records']} records in {row['seconds']}s "
                f"({row['records_per_second']} records/s)"
            )
            continue
        if row["scenario"] in ("index_build", "storage_footprint"):
            print(f"  {row['scenario']}: {row.get('seconds') or row.get('bytes')}")
            continue
        line = (
            f"  {row['scenario']}: p50={row['latency_p50_ms']}ms "
            f"p95={row['latency_p95_ms']}ms mean={row['latency_mean_ms']}ms"
        )
        if row.get("recall_at_k") is not None:
            line += f" recall@{row['top_k']}={row['recall_at_k']}"
        if row.get("unsupported"):
            line += f" (unsupported: {row['unsupported']})"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
