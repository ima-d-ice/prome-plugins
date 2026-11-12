"""Performance benchmark of the hybrid retrieval engine.

Measures the cost side of the engine:

* the indexing throughput of the dual-index coordinator (dense + lexical);
* the query latency of the three systems — dense alone, BM25 alone and the
  hybrid combination — as p50 and p95;
* (``--scaling``) how the latency grows with the corpus size.

The corpus is synthetic and the vectors come from a seeded generator, so two
runs on the same machine produce the same numbers.  With ``--embedding qwen``
the real encoder is used instead.

Usage:
  python benchmarks/retrieval/hybrid/benchmark.py
  python benchmarks/retrieval/hybrid/benchmark.py --embedding qwen --scaling
  python benchmarks/retrieval/hybrid/benchmark.py --json report.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.retrieval.hierarchical.benchmark import SeededEmbeddingModel
from chunkers.recursive.chunker import RecursiveChunker, RecursiveConfig
from parsers.contracts.models import BlockType, ParsedBlock, ParsedDocument
from retrieval.contracts.models import RetrievalQuery
from retrieval.dense.qwen import QwenDenseRetriever
from retrieval.indexing import IndexingCoordinator
from retrieval.lexical.bm25.sqlite_fts5 import BM25Retriever, LexicalRecord, SQLiteFTS5Index
from retrieval.strategies.hybrid import HybridRetrievalStrategy
from embeddings.contracts.models import RepresentationType
from vector_stores.contracts.models import (
    CollectionConfig,
    DistanceMetric,
    VectorSpaceConfig,
)
from vector_stores.qdrant import QdrantVectorStore


def synthetic_document(index: int, sections: int, paragraphs: int) -> ParsedDocument:
    """Build a deterministic synthetic document."""
    blocks: list[ParsedBlock] = []
    rng = random.Random(1000 * index)
    words = [f"w{index}-{rng.randint(0, 100)}" for _ in range(200)]
    for section in range(sections):
        blocks.append(
            ParsedBlock(
                BlockType.HEADING,
                f"Section {section} of document {index}",
                metadata={"heading_level": 1},
            )
        )
        for paragraph in range(paragraphs):
            blocks.append(
                ParsedBlock(BlockType.TEXT, " ".join(words[:rng.randint(60, 120)]))
            )
    return ParsedDocument(
        source=f"synthetic-{index:04d}.md",
        parser_name="benchmark",
        parser_version="1.0",
        blocks=blocks,
    )


def build_systems(store, model, collection: str, documents: int, sections: int, paragraphs: int):
    """Index the synthetic corpus; return the systems."""
    index_path = Path(f"/tmp/bench_hybrid_{collection}.db")
    index_path.unlink(missing_ok=True)
    index = SQLiteFTS5Index(index_path)
    index.open()

    space = VectorSpaceConfig(
        name="dense",
        vector_type=RepresentationType.DENSE,
        embedding_space=model.get_embedding_space(),
        dimensions=model.capabilities.embedding_dimensions
        if hasattr(model, "capabilities")
        else model.dimension,
        distance=DistanceMetric.COSINE,
    )
    store.create_collection(
        CollectionConfig(name=collection, vector_spaces={"dense": space}),
        exist_ok=True,
    )
    indexer = IndexingCoordinator(
        store,
        index,
        collection=collection,
        embedding_model=model,
    )

    chunker = RecursiveChunker(RecursiveConfig(chunk_size=400, overlap=80))
    all_chunks = []
    for i in range(documents):
        all_chunks.extend(chunker.chunk(synthetic_document(i, sections, paragraphs)))

    started = time.perf_counter()
    indexer.upsert(all_chunks)
    elapsed = time.perf_counter() - started

    dense = QwenDenseRetriever(store, model, collection=collection, expected_model_id=None)
    lexical = BM25Retriever(index)
    hybrid = HybridRetrievalStrategy(dense, lexical)
    return indexer, index, dense, lexical, hybrid, all_chunks, elapsed


def build_model(kind: str):
    if kind == "qwen":
        from embeddings.qwen3 import Qwen3EmbeddingModel

        return Qwen3EmbeddingModel()
    return SeededEmbeddingModel()


def measure_latency(store, model, collection, documents, sections, paragraphs, queries):
    store.delete_collection(collection)
    indexer, index, dense, lexical, hybrid, all_chunks, elapsed = build_systems(
        store, model, collection, documents, sections, paragraphs
    )
    latencies = {"dense": [], "bm25": [], "hybrid": []}
    for i in range(queries):
        query_text = f"query number {i} about section {i % sections}"
        for name, system in (("dense", dense), ("bm25", lexical), ("hybrid", hybrid)):
            t0 = time.perf_counter()
            system.retrieve(RetrievalQuery(text=query_text, limit=10))
            latencies[name].append((time.perf_counter() - t0) * 1000.0)
    for values in latencies.values():
        values.sort()
    p50 = {n: v[len(v) // 2] for n, v in latencies.items()}
    p95 = {n: v[max(0, int(0.95 * (len(v) - 1)))] for n, v in latencies.items()}
    mean = {n: sum(vals) / len(vals) for n, vals in latencies.items()}
    return {
        "indexing": {
            "documents": documents,
            "records": len(all_chunks),
            "seconds": elapsed,
            "records_per_second": len(all_chunks) / elapsed if elapsed else None,
        },
        "latency_ms": {
            n: {"p50": p50[n], "p95": p95[n], "mean": mean[n]}
            for n in ("dense", "bm25", "hybrid")
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hybrid retrieval benchmark")
    parser.add_argument("--embedding", choices=["seeded", "qwen"], default="seeded")
    parser.add_argument("--documents", type=int, default=100)
    parser.add_argument("--sections", type=int, default=8)
    parser.add_argument("--paragraphs", type=int, default=3)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--scaling", action="store_true")
    args = parser.parse_args(argv)

    model = build_model(args.embedding)

    store = QdrantVectorStore.memory()
    try:
        if args.scaling:
            rows = []
            for size in (50, 100, 200):
                result = measure_latency(
                    store, model, "bench", size, args.sections, args.paragraphs, args.queries
                )
                rows.append({
                    "documents": size,
                    "records": result["indexing"]["records"],
                    **{k: v for k, v in result["latency_ms"].items()},
                })
            print("scaling:")
            for row in rows:
                print(
                    f"  {row['documents']:>5} documents ({row['records']} records): "
                    f"dense p50 {row['dense']['p50']:.2f} ms, "
                    f"bm25 p50 {row['bm25']['p50']:.2f} ms, "
                    f"hybrid p50 {row['hybrid']['p50']:.2f} ms"
                )
            if args.json:
                args.json.write_text(json.dumps({"scaling": rows}, indent=2), encoding="utf-8")
            return 0

        result = measure_latency(
            store, model, "bench", args.documents, args.sections, args.paragraphs, args.queries
        )
        idx = result["indexing"]
        print(
            f"indexing: {idx['records']} records in {idx['seconds']:.2f}s "
            f"({idx['records_per_second']:.1f} records/s)"
        )
        print()
        header = f"{'system':<10}{'p50 ms':>9}{'p95 ms':>9}{'mean ms':>9}"
        print(header)
        print("-" * len(header))
        for name in ("dense", "bm25", "hybrid"):
            row = result["latency_ms"][name]
            print(f"{name:<10}{row['p50']:>9.2f}{row['p95']:>9.2f}{row['mean']:>9.2f}")

        if args.json:
            args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"\nreport written to {args.json}")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
