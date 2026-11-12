"""Performance benchmark of the hierarchical retrieval strategy.

The benchmark measures the *cost* side of the strategy, which the quality
evaluation does not cover:

* the indexing throughput (how many parent and child records are written per
  second);
* the query latency of the three systems -- flat, parent-only and the full
  coarse-to-fine strategy -- as p50 and p95;
* the candidate reduction the hierarchy buys, as a function of the number of
  selected parents;
* how the latency and the reduction scale with the size of the corpus.

The corpus is synthetic and deterministic, and the vectors come from a
seeded generator by default, so two runs on the same machine produce the same
numbers.  With ``--embedding qwen`` the real production encoder is used
instead and the corpus is embedded for real.

Nothing is downloaded: the Qwen adapter raises if the weights are missing.

Usage:
  python benchmarks/retrieval/hierarchical/benchmark.py
  python benchmarks/retrieval/hierarchical/benchmark.py --embedding qwen
  python benchmarks/retrieval/hierarchical/benchmark.py --json report.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.retrieval.hierarchical.evaluate import (
    FlatChildSearch,
    HierarchicalSearch,
    ParentOnlySearch,
)
from benchmarks.retrieval.hierarchical.metrics import percentile
from chunkers.hierarchical.chunker import HierarchicalChunker
from chunkers.hierarchical.chunker import HierarchicalConfig as ChunkerConfig
from embeddings.contracts.models import (
    EmbeddingInput,
    EmbeddingResult,
    Normalization,
    RepresentationType,
)
from parsers.contracts.models import BlockType, ParsedBlock, ParsedDocument
from retrieval.dense import DenseRetriever
from retrieval.strategies.hierarchical import (
    HierarchicalConfig as RetrievalConfig,
    HierarchicalIndexer,
    HierarchicalRetrievalStrategy,
)


class SeededEmbeddingModel:
    """A deterministic embedding model built on a seeded generator.

    The vectors are pseudo random but reproducible, which is what a
    performance benchmark needs: the latency must not depend on the
    semantics of the text.
    """

    name = "benchmark/seeded"

    def __init__(self, dimension: int = 128, seed: int = 20260912) -> None:
        self.dimension = dimension
        self.seed = seed

    @property
    def capabilities(self):
        from embeddings.contracts.models import EmbeddingCapabilities

        return EmbeddingCapabilities(
            dense=True,
            query_embedding=True,
            document_embedding=True,
            batch_inference=True,
            embedding_dimensions=self.dimension,
        )

    def get_model_info(self) -> dict[str, object]:
        return {
            "model_id": self.name,
            "adapter_name": "seeded",
            "dimensions": self.dimension,
            "representation_type": "dense",
            "normalization": "l2",
        }

    def get_embedding_space(self):
        from embeddings.contracts.models import EmbeddingSpaceDescriptor

        return EmbeddingSpaceDescriptor(
            model_id=self.name,
            model_revision=None,
            adapter_name="seeded",
            dimensions=self.dimension,
            representation_type=RepresentationType.DENSE,
            normalization=Normalization.L2,
            supported_languages=None,
        )

    def embed_documents(self, inputs: list[EmbeddingInput]) -> list[EmbeddingResult]:
        return [self._result(item, offset=0) for item in inputs]

    def embed_query(self, input: EmbeddingInput) -> EmbeddingResult:
        return self._result(input, offset=1)

    def embed_queries(self, inputs: list[EmbeddingInput]) -> list[EmbeddingResult]:
        return [self._result(item, offset=1) for item in inputs]

    def _result(self, item: EmbeddingInput, *, offset: int) -> EmbeddingResult:
        import random

        digest = hash((self.seed, offset, item.text or ""))
        rng = random.Random(digest)
        vector = [rng.uniform(-1.0, 1.0) for _ in range(self.dimension)]
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return EmbeddingResult(
            model_id=self.name,
            dimensions=self.dimension,
            dense_vector=[value / norm for value in vector],
            representation_type=RepresentationType.DENSE,
            normalization=Normalization.L2,
            metadata={"adapter_name": "seeded"},
            doc_id=item.doc_id,
            chunk_id=item.chunk_id,
        )


@dataclass
class Latency:
    """The latency percentiles of a set of measurements."""

    p50: float | None = None
    p95: float | None = None
    mean: float | None = None

    @classmethod
    def of(cls, values: list[float]) -> Latency:
        if not values:
            return cls()
        return cls(
            p50=percentile(values, 0.50),
            p95=percentile(values, 0.95),
            mean=statistics.fmean(values),
        )

    def as_dict(self) -> dict[str, float | None]:
        return {"p50": self.p50, "p95": self.p95, "mean": self.mean}


def synthetic_document(index: int, sections: int = 6, paragraphs: int = 3) -> ParsedDocument:
    """Build a deterministic synthetic document."""
    blocks: list[ParsedBlock] = []
    for section in range(sections):
        blocks.append(
            ParsedBlock(
                BlockType.HEADING,
                f"Section {section} of document {index}",
                metadata={"heading_level": 1},
            )
        )
        for paragraph in range(paragraphs):
            words = [
                f"w{index}-{section}-{paragraph}-{index * 7 + section * 13 + paragraph * 3 + token}"
                for token in range(120)
            ]
            blocks.append(BlockType.TEXT and ParsedBlock(BlockType.TEXT, " ".join(words)))
    return ParsedDocument(
        source=f"synthetic-{index:04d}.md",
        parser_name="benchmark",
        parser_version="1.0",
        blocks=blocks,
    )


def index_synthetic(
    store,
    model,
    *,
    collection: str,
    documents: int,
    sections: int,
    paragraphs: int,
) -> tuple[float, int, int]:
    """Index the synthetic corpus; return the seconds and the record counts."""
    indexer = HierarchicalIndexer(store, model, collection=collection)
    indexer.ensure_collection()
    chunker = HierarchicalChunker(
        ChunkerConfig(parent_size=1200, child_size=400, child_overlap=60)
    )
    parents = 0
    children = 0
    started = time.perf_counter()
    for index in range(documents):
        result = chunker.chunk(synthetic_document(index, sections, paragraphs))
        indexer.index_result(result)
        parents += len(result.parents)
        children += len(result.children)
    elapsed = time.perf_counter() - started
    return elapsed, parents, children


@dataclass
class BenchmarkReport:
    """The measurements of the benchmark."""

    indexing: dict[str, object] = field(default_factory=dict)
    latency: dict[str, dict[str, float | None]] = field(default_factory=dict)
    reduction: dict[str, object] = field(default_factory=dict)
    scaling: list[dict[str, object]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "indexing": self.indexing,
            "query_latency_ms": self.latency,
            "candidate_reduction": self.reduction,
            "scaling": self.scaling,
        }


def benchmark(
    store,
    model,
    *,
    collection: str = "bench",
    documents: int = 200,
    sections: int = 6,
    paragraphs: int = 3,
    queries: int = 200,
    parent_top_k: int = 8,
    child_top_k: int = 40,
    final_top_k: int = 10,
) -> BenchmarkReport:
    """Run the benchmark and return the report."""
    elapsed, parents, children = index_synthetic(
        store,
        model,
        collection=collection,
        documents=documents,
        sections=sections,
        paragraphs=paragraphs,
    )
    total = parents + children
    report = BenchmarkReport()
    report.indexing = {
        "documents": documents,
        "parents": parents,
        "children": children,
        "records": total,
        "seconds": elapsed,
        "records_per_second": (total / elapsed) if elapsed else None,
    }

    retriever = DenseRetriever(store, model, collection=collection)
    flat = FlatChildSearch(retriever)
    parent_only = ParentOnlySearch(retriever, top_k=parent_top_k)
    strategy = HierarchicalRetrievalStrategy.create(
        store,
        model,
        collection=collection,
        config=RetrievalConfig(
            parent_top_k=parent_top_k,
            child_top_k=child_top_k,
            final_top_k=final_top_k,
        ),
    )
    hierarchical = HierarchicalSearch(strategy)

    query_texts = [f"query number {index} about w{index}-2-0-{index * 7}" for index in range(queries)]

    latencies: dict[str, list[float]] = {"flat": [], "parent": [], "hierarchical": []}
    reductions: list[float] = []
    for text in query_texts:
        for name, system in (
            ("flat", flat),
            ("parent", parent_only),
            ("hierarchical", hierarchical),
        ):
            started = time.perf_counter()
            system.search(text, final_top_k)
            latencies[name].append((time.perf_counter() - started) * 1000.0)
        diagnostics = hierarchical.last_response.diagnostics
        if diagnostics.candidates_total:
            reductions.append(
                1.0 - (diagnostics.candidates_searched / diagnostics.candidates_total)
            )

    report.latency = {
        name: Latency.of(values).as_dict() for name, values in latencies.items()
    }
    report.reduction = {
        "mean": statistics.fmean(reductions) if reductions else None,
        "min": min(reductions) if reductions else None,
        "max": max(reductions) if reductions else None,
        "candidates_total": children,
        "parent_top_k": parent_top_k,
    }
    return report


def scaling(store, model, *, collection: str, sizes: list[int], **kwargs) -> list[dict[str, object]]:
    """Measure how the latency and the reduction grow with the corpus size."""
    rows: list[dict[str, object]] = []
    for size in sizes:
        name = f"{collection}-{size}"
        store.create_collection(
            HierarchicalIndexer(store, model, collection=name).collection_config(),
            exist_ok=True,
        )
        result = benchmark(store, model, collection=name, documents=size, **kwargs)
        rows.append(
            {
                "documents": size,
                "records": result.indexing["records"],
                "flat_p50_ms": result.latency["flat"]["p50"],
                "flat_p95_ms": result.latency["flat"]["p95"],
                "hierarchical_p50_ms": result.latency["hierarchical"]["p50"],
                "hierarchical_p95_ms": result.latency["hierarchical"]["p95"],
                "candidate_reduction_mean": result.reduction["mean"],
            }
        )
    return rows


def build_model(kind: str):
    """Return the embedding model for the benchmark."""
    if kind == "qwen":
        from embeddings.qwen3 import Qwen3EmbeddingModel

        return Qwen3EmbeddingModel()
    if kind == "seeded":
        return SeededEmbeddingModel()
    raise ValueError(f"unknown embedding model {kind!r}")


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and print the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--embedding",
        choices=["seeded", "qwen"],
        default="seeded",
        help="embedding model used for the corpus and the queries",
    )
    parser.add_argument("--documents", type=int, default=200)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--parent-top-k", type=int, default=8)
    parser.add_argument("--json", type=Path, default=None, help="write the report as JSON")
    parser.add_argument(
        "--scaling",
        action="store_true",
        help="also measure how the latency grows with the corpus size",
    )
    args = parser.parse_args(argv)

    __import__("qdrant_client", fromlist=["qdrant_client"])
    from vector_stores.qdrant import QdrantVectorStore

    store = QdrantVectorStore.memory()
    try:
        model = build_model(args.embedding)
        report = benchmark(
            store,
            model,
            documents=args.documents,
            queries=args.queries,
            parent_top_k=args.parent_top_k,
        )
        if args.scaling:
            report.scaling = scaling(
                store,
                model,
                collection="bench-scale",
                sizes=[50, 100, 200],
                queries=args.queries,
            )
    finally:
        store.close()

    payload = report.as_dict()
    indexing = payload["indexing"]
    print(
        f"indexing: {indexing['records']} records "
        f"({indexing['parents']} parents, {indexing['children']} children) "
        f"in {indexing['seconds']:.2f}s "
        f"({_fmt(indexing['records_per_second'])} records/s)"
    )
    print()
    header = f"{'system':<14}{'p50 ms':>10}{'p95 ms':>10}{'mean ms':>10}"
    print(header)
    print("-" * len(header))
    for name, values in payload["query_latency_ms"].items():
        print(
            f"{name:<14}{_fmt(values['p50']):>10}{_fmt(values['p95']):>10}"
            f"{_fmt(values['mean']):>10}"
        )
    reduction = payload["candidate_reduction"]
    print()
    print(
        f"candidate reduction: mean {_fmt(reduction['mean'])} "
        f"(min {_fmt(reduction['min'])}, max {_fmt(reduction['max'])}) "
        f"of {reduction['candidates_total']} child records"
    )
    if payload["scaling"]:
        print()
        print("scaling:")
        for row in payload["scaling"]:
            print(
                f"  {row['documents']:>5} documents "
                f"({row['records']} records): flat p50 {_fmt(row['flat_p50_ms'])} ms, "
                f"hierarchical p50 {_fmt(row['hierarchical_p50_ms'])} ms, "
                f"reduction {_fmt(row['candidate_reduction_mean'])}"
            )

    if args.json:
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nreport written to {args.json}")
    return 0


def _fmt(value: float | None) -> str:
    """Format a number for the report."""
    if value is None:
        return "n/a"
    return f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
