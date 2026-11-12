"""Evaluation of the hybrid retrieval engine.

Three systems are compared on the same corpus, with the same chunks and the
same model:

``dense``  — the dense channel alone (Qwen3-Embedding-0.6B);
``bm25``   — the lexical channel alone (SQLite FTS5 BM25);
``hybrid`` — both channels, fused with Reciprocal Rank Fusion (k=60).

Metrics: recall at 1/3/5/10, hit rate at 1/3/5/10, MRR, nDCG at 5 and 10.

Usage:
  python benchmarks/retrieval/hybrid/evaluate.py
  python benchmarks/retrieval/hybrid/evaluate.py --embedding qwen
  python benchmarks/retrieval/hybrid/evaluate.py --json report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.retrieval.hybrid.corpus import CORPUS, QUERIES
from benchmarks.retrieval.hierarchical.metrics import (
    hit_at_k,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
    summarize,
)
from chunkers.recursive.chunker import RecursiveChunker, RecursiveConfig
from embeddings.contracts.models import EmbeddingInput, RepresentationType
from embeddings.contracts.protocols import EmbeddingModel
from parsers.contracts.models import BlockType as ParserBlockType
from parsers.contracts.models import ParsedBlock, ParsedDocument
from retrieval.contracts.models import RetrievalQuery
from retrieval.dense.qwen import QwenDenseRetriever
from retrieval.lexical.bm25.sqlite_fts5 import (
    BM25Retriever,
    LexicalRecord,
    SQLiteFTS5Index,
)
from retrieval.strategies.hybrid import HybridRetrievalStrategy
from vector_stores.contracts.models import (
    CollectionConfig,
    DistanceMetric,
    SpaceVectors,
    StoredRecord,
    VectorSpaceConfig,
)
from vector_stores.contracts.filters import eq


def to_parsed_document(doc_source: str, title: str, sections) -> ParsedDocument:
    """Turn a corpus document into a parsed document."""
    blocks: list[ParsedBlock] = []
    for heading, paragraphs in sections:
        blocks.append(
            ParsedBlock(
                ParserBlockType.HEADING, heading, metadata={"heading_level": 1}
            )
        )
        for paragraph in paragraphs:
            blocks.append(ParsedBlock(ParserBlockType.TEXT, paragraph))
    return ParsedDocument(
        source=doc_source,
        parser_name="corpus",
        parser_version="1.0",
        metadata={"title": title},
        blocks=blocks,
    )


def chunk_corpus(docs, chunker):
    """Return every chunk of the corpus."""
    chunks = []
    for doc in docs:
        parsed = to_parsed_document(doc.source, doc.title, doc.sections)
        chunks.extend(chunker.chunk(parsed))
    return chunks


def build_systems(store, model, collection, chunks):
    """Build the dense, the lexical and the hybrid system."""
    # Store the chunks in the fake store with precomputed embeddings.
    space = VectorSpaceConfig(
        name="dense",
        vector_type=RepresentationType.DENSE,
        embedding_space=model.get_embedding_space(),
        dimensions=model.capabilities.embedding_dimensions,
        distance=DistanceMetric.COSINE,
    )
    store.create_collection(
        CollectionConfig(name=collection, vector_spaces={"dense": space})
    )
    records = []
    for chunk in chunks:
        result = model.embed_documents([EmbeddingInput(text=chunk.text)])[0]
        records.append(
            StoredRecord(
                chunk=chunk,
                vectors={
                    "dense": SpaceVectors(
                        space="dense",
                        embedding_space=model.get_embedding_space(),
                        dense=list(result.dense_vector),
                    )
                },
            )
        )
    store.upsert(collection, records)

    # Write the chunks into the lexical index.
    index = SQLiteFTS5Index(":memory:")
    index.open()
    index.upsert([LexicalRecord.from_chunk(chunk) for chunk in chunks])

    dense = QwenDenseRetriever(store, model, collection=collection, expected_model_id=None)
    lexical = BM25Retriever(index)
    hybrid = HybridRetrievalStrategy(dense, lexical)
    return dense, lexical, hybrid, index


def build_model(kind: str):
    if kind == "qwen":
        from embeddings.qwen3 import Qwen3EmbeddingModel

        return Qwen3EmbeddingModel()
    if kind == "deterministic":
        from embeddings.deterministic import DeterministicEmbeddingModel

        return DeterministicEmbeddingModel(dimension=256)
    raise ValueError(f"unknown embedding model {kind!r}")


Q_METRICS = ("recall@1", "recall@3", "recall@5", "recall@10",
             "hit@1", "hit@3", "hit@5", "hit@10",
             "mrr", "ndcg@5", "ndcg@10")


def evaluate(store, model, *, collection: str = "eval", final_top_k: int = 10):
    """Run the evaluation and return the report."""
    from chunkers.recursive.chunker import RecursiveChunker, RecursiveConfig

    chunker = RecursiveChunker(RecursiveConfig(chunk_size=500, overlap=80))
    chunks = chunk_corpus(CORPUS, chunker)

    dense, lexical, hybrid, index = build_systems(store, model, collection, chunks)

    systems = {"dense": dense, "bm25": lexical, "hybrid": hybrid}
    collected = {name: {k: [] for k in Q_METRICS} for name in systems}
    latencies: dict[str, list[float]] = {name: [] for name in systems}

    for query in QUERIES:
        evidence = query["evidence"]
        # A chunk is relevant if it contains the evidence sentence.
        relevant = [c.id for c in chunks if evidence in c.text]
        if not relevant:
            print(f"WARNING: query {query['id']} has no relevant chunk")
            continue
        for name, system in systems.items():
            started = time.perf_counter()
            response = system.retrieve(RetrievalQuery(text=query["text"], limit=final_top_k))
            latencies[name].append((time.perf_counter() - started) * 1000.0)
            ranked = [r.chunk_id for r in response.results]
            bucket = collected[name]
            bucket["recall@1"].append(recall_at_k(ranked, relevant, 1))
            bucket["recall@3"].append(recall_at_k(ranked, relevant, 3))
            bucket["recall@5"].append(recall_at_k(ranked, relevant, 5))
            bucket["recall@10"].append(recall_at_k(ranked, relevant, 10))
            bucket["hit@1"].append(hit_at_k(ranked, relevant, 1))
            bucket["hit@3"].append(hit_at_k(ranked, relevant, 3))
            bucket["hit@5"].append(hit_at_k(ranked, relevant, 5))
            bucket["hit@10"].append(hit_at_k(ranked, relevant, 10))
            bucket["mrr"].append(reciprocal_rank(ranked, relevant))
            bucket["ndcg@5"].append(ndcg_at_k(ranked, relevant, 5))
            bucket["ndcg@10"].append(ndcg_at_k(ranked, relevant, 10))

    index.close()
    return {
        "corpus": {
            "documents": len(CORPUS),
            "chunks": len(chunks),
            "queries": len(QUERIES),
        },
        "systems": {
            name: {
                "metrics": {
                    k: summarize(k, values).value for k, values in bucket.items()
                },
                "latency_ms": {
                    "p50": percentile(latencies[name], 0.50),
                    "p95": percentile(latencies[name], 0.95),
                    "mean": (
                        sum(latencies[name]) / len(latencies[name])
                        if latencies[name] else None
                    ),
                },
            }
            for name, bucket in collected.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation and print the report."""
    parser = argparse.ArgumentParser(description="Hybrid retrieval evaluation")
    parser.add_argument(
        "--embedding", choices=["deterministic", "qwen"], default="deterministic"
    )
    parser.add_argument("--json", type=Path, default=None, help="write the report as JSON")
    parser.add_argument("--final-top-k", type=int, default=10)
    args = parser.parse_args(argv)

    from vector_stores.qdrant import QdrantVectorStore

    store = QdrantVectorStore.memory()
    try:
        model = build_model(args.embedding)
        report = evaluate(store, model, final_top_k=args.final_top_k)
    finally:
        store.close()

    print(
        f"corpus: {report['corpus']['documents']} documents, "
        f"{report['corpus']['chunks']} chunks, {report['corpus']['queries']} queries"
    )
    header = f"{'system':<10}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'R@10':>8}{'MRR':>8}{'p50':>9}{'p95':>9}"
    print(header)
    print("-" * len(header))
    for name, system in report["systems"].items():
        m = system["metrics"]
        lat = system["latency_ms"]
        print(
            f"{name:<10}{_fmt(m['recall@1']):>8}{_fmt(m['recall@3']):>8}"
            f"{_fmt(m['recall@5']):>8}{_fmt(m['recall@10']):>8}{_fmt(m['mrr']):>8}"
            f"{_fmt(lat['p50']):>9}{_fmt(lat['p95']):>9}"
        )

    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nreport written to {args.json}")
    return 0


def _fmt(value):
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
