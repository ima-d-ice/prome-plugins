"""Evaluation of the hierarchical retrieval strategy.

Three systems are compared on the same corpus, the same chunks and the same
embeddings:

``flat``
    a dense search over **every** child chunk of the corpus;
``parent``
    the coarse stage alone: a dense search over the parent sections;
``hierarchical``
    the full strategy: the coarse stage selects the parents, the fine stage
    retrieves the children *inside them*.

The comparison answers two questions:

* does the hierarchy **reduce the search space**?  (the candidate reduction
  ratio)
* does the hierarchy **keep or improve the quality** of the evidence?  (recall,
  MRR, nDCG)

The quality numbers only mean something when the two systems use the same
embeddings, so the harness embeds the corpus once and reuses the vectors.

Usage:
  python benchmarks/retrieval/hierarchical/evaluate.py
  python benchmarks/retrieval/hierarchical/evaluate.py --embedding qwen
  python benchmarks/retrieval/hierarchical/evaluate.py --json report.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.retrieval.hierarchical.corpus import CORPUS, QUERIES, Document, Query
from benchmarks.retrieval.hierarchical.metrics import (
    hit_at_k,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
    summarize,
)
from chunkers.hierarchical.chunker import HierarchicalChunker
from chunkers.hierarchical.chunker import HierarchicalConfig as ChunkerConfig
from embeddings.contracts.protocols import EmbeddingModel
from parsers.contracts.models import BlockType, ParsedBlock, ParsedDocument
from retrieval.dense import DenseRetriever
from retrieval.strategies.hierarchical import (
    HierarchicalConfig as RetrievalConfig,
    HierarchicalIndexer,
    HierarchicalRetrievalStrategy,
)
from retrieval.strategies.hierarchical.metadata import CHUNK_LEVEL_FIELD
from vector_stores.contracts.filters import eq
from vector_stores.contracts.protocols import VectorStore


# ----------------------------------------------------------------------
# corpus preparation
# ----------------------------------------------------------------------
def to_parsed_document(document: Document) -> ParsedDocument:
    """Turn a corpus document into a parsed document."""
    blocks: list[ParsedBlock] = []
    for section in document.sections:
        blocks.append(
            ParsedBlock(
                BlockType.HEADING,
                section.heading,
                metadata={"heading_level": section.level},
            )
        )
        for paragraph in section.paragraphs:
            blocks.append(ParsedBlock(BlockType.TEXT, paragraph))
    return ParsedDocument(
        source=document.source,
        parser_name="corpus",
        parser_version="1.0",
        metadata={"title": document.title},
        blocks=blocks,
    )


@dataclass
class IndexedCorpus:
    """The chunked and indexed corpus."""

    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    parent_of_child: dict[str, str] = field(default_factory=dict)
    text_of: dict[str, str] = field(default_factory=dict)
    heading_of_parent: dict[str, str] = field(default_factory=dict)
    heading_of_child: dict[str, str] = field(default_factory=dict)
    child_of_parent: dict[str, list[str]] = field(default_factory=dict)


def build_corpus(
    store: VectorStore,
    model: EmbeddingModel,
    *,
    collection: str = "eval",
    parent_size: int = 1200,
    child_size: int = 400,
    child_overlap: int = 60,
) -> IndexedCorpus:
    """Chunk, embed and index the corpus; return the chunk ids."""
    indexer = HierarchicalIndexer(store, model, collection=collection)
    indexer.ensure_collection()
    chunker = HierarchicalChunker(
        ChunkerConfig(
            parent_size=parent_size,
            child_size=child_size,
            child_overlap=child_overlap,
        )
    )
    corpus = IndexedCorpus()
    for document in CORPUS:
        result = chunker.chunk(to_parsed_document(document))
        indexer.index_result(result)
        for parent in result.parents:
            corpus.parents.append(parent.id)
            corpus.text_of[parent.id] = parent.text
            corpus.heading_of_parent[parent.id] = (
                (parent.metadata.get("heading_path") or [""])[-1]
            )
            corpus.child_of_parent[parent.id] = list(
                parent.metadata.get("child_ids") or []
            )
        for child in result.children:
            corpus.children.append(child.id)
            corpus.text_of[child.id] = child.text
            corpus.parent_of_child[child.id] = child.metadata.get("parent_id")
            corpus.heading_of_child[child.id] = (
                (child.metadata.get("heading_path") or [""])[-1]
            )
    return corpus


# ----------------------------------------------------------------------
# ground truth
# ----------------------------------------------------------------------
def resolve_ground_truth(corpus: IndexedCorpus, query: Query) -> tuple[list[str], list[str]]:
    """Resolve the labels of ``query`` to chunk ids.

    The labels are written on the text, so they are resolved after chunking.
    A relevant sentence belongs to the child chunk that contains it; the
    relevant parent is the parent of that child, which is also matched by its
    heading.
    """
    relevant_children: list[str] = []
    for sentence in query.relevant_evidence:
        for child_id in corpus.children:
            if sentence in corpus.text_of.get(child_id, ""):
                if child_id not in relevant_children:
                    relevant_children.append(child_id)
                break

    relevant_parents: list[str] = []
    for parent_id in corpus.parents:
        heading = corpus.heading_of_parent.get(parent_id, "")
        if heading in query.relevant_sections:
            relevant_parents.append(parent_id)

    # A parent that carries a relevant child is relevant even when the
    # heading of the section was split over several parents.
    for child_id in relevant_children:
        parent_id = corpus.parent_of_child.get(child_id)
        if parent_id and parent_id not in relevant_parents:
            relevant_parents.append(parent_id)
    return relevant_parents, relevant_children


# ----------------------------------------------------------------------
# the systems
# ----------------------------------------------------------------------
class FlatChildSearch:
    """A dense search over every child chunk of the corpus."""

    name = "flat"

    def __init__(self, retriever: DenseRetriever) -> None:
        self.retriever = retriever

    def search(self, query: str, limit: int) -> list[str]:
        """Return the ranked child ids."""
        vector = self.retriever.embed_query(query)
        hits = self.retriever.search(
            vector,
            limit=limit,
            filter=eq(CHUNK_LEVEL_FIELD, "child"),
        )
        return [hit.chunk_id for hit in hits]


class ParentOnlySearch:
    """The coarse stage alone: a dense search over the parent sections."""

    name = "parent"

    def __init__(self, retriever: DenseRetriever, top_k: int = 8) -> None:
        self.retriever = retriever
        self.top_k = top_k

    def search(self, query: str, limit: int) -> list[str]:
        """Return the ranked parent ids."""
        vector = self.retriever.embed_query(query)
        hits = self.retriever.search(
            vector,
            limit=limit,
            filter=eq(CHUNK_LEVEL_FIELD, "parent"),
        )
        return [hit.chunk_id for hit in hits]


class HierarchicalSearch:
    """The full coarse-to-fine strategy."""

    name = "hierarchical"

    def __init__(self, strategy: HierarchicalRetrievalStrategy) -> None:
        self.strategy = strategy
        self.last_response = None

    def search(self, query: str, limit: int) -> list[str]:
        """Return the ranked child ids."""
        response = self.strategy.retrieve(query)
        self.last_response = response
        return [result.chunk_id for result in response.results][:limit]


# ----------------------------------------------------------------------
# the evaluation
# ----------------------------------------------------------------------
@dataclass
class SystemReport:
    """The metrics of one system."""

    name: str
    metrics: dict[str, float | None]
    latencies_ms: list[float]
    candidate_reduction: list[float]

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "metrics": self.metrics,
            "latency_ms": {
                "p50": percentile(self.latencies_ms, 0.50),
                "p95": percentile(self.latencies_ms, 0.95),
                "mean": (
                    statistics.fmean(self.latencies_ms) if self.latencies_ms else None
                ),
            },
            "candidate_reduction_ratio": {
                "mean": (
                    statistics.fmean(self.candidate_reduction)
                    if self.candidate_reduction
                    else None
                ),
                "min": min(self.candidate_reduction) if self.candidate_reduction else None,
                "max": max(self.candidate_reduction) if self.candidate_reduction else None,
            },
        }


def evaluate(
    store: VectorStore,
    model: EmbeddingModel,
    *,
    collection: str = "eval",
    final_top_k: int = 10,
    parent_top_k: int = 8,
    child_top_k: int = 40,
) -> dict[str, object]:
    """Run the evaluation and return the report."""
    corpus = build_corpus(store, model, collection=collection)
    retriever = DenseRetriever(store, model, collection=collection)

    flat = FlatChildSearch(retriever)
    parent_only = ParentOnlySearch(retriever, top_k=parent_top_k)
    hierarchical = HierarchicalSearch(
        HierarchicalRetrievalStrategy.create(
            store,
            model,
            collection=collection,
            config=RetrievalConfig(
                parent_top_k=parent_top_k,
                child_top_k=child_top_k,
                final_top_k=final_top_k,
            ),
        )
    )

    systems = {
        "flat": flat,
        "parent": parent_only,
        "hierarchical": hierarchical,
    }

    # The flat and the hierarchical system both answer with child ids; the
    # parent-only system answers with parent ids.
    answer_level = {
        "flat": "child",
        "parent": "parent",
        "hierarchical": "child",
    }

    collected: dict[str, dict[str, list[float | None]]] = {
        name: {
            "recall@1": [],
            "recall@3": [],
            "recall@5": [],
            "hit@1": [],
            "hit@3": [],
            "hit@5": [],
            "mrr": [],
            "ndcg@5": [],
            "ndcg@10": [],
        }
        for name in systems
    }
    latencies: dict[str, list[float]] = {name: [] for name in systems}
    reductions: dict[str, list[float]] = {name: [] for name in systems}
    parent_recall: list[float | None] = []
    child_recall: list[float | None] = []
    unresolved: list[str] = []

    for query in QUERIES:
        relevant_parents, relevant_children = resolve_ground_truth(corpus, query)
        if not relevant_children:
            unresolved.append(query.text)
            continue

        for name, system in systems.items():
            started = time.perf_counter()
            ranked = system.search(query.text, final_top_k)
            elapsed = (time.perf_counter() - started) * 1000.0
            latencies[name].append(elapsed)

            if answer_level[name] == "parent":
                relevant = relevant_parents
            else:
                relevant = relevant_children
            bucket = collected[name]
            bucket["recall@1"].append(recall_at_k(ranked, relevant, 1))
            bucket["recall@3"].append(recall_at_k(ranked, relevant, 3))
            bucket["recall@5"].append(recall_at_k(ranked, relevant, 5))
            bucket["hit@1"].append(hit_at_k(ranked, relevant, 1))
            bucket["hit@3"].append(hit_at_k(ranked, relevant, 3))
            bucket["hit@5"].append(hit_at_k(ranked, relevant, 5))
            bucket["mrr"].append(reciprocal_rank(ranked, relevant))
            bucket["ndcg@5"].append(ndcg_at_k(ranked, relevant, 5))
            bucket["ndcg@10"].append(ndcg_at_k(ranked, relevant, 10))

            if name == "hierarchical":
                # The reduction is measured on the fine stage: how many of
                # the child chunks of the corpus were actually candidates.
                diagnostics = hierarchical.last_response.diagnostics
                searched = diagnostics.candidates_searched
                total = diagnostics.candidates_total
                reductions[name].append(
                    1.0 - (searched / total) if total else 0.0
                )
            elif name == "flat":
                # The flat search considers every child chunk of the corpus,
                # so it never skips a candidate.
                reductions[name].append(0.0)
            else:
                # The coarse stage searches the parent sections only; the
                # ratio is relative to the parents of the corpus.
                total_parents = len(corpus.parents)
                reductions[name].append(
                    1.0 - (len(ranked) / total_parents) if total_parents else 0.0
                )

        # The parent recall of the coarse stage is measured on the parents
        # the strategy selected.
        selected = list(hierarchical.last_response.diagnostics.selected_parent_ids)
        parent_recall.append(recall_at_k(selected, relevant_parents, parent_top_k))
        child_recall.append(
            recall_at_k(
                [result.chunk_id for result in hierarchical.last_response.results],
                relevant_children,
                5,
            )
        )

    reports: list[SystemReport] = []
    for name in systems:
        bucket = collected[name]
        metrics = {
            key: summarize(key, values).value for key, values in bucket.items()
        }
        reports.append(
            SystemReport(
                name=name,
                metrics=metrics,
                latencies_ms=latencies[name],
                candidate_reduction=reductions.get(name, []),
            )
        )

    return {
        "corpus": {
            "documents": len(CORPUS),
            "queries": len(QUERIES),
            "unresolved_queries": unresolved,
            "parents": len(corpus.parents),
            "children": len(corpus.children),
        },
        "systems": [report.as_dict() for report in reports],
        "hierarchical_diagnostics": {
            "parent_recall_at_k": summarize("parent_recall_at_k", parent_recall).value,
            "child_recall_at_5": summarize("child_recall_at_5", child_recall).value,
        },
    }


# ----------------------------------------------------------------------
def build_model(kind: str) -> EmbeddingModel:
    """Return the embedding model for the evaluation."""
    if kind == "qwen":
        from embeddings.qwen3 import Qwen3EmbeddingModel

        return Qwen3EmbeddingModel()
    if kind == "deterministic":
        from embeddings.deterministic import DeterministicEmbeddingModel

        return DeterministicEmbeddingModel(dimension=256)
    raise ValueError(f"unknown embedding model {kind!r}")


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation and print the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--embedding",
        choices=["deterministic", "qwen"],
        default="deterministic",
        help="embedding model used for the corpus and the queries",
    )
    parser.add_argument("--json", type=Path, default=None, help="write the report as JSON")
    parser.add_argument(
        "--final-top-k", type=int, default=10, help="number of results per query"
    )
    args = parser.parse_args(argv)

    qdrant = __import__("qdrant_client", fromlist=["qdrant_client"])
    from vector_stores.qdrant import QdrantVectorStore

    store = QdrantVectorStore.memory()
    try:
        model = build_model(args.embedding)
        report = evaluate(
            store, model, final_top_k=args.final_top_k
        )
    finally:
        store.close()

    corpus = report["corpus"]
    print(
        f"corpus: {corpus['documents']} documents, {corpus['parents']} parents, "
        f"{corpus['children']} children, {corpus['queries']} queries"
    )
    if corpus["unresolved_queries"]:
        print(f"WARNING: {len(corpus['unresolved_queries'])} queries could not be resolved")
    print()
    header = f"{'system':<14}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'MRR':>8}{'nDCG@10':>10}{'p50ms':>9}{'p95ms':>9}"
    print(header)
    print("-" * len(header))
    for system in report["systems"]:
        metrics = system["metrics"]
        latency = system["latency_ms"]
        print(
            f"{system['name']:<14}"
            f"{_fmt(metrics['recall@1']):>8}"
            f"{_fmt(metrics['recall@3']):>8}"
            f"{_fmt(metrics['recall@5']):>8}"
            f"{_fmt(metrics['mrr']):>8}"
            f"{_fmt(metrics['ndcg@10']):>10}"
            f"{_fmt(latency['p50']):>9}"
            f"{_fmt(latency['p95']):>9}"
        )
    diagnostics = report["hierarchical_diagnostics"]
    print()
    print(
        f"hierarchical: parent recall@{8} = {_fmt(diagnostics['parent_recall_at_k'])}, "
        f"child recall@5 = {_fmt(diagnostics['child_recall_at_5'])}"
    )
    for system in report["systems"]:
        reduction = system["candidate_reduction_ratio"]
        if reduction["mean"] is not None:
            print(
                f"candidate reduction ({system['name']}): "
                f"mean {_fmt(reduction['mean'])}"
            )

    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nreport written to {args.json}")
    return 0


def _fmt(value: float | None) -> str:
    """Format a metric for the report."""
    if value is None:
        return "n/a"
    return f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
