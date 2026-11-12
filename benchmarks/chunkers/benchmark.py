"""Reproducible chunker benchmark for Prome RAG Plugins.

Parses a fixed set of parser fixtures with their fast default parsers and
chunks each document with the five core chunkers (fixed, recursive,
structure_aware, hierarchical, semantic). Quality and performance are
measured SEPARATELY; no combined score is produced and no strategy is
reported as universally better.

Metrics (per chunker, per fixture, in document order):

  Performance:  best/mean wall-clock time per document (chunking only,
                parsing is excluded), chunks produced per second. For the
                semantic chunker the embedding inference time is reported
                separately, together with the embedding throughput
                (segments embedded per second) and the batch sizes used.

  Quality:      chunk-length statistics + size distribution, content
                coverage (non-whitespace flattened characters covered at
                least once), duplicated coverage (overlap overhead),
                word-boundary start/end rates, structure metrics
                (section-boundary violation rate, tables kept intact,
                lists kept intact, chunks carrying heading context),
                source-reference preservation (content blocks referenced,
                pages referenced, references per chunk).

  Hierarchy:    for the hierarchical chunker, the number of parents, the
                number of children and the correctness of the parent-child
                relationship (orphan children, parents without children,
                bidirectional link consistency).

  A section *boundary* is a heading whose level is the same as or higher
  than the previous heading's level (the points where the structure-aware
  strategy opens a new chunk). A boundary is *violated* when it falls
  strictly inside a chunk; the violation rate is violations / boundaries,
  reported as N/A (null) when the corpus document has no such boundaries.

Null (N/A) is used when a metric does not apply (e.g. heading context
metrics on a document with no headings); N/A values are excluded from
averages and are never treated as 0.

Existing reports are never silently destroyed: before overwriting, previous
report files are copied into reports/archive/ with a UTC timestamp.

Semantic chunking uses a deterministic fake embedding model by default, so
the algorithm comparison is reproducible and needs no model weights. With
--semantic-embedding local the cached local transformer model is used
instead; nothing is ever downloaded and an unavailable model is reported
as skipped.

Usage:
  python benchmarks/chunkers/benchmark.py
  python benchmarks/chunkers/benchmark.py --repeats 5 --output-dir reports
  python benchmarks/chunkers/benchmark.py --chunkers fixed structure_aware
  python benchmarks/chunkers/benchmark.py --chunk-size 800 --overlap 100
  python benchmarks/chunkers/benchmark.py --semantic-embedding local
  python benchmarks/chunkers/benchmark.py --no-archive
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FIXTURES = {
    "pdf_simple": ("benchmarks/parsers/pdf/fixtures/simple_digital.pdf", "pymupdf"),
    "pdf_multipage": ("benchmarks/parsers/pdf/fixtures/multipage_structured.pdf", "pymupdf"),
    "pdf_tables": ("benchmarks/parsers/pdf/fixtures/tables.pdf", "pymupdf"),
    "html_headings": ("benchmarks/parsers/html/fixtures/documentation_headings.html", "beautifulsoup"),
    "html_long": ("benchmarks/parsers/html/fixtures/long_document.html", "beautifulsoup"),
    "html_nested_lists": ("benchmarks/parsers/html/fixtures/nested_lists.html", "beautifulsoup"),
    "html_table": ("benchmarks/parsers/html/fixtures/complex_table.html", "beautifulsoup"),
    "html_unicode": ("benchmarks/parsers/html/fixtures/unicode_text.html", "beautifulsoup"),
    "docx_headings": ("benchmarks/parsers/office/fixtures/headings.docx", "python_docx"),
    "docx_tables": ("benchmarks/parsers/office/fixtures/tables.docx", "python_docx"),
    "docx_nested_lists": ("benchmarks/parsers/office/fixtures/nested_lists.docx", "python_docx"),
    "pptx_multi_slide": ("benchmarks/parsers/office/fixtures/multi_slide.pptx", "python_pptx"),
    "pptx_table": ("benchmarks/parsers/office/fixtures/table.pptx", "python_pptx"),
    "xlsx_multi_sheet": ("benchmarks/parsers/spreadsheets/fixtures/multi_sheet.xlsx", "openpyxl"),
    "csv_standard": ("benchmarks/parsers/spreadsheets/fixtures/standard.csv", "csv"),
}

DEFAULT_REPORTS_DIR = Path(__file__).parent / "reports"
REPORT_VERSION = 2

ALL_CHUNKERS = ["fixed", "recursive", "structure_aware", "hierarchical", "semantic"]


def build_parser(name: str):
    if name == "pymupdf":
        from parsers.pdf.pymupdf import PyMuPDFParser

        return PyMuPDFParser()
    if name == "beautifulsoup":
        from parsers.html.beautifulsoup import BeautifulSoupParser

        return BeautifulSoupParser()
    if name == "python_docx":
        from parsers.office.docx.python_docx import PythonDocxParser

        return PythonDocxParser()
    if name == "python_pptx":
        from parsers.office.pptx.python_pptx import PythonPptxParser

        return PythonPptxParser()
    if name == "openpyxl":
        from parsers.spreadsheets.xlsx.openpyxl import OpenpyxlParser

        return OpenpyxlParser()
    if name == "csv":
        from parsers.spreadsheets.csv import CsvParser

        return CsvParser()
    raise ValueError(name)


class InstrumentedEmbeddingModel:
    """Wraps an embedding model and records inference time and batch sizes."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.calls = 0
        self.texts = 0
        self.seconds = 0.0
        self.batch_sizes: list[int] = []

    @property
    def model_id(self) -> str:
        return getattr(self._inner, "model_id", None) or getattr(self._inner, "name", "")

    @property
    def name(self) -> str:
        return self.model_id

    def embed_documents(self, inputs):
        payload = list(inputs)
        start = time.perf_counter()
        result = self._inner.embed_documents(payload)
        self.seconds += time.perf_counter() - start
        self.calls += 1
        self.texts += len(payload)
        self.batch_sizes.append(len(payload))
        return result

    def embed_queries(self, inputs):
        return self._inner.embed_queries(inputs)

    def embed_query(self, input):
        return self._inner.embed_query(input)

    def reset(self) -> None:
        self.calls = 0
        self.texts = 0
        self.seconds = 0.0
        self.batch_sizes = []


def build_embedding_model(kind: str):
    """Return an embedding model for the benchmark.

    ``fake`` is the deterministic, dependency-free model used for
    reproducible algorithm comparisons. ``local`` uses the locally cached
    transformer model and never downloads anything.
    """
    if kind == "fake":
        from embeddings.deterministic import DeterministicEmbeddingModel

        return DeterministicEmbeddingModel(dimension=64)
    if kind == "local":
        from embeddings.local import LocalTransformerEmbeddingModel

        return LocalTransformerEmbeddingModel()
    raise ValueError(kind)


def build_chunker(
    name: str,
    chunk_size: int,
    overlap: int,
    *,
    semantic_embedding: str = "fake",
    semantic_threshold: float = 0.5,
    child_size: int | None = None,
    child_overlap: int | None = None,
):
    """Build a chunker instance for the benchmark.

    Returns ``(chunker, metadata)``. The metadata records the effective
    configuration so the report is self-describing.
    """
    if name == "fixed":
        from chunkers.fixed import FixedSizeChunker, FixedSizeConfig

        return FixedSizeChunker(FixedSizeConfig(chunk_size=chunk_size, overlap=overlap)), {
            "chunk_size": chunk_size,
            "overlap": overlap,
        }
    if name == "recursive":
        from chunkers.recursive import RecursiveChunker, RecursiveConfig

        return RecursiveChunker(RecursiveConfig(chunk_size=chunk_size, overlap=overlap)), {
            "chunk_size": chunk_size,
            "overlap": overlap,
        }
    if name == "structure_aware":
        from chunkers.structure_aware import StructureAwareChunker, StructureAwareConfig

        return StructureAwareChunker(StructureAwareConfig(chunk_size=chunk_size)), {
            "chunk_size": chunk_size,
            "overlap": None,
        }
    if name == "hierarchical":
        from chunkers.hierarchical import HierarchicalChunker, HierarchicalConfig

        effective_child = child_size if child_size is not None else max(1, chunk_size // 2)
        effective_overlap = child_overlap if child_overlap is not None else min(overlap, effective_child - 1)
        effective_overlap = max(0, effective_overlap)
        return HierarchicalChunker(
            HierarchicalConfig(
                parent_size=chunk_size,
                child_size=effective_child,
                child_overlap=effective_overlap,
            )
        ), {
            "chunk_size": chunk_size,
            "overlap": None,
            "parent_size": chunk_size,
            "child_size": effective_child,
            "child_overlap": effective_overlap,
        }
    if name == "semantic":
        from chunkers.semantic import SemanticChunker, SemanticConfig

        model = InstrumentedEmbeddingModel(build_embedding_model(semantic_embedding))
        return SemanticChunker(
            SemanticConfig(
                embedding_model=model,
                similarity_threshold=semantic_threshold,
                max_chunk_size=chunk_size,
                overlap=min(overlap, chunk_size - 1),
            )
        ), {
            "chunk_size": chunk_size,
            "overlap": min(overlap, chunk_size - 1),
            "embedding_model": model.model_id,
            "embedding_backend": semantic_embedding,
            "similarity_threshold": semantic_threshold,
        }
    raise ValueError(name)


PARSER_LIBS = {
    "pymupdf": "pymupdf",
    "beautifulsoup": "bs4",
    "python_docx": "docx",
    "python_pptx": "pptx",
    "openpyxl": "openpyxl",
    "csv": None,
}


def parse_corpus():
    docs = {}
    skipped = []
    for fixture_name, (rel_path, parser_name) in FIXTURES.items():
        fixture = ROOT / rel_path
        library = PARSER_LIBS[parser_name]
        if not fixture.exists():
            skipped.append({"fixture": fixture_name, "reason": f"missing fixture {rel_path}"})
            continue
        if library is not None:
            try:
                __import__(library)
            except ImportError as exc:
                skipped.append({"fixture": fixture_name, "reason": f"missing optional dependency {library}: {exc}"})
                continue
        try:
            docs[fixture_name] = build_parser(parser_name).parse(fixture)
        except Exception as exc:  # noqa: BLE001 - benchmark must not die on one bad fixture
            skipped.append({"fixture": fixture_name, "reason": f"parse failed: {type(exc).__name__}: {exc}"})
    return docs, skipped


def _percentile(sorted_values, fraction):
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, int(fraction * len(sorted_values)))
    return sorted_values[index]


def flatten_with_spans(doc):
    from chunkers.contracts.flatten import flatten_document

    return flatten_document(doc)


def boundary_stats(doc, flats, chunk_ranges):
    """Top-level section-boundary preservation.

    A *boundary* is a heading whose level is the same as or higher than the
    previous heading's level -- exactly the points where the structure-aware
    chunker opens a new chunk (and where the flattened document changes
    section). A boundary is *violated* when it falls strictly inside a chunk
    (chunk start < boundary < chunk end), i.e. the chunk spans both sides of
    the boundary. Rate is violations / boundaries; N/A (None) when the
    document has no boundaries.
    """
    from chunkers.structure_aware.chunker import _heading_level

    headings = [fb for fb in flats if fb.block.block_type.value == "heading"]
    total = 0
    violated = 0
    previous_level = None
    for fb in headings:
        level = _heading_level(fb.block)
        if previous_level is not None and level <= previous_level:
            total += 1
            boundary = fb.start
            if any(cs < boundary < ce for cs, ce in chunk_ranges):
                violated += 1
        previous_level = level
    return total, (round(violated / total, 4) if total else None)


def measure(doc, chunks, text, flats, reference_limit_holder):
    """Quality metrics for an already-produced chunk list."""
    n = len(text)
    coverage_counts = [0] * (n + 1)
    for chunk in chunks:
        cs, ce = chunk.metadata["char_start"], chunk.metadata["char_end"]
        coverage_counts[cs] += 1
        coverage_counts[ce] -= 1
    running = 0
    total_chunk_chars = 0
    duplicated_chars = 0
    covered = 0
    for i, ch in enumerate(text):
        running += coverage_counts[i]
        if not ch.isspace() and running > 0:
            covered += 1
        if running > 1:
            duplicated_chars += 1
    for chunk in chunks:
        total_chunk_chars += len(chunk.text)

    non_ws = sum(1 for ch in text if not ch.isspace())
    coverage = covered / non_ws if non_ws else 1.0
    duplicate_rate = duplicated_chars / total_chunk_chars if total_chunk_chars else 0.0

    lengths = sorted(len(c.text) for c in chunks)
    avg_len = statistics.fmean(lengths) if lengths else 0.0
    stdev_len = statistics.pstdev(lengths) if len(lengths) > 1 else 0.0

    word_start = sum(1 for c in chunks if c.metadata["char_start"] == 0 or text[c.metadata["char_start"] - 1].isspace())
    word_end = sum(1 for c in chunks if c.metadata["char_end"] >= n or text[c.metadata["char_end"]].isspace())

    block_spans = [(fb.start, fb.end, fb.block.block_type.value) for fb in flats]
    chunk_ranges = [(c.metadata["char_start"], c.metadata["char_end"]) for c in chunks]

    def block_intact(span):
        return any(cs <= span[0] and ce >= span[1] for cs, ce in chunk_ranges)

    tables = [s[:2] for s in block_spans if s[2] == "table"]
    lists = [s[:2] for s in block_spans if s[2] == "list"]
    tables_intact = sum(1 for t in tables if block_intact(t)) if tables else None
    lists_intact = sum(1 for t in lists if block_intact(t)) if lists else None

    boundaries_total, boundary_violation = boundary_stats(doc, flats, chunk_ranges)

    referenced_blocks = {
        (ref.page_number, ref.block_index)
        for chunk in chunks
        for ref in chunk.block_refs
    }
    content_blocks = {
        ((fb.page_number), fb.block_index)
        for fb in flats
    }
    block_ref_rate = len(referenced_blocks) / len(content_blocks) if content_blocks else None

    all_pages = sorted({fb.page_number for fb in flats if fb.page_number is not None})
    referenced_pages = sorted({p for p in all_pages if p in {ref.page_number for chunk in chunks for ref in chunk.block_refs}})
    page_ref_rate = (len(referenced_pages) / len(all_pages)) if all_pages else None

    headings_present = any(fb.block.block_type.value == "heading" for fb in flats)
    if headings_present:
        heading_rate = sum(1 for c in chunks if c.heading) / len(chunks) if chunks else 0.0
    else:
        heading_rate = None

    dist = {
        "<=25%": 0,
        "25-50%": 0,
        "50-75%": 0,
        "75-100%": 0,
        ">100%": 0,
    }
    reference = max(1, reference_limit_holder)
    for length in lengths:
        ratio = length / reference
        if ratio <= 0.25:
            dist["<=25%"] += 1
        elif ratio <= 0.50:
            dist["25-50%"] += 1
        elif ratio <= 0.75:
            dist["50-75%"] += 1
        elif ratio <= 1.00:
            dist["75-100%"] += 1
        else:
            dist[">100%"] += 1

    return {
        "num_chunks": len(chunks),
        "length_chars_avg": round(avg_len, 2),
        "length_chars_min": min(lengths) if lengths else 0,
        "length_chars_max": max(lengths) if lengths else 0,
        "length_chars_p50": _percentile(lengths, 0.50),
        "length_chars_p95": _percentile(lengths, 0.95),
        "length_chars_stdev": round(stdev_len, 2),
        "size_distribution_vs_limit": dist,
        "coverage": round(coverage, 4),
        "duplicate_char_rate": round(duplicate_rate, 4),
        "total_chunk_chars": total_chunk_chars,
        "word_boundary_start_rate": round(word_start / len(chunks), 4) if chunks else None,
        "word_boundary_end_rate": round(word_end / len(chunks), 4) if chunks else None,
        "boundaries_total": boundaries_total,
        "boundary_violation_rate": boundary_violation,
        "tables_total": len(tables),
        "tables_intact_rate": round(tables_intact / len(tables), 4) if tables else None,
        "lists_total": len(lists),
        "lists_intact_rate": round(lists_intact / len(lists), 4) if lists else None,
        "heading_context_rate": round(heading_rate, 4) if heading_rate is not None else None,
        "content_blocks_total": len(content_blocks),
        "block_ref_rate": round(block_ref_rate, 4) if block_ref_rate is not None else None,
        "pages_total": len(all_pages),
        "page_ref_rate": round(page_ref_rate, 4) if page_ref_rate is not None else None,
        "refs_per_chunk": round(sum(len(c.block_refs) for c in chunks) / len(chunks), 3) if chunks else 0.0,
    }


def hierarchy_metrics(result):
    """Parent/child relationship correctness for the hierarchical chunker.

    Reports the number of parents and children, orphan children (a child
    whose parent does not exist), parents without children and the
    bidirectional consistency of the links (a parent listing a child that
    does not point back at it).
    """
    parents = result.parents
    children = result.children
    parent_ids = {p.id for p in parents}
    child_by_id = {c.id: c for c in children}

    orphan_children = sum(
        1 for c in children if c.metadata.get("parent_id") not in parent_ids
    )
    parents_without_children = sum(1 for p in parents if not p.metadata.get("child_ids"))

    links = 0
    consistent_links = 0
    for parent in parents:
        for child_id in parent.metadata.get("child_ids", []):
            links += 1
            child = child_by_id.get(child_id)
            if child is not None and child.metadata.get("parent_id") == parent.id:
                consistent_links += 1

    return {
        "num_parents": len(parents),
        "num_children": len(children),
        "orphan_children": orphan_children,
        "parents_without_children": parents_without_children,
        "parent_child_links": links,
        "parent_child_consistency_rate": (
            round(consistent_links / links, 4) if links else None
        ),
        "parent_child_link_rate": (
            round((len(children) - orphan_children) / len(children), 4) if children else None
        ),
    }


def embedding_metrics(model):
    """Embedding inference statistics for the semantic chunker."""
    if model is None:
        return {
            "embedding_seconds": None,
            "embedding_calls": None,
            "embedding_segments": None,
            "embedding_segments_per_second": None,
            "embedding_mean_batch_size": None,
            "embedding_max_batch_size": None,
        }
    return {
        "embedding_seconds": round(model.seconds, 6),
        "embedding_calls": model.calls,
        "embedding_segments": model.texts,
        "embedding_segments_per_second": (
            round(model.texts / model.seconds, 1) if model.seconds > 0 else None
        ),
        "embedding_mean_batch_size": (
            round(statistics.fmean(model.batch_sizes), 2) if model.batch_sizes else None
        ),
        "embedding_max_batch_size": max(model.batch_sizes) if model.batch_sizes else None,
    }


def evaluate_chunker(name, chunker, docs, repeats, args_chunk_size, embedding_model=None):
    rows = []
    for fixture_name, doc in docs.items():
        text, flats = flatten_with_spans(doc)
        # Timed runs: best and mean wall clock per document.
        times = []
        last_chunks = None
        error = None
        extra: dict = {}
        if embedding_model is not None:
            embedding_model.reset()
        for _ in range(max(1, repeats)):
            start = time.perf_counter()
            try:
                outcome = chunker.chunk(doc)
                times.append(time.perf_counter() - start)
            except Exception as exc:  # noqa: BLE001 - record failure, keep benchmarking
                error = f"{type(exc).__name__}: {exc}"
                break
            if name == "hierarchical":
                last_chunks = outcome.children
                extra = hierarchy_metrics(outcome)
            else:
                last_chunks = outcome
        if error is not None:
            rows.append({"fixture": fixture_name, "chunker": name, "success": False, "error": error})
            continue

        reference = args_chunk_size
        if name == "hierarchical":
            reference = max(1, args_chunk_size // 2)
        metrics = measure(doc, last_chunks, text, flats, reference)

        # Parent-level quality for the hierarchical chunker.
        if name == "hierarchical":
            parent_metrics = measure(doc, outcome.parents, text, flats, args_chunk_size)
            extra["parent_length_chars_avg"] = parent_metrics["length_chars_avg"]
            extra["parent_length_chars_max"] = parent_metrics["length_chars_max"]
            extra["parent_coverage"] = parent_metrics["coverage"]
            extra["parent_boundary_violation_rate"] = parent_metrics["boundary_violation_rate"]
            extra["parent_block_ref_rate"] = parent_metrics["block_ref_rate"]

        if name == "semantic":
            extra.update(embedding_metrics(embedding_model))

        best = min(times)
        mean = statistics.fmean(times)
        rows.append(
            {
                "fixture": fixture_name,
                "chunker": name,
                "success": True,
                "error": None,
                "source": doc.source,
                "time_seconds_best": round(best, 6),
                "time_seconds_mean": round(mean, 6),
                "chunks_per_second": round(len(last_chunks) / mean, 1) if mean > 0 else None,
                **metrics,
                **extra,
            }
        )
    return rows


def _mean(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return round(statistics.fmean(values), 4) if values else None


def summarize(rows, chunker_names):
    summaries = []
    for name in chunker_names:
        mine = [r for r in rows if r["chunker"] == name and r["success"]]
        failed = [r for r in rows if r["chunker"] == name and not r["success"]]
        summaries.append(
            {
                "chunker": name,
                "fixtures_total": len(mine) + len(failed),
                "fixtures_succeeded": len(mine),
                "fixtures_failed": len(failed),
                "chunks_total": sum(r["num_chunks"] for r in mine),
                "time_seconds_total": round(sum(r["time_seconds_mean"] for r in mine), 6),
                "chunks_per_second_overall": (
                    round(sum(r["num_chunks"] for r in mine) / sum(r["time_seconds_mean"] for r in mine), 1)
                    if mine and sum(r["time_seconds_mean"] for r in mine) > 0
                    else None
                ),
                # Quality: means over fixtures; N/A metrics excluded from the mean.
                "coverage_mean": _mean(mine, "coverage"),
                "duplicate_char_rate_mean": _mean(mine, "duplicate_char_rate"),
                "length_chars_avg_mean": _mean(mine, "length_chars_avg"),
                "length_chars_max_mean": _mean(mine, "length_chars_max"),
                "word_boundary_start_rate_mean": _mean(mine, "word_boundary_start_rate"),
                "word_boundary_end_rate_mean": _mean(mine, "word_boundary_end_rate"),
                "boundary_violation_rate_mean": _mean(mine, "boundary_violation_rate"),
                "tables_intact_rate_mean": _mean(mine, "tables_intact_rate"),
                "lists_intact_rate_mean": _mean(mine, "lists_intact_rate"),
                "heading_context_rate_mean": _mean(mine, "heading_context_rate"),
                "block_ref_rate_mean": _mean(mine, "block_ref_rate"),
                "page_ref_rate_mean": _mean(mine, "page_ref_rate"),
                "refs_per_chunk_mean": _mean(mine, "refs_per_chunk"),
                # Hierarchy metrics (hierarchical chunker only).
                "num_parents_total": sum(r.get("num_parents") or 0 for r in mine),
                "num_children_total": sum(r.get("num_children") or 0 for r in mine),
                "orphan_children_total": sum(r.get("orphan_children") or 0 for r in mine),
                "parents_without_children_total": sum(r.get("parents_without_children") or 0 for r in mine),
                "parent_child_consistency_rate_mean": _mean(mine, "parent_child_consistency_rate"),
                "parent_child_link_rate_mean": _mean(mine, "parent_child_link_rate"),
                "parent_length_chars_avg_mean": _mean(mine, "parent_length_chars_avg"),
                "parent_length_chars_max_mean": _mean(mine, "parent_length_chars_max"),
                "parent_coverage_mean": _mean(mine, "parent_coverage"),
                "parent_boundary_violation_rate_mean": _mean(mine, "parent_boundary_violation_rate"),
                "parent_block_ref_rate_mean": _mean(mine, "parent_block_ref_rate"),
                # Embedding metrics (semantic chunker only).
                "embedding_seconds_total": (
                    round(sum(r["embedding_seconds"] for r in mine if r.get("embedding_seconds") is not None), 6)
                    if any(r.get("embedding_seconds") is not None for r in mine)
                    else None
                ),
                "embedding_segments_total": sum(r.get("embedding_segments") or 0 for r in mine),
                "embedding_segments_per_second_mean": _mean(mine, "embedding_segments_per_second"),
                "embedding_mean_batch_size_mean": _mean(mine, "embedding_mean_batch_size"),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            if isinstance(flat.get("size_distribution_vs_limit"), dict):
                dist = flat.pop("size_distribution_vs_limit")
                for key, value in dist.items():
                    flat[f"size_dist_{key}"] = value
            writer.writerow({key: flat.get(key, "") for key in fieldnames})


def archive_existing_reports(output_dir: Path) -> list[str]:
    archive_dir = output_dir / "archive"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = []
    for name in ("benchmark_report.json", "benchmark_results.csv", "benchmark_summary.csv"):
        target = output_dir / name
        if not target.exists():
            continue
        archive_dir.mkdir(parents=True, exist_ok=True)
        destination = archive_dir / f"{target.stem}.{stamp}{target.suffix}"
        shutil.copy2(target, destination)
        archived.append(str(destination))
    return archived


RESULT_FIELDS = [
    "chunker", "fixture", "success", "error", "source",
    "time_seconds_best", "time_seconds_mean", "chunks_per_second",
    "num_chunks", "length_chars_avg", "length_chars_min", "length_chars_max",
    "length_chars_p50", "length_chars_p95", "length_chars_stdev",
    "size_dist_<=25%", "size_dist_25-50%", "size_dist_50-75%", "size_dist_75-100%", "size_dist_>100%",
    "coverage", "duplicate_char_rate", "total_chunk_chars",
    "word_boundary_start_rate", "word_boundary_end_rate",
    "boundaries_total", "boundary_violation_rate", "tables_total", "tables_intact_rate",
    "lists_total", "lists_intact_rate", "heading_context_rate",
    "content_blocks_total", "block_ref_rate", "pages_total", "page_ref_rate", "refs_per_chunk",
    # Hierarchical chunker.
    "num_parents", "num_children", "orphan_children", "parents_without_children",
    "parent_child_links", "parent_child_consistency_rate", "parent_child_link_rate",
    "parent_length_chars_avg", "parent_length_chars_max", "parent_coverage",
    "parent_boundary_violation_rate", "parent_block_ref_rate",
    # Semantic chunker.
    "embedding_seconds", "embedding_calls", "embedding_segments",
    "embedding_segments_per_second", "embedding_mean_batch_size", "embedding_max_batch_size",
]

SUMMARY_FIELDS = [
    "chunker", "fixtures_total", "fixtures_succeeded", "fixtures_failed",
    "chunks_total", "time_seconds_total", "chunks_per_second_overall",
    "coverage_mean", "duplicate_char_rate_mean", "length_chars_avg_mean", "length_chars_max_mean",
    "word_boundary_start_rate_mean", "word_boundary_end_rate_mean",
    "boundary_violation_rate_mean", "tables_intact_rate_mean", "lists_intact_rate_mean",
    "heading_context_rate_mean", "block_ref_rate_mean", "page_ref_rate_mean", "refs_per_chunk_mean",
    "num_parents_total", "num_children_total", "orphan_children_total",
    "parents_without_children_total", "parent_child_consistency_rate_mean",
    "parent_child_link_rate_mean", "parent_length_chars_avg_mean",
    "parent_length_chars_max_mean", "parent_coverage_mean",
    "parent_boundary_violation_rate_mean", "parent_block_ref_rate_mean",
    "embedding_seconds_total", "embedding_segments_total",
    "embedding_segments_per_second_mean", "embedding_mean_batch_size_mean",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunkers", nargs="+", choices=ALL_CHUNKERS, default=ALL_CHUNKERS)
    parser.add_argument("--repeats", type=int, default=3, help="timed repeats per document (default: 3)")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--child-size", type=int, default=None,
                        help="child chunk size for the hierarchical chunker (default: chunk_size // 2)")
    parser.add_argument("--child-overlap", type=int, default=None,
                        help="child overlap for the hierarchical chunker (default: min(overlap, child_size - 1))")
    parser.add_argument("--semantic-embedding", choices=["fake", "local"], default="fake",
                        help="embedding backend for the semantic chunker (default: fake, deterministic)")
    parser.add_argument("--semantic-threshold", type=float, default=0.5,
                        help="minimum adjacent cosine similarity for the semantic chunker")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--no-archive", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archived = [] if args.no_archive else archive_existing_reports(args.output_dir)

    docs, skipped = parse_corpus()
    print(f"Parsed {len(docs)} fixtures ({len(skipped)} skipped)")

    rows = []
    chunker_configs = {}
    for name in args.chunkers:
        try:
            chunker, config = build_chunker(
                name,
                args.chunk_size,
                args.overlap,
                semantic_embedding=args.semantic_embedding,
                semantic_threshold=args.semantic_threshold,
                child_size=args.child_size,
                child_overlap=args.child_overlap,
            )
        except Exception as exc:  # noqa: BLE001 - e.g. the local model is not cached
            print(f"Skipping {name}: {type(exc).__name__}: {exc}")
            skipped.append({"fixture": f"*:{name}", "reason": f"{type(exc).__name__}: {exc}"})
            continue
        chunker_configs[name] = config
        print(f"Chunking with {name} ...", flush=True)
        embedding_model = getattr(chunker.config, "embedding_model", None) if name == "semantic" else None
        rows.extend(
            evaluate_chunker(name, chunker, docs, args.repeats, args.chunk_size, embedding_model)
        )

    summaries = summarize(rows, args.chunkers)

    report = {
        "benchmark": "prome-chunkers",
        "report_version": REPORT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "chunkers": args.chunkers,
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "structure_aware_overlap": None,
            "repeats": args.repeats,
            "fixtures": sorted(docs),
            "length_unit": "characters",
            "semantic_embedding": args.semantic_embedding,
            "semantic_threshold": args.semantic_threshold,
            "chunker_configs": chunker_configs,
            "note": (
                "Performance (time, chunks/s) and quality metrics are reported separately; "
                "there is no combined score. Metrics describe behavior on this synthetic "
                "fixture corpus only; no strategy is universally better. N/A means the "
                "metric did not apply (e.g. no headings in the document) and is excluded "
                "from averages; it is never reported as 0. The semantic chunker uses a "
                "deterministic fake embedding model by default so the algorithm "
                "comparison is reproducible; --semantic-embedding local uses the cached "
                "local transformer model and never downloads weights."
            ),
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "skipped_fixtures": skipped,
        "results": rows,
        "summary": summaries,
        "archived_previous": archived,
    }

    json_path = args.output_dir / "benchmark_report.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n")

    write_csv(args.output_dir / "benchmark_results.csv", rows, RESULT_FIELDS)
    write_csv(args.output_dir / "benchmark_summary.csv", summaries, SUMMARY_FIELDS)

    print(f"\nWrote {json_path}")
    print(f"Wrote {args.output_dir / 'benchmark_results.csv'}")
    print(f"Wrote {args.output_dir / 'benchmark_summary.csv'}")
    if archived:
        print(f"Archived previous reports: {', '.join(archived)}")
    for s in summaries:
        line = (
            f"  {s['chunker']}: chunks={s['chunks_total']} time={s['time_seconds_total']}s "
            f"coverage={s['coverage_mean']} boundary_violations={s['boundary_violation_rate_mean']} "
            f"tables_intact={s['tables_intact_rate_mean']} block_refs={s['block_ref_rate_mean']}"
        )
        if s["chunker"] == "hierarchical":
            line += (
                f" | parents={s['num_parents_total']} children={s['num_children_total']}"
                f" orphans={s['orphan_children_total']} link_ok={s['parent_child_consistency_rate_mean']}"
            )
        if s["chunker"] == "semantic":
            line += (
                f" | embed_segments={s['embedding_segments_total']}"
                f" embed_s={s['embedding_seconds_total']}"
                f" embed_seg_per_s={s['embedding_segments_per_second_mean']}"
            )
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
