# Chunker Benchmark — Prome RAG Plugins

Reproducible comparison of the five core chunking strategies on a shared
fixture corpus (the same synthetic fixtures the parser benchmarks use,
parsed with the fast default parser per family).

Quality and performance are measured **separately**. No combined score is
produced and no strategy is claimed to be universally superior — the numbers
describe behavior on this corpus only.

## What is measured

Per (strategy, fixture) row; one document parsed once, then chunked `repeats`
times. Chunking is timed; parsing time is excluded.

### Performance

| Metric | Meaning |
|---|---|
| `time_seconds_best` / `time_seconds_mean` | Wall-clock per `chunker.chunk(doc)` call |
| `chunks_per_second` | `num_chunks / time_seconds_mean` |

### Quality

| Metric | Meaning |
|---|---|
| `length_chars_avg/min/max/p50/p95/stdev` | Distribution of chunk lengths (characters) |
| `size_distribution_vs_limit` | Histogram: `<=25%`, `25-50%`, `50-75%`, `75-100%`, `>100%` of the configured limit |
| `coverage` | Fraction of the flattened document's non-whitespace characters covered by ≥ 1 chunk (1.0 = nothing missing) |
| `duplicate_char_rate` | Fraction of chunked characters emitted more than once (overlap overhead) |
| `word_boundary_start_rate` / `word_boundary_end_rate` | Fraction of chunks starting/ending on a whitespace boundary (or at document edges) |
| `boundaries_total` / `boundary_violation_rate` | A *boundary* is a heading whose level is the same as or higher than the previous heading's level (where the structure-aware strategy opens a chunk). A boundary is *violated* when it falls strictly inside a chunk. Rate = violations / boundaries; N/A when a document has no boundaries |
| `tables_intact_rate` (`tables_intact`) | Fraction of TABLE blocks fully contained in one chunk |
| `lists_intact_rate` | Same for LIST blocks |
| `heading_context_rate` | Fraction of chunks carrying a `heading` value (0 for structure-agnostic strategies by design) |
| `block_ref_rate` | Fraction of content-bearing blocks referenced by ≥ 1 chunk (1.0 = full provenance) |
| `page_ref_rate` | Fraction of referenced pages (N/A for pageless formats) |
| `refs_per_chunk` | Average block references per chunk |

Null (N/A) is used whenever a metric does not apply (e.g. `heading_context_rate`
when the document has no headings, or `boundary_violation_rate` when it has
no section boundaries). N/A is excluded from averages and never reported as 0.

### Hierarchy (hierarchical chunker only)

The hierarchical chunker produces two levels. The quality metrics above are
measured on the **children** (the units meant for embedding); the parent
level is reported separately.

| Metric | Meaning |
|---|---|
| `num_parents` / `num_children` | Chunks produced at each level |
| `orphan_children` | Children whose `parent_id` does not resolve to a parent (must be 0) |
| `parents_without_children` | Parents that list no children (must be 0) |
| `parent_child_links` | Total number of parent→child links |
| `parent_child_consistency_rate` | Fraction of links where the child points back at the parent (1.0 = bidirectional) |
| `parent_child_link_rate` | Fraction of children linked to an existing parent (1.0 = no orphans) |
| `parent_length_chars_avg/max` | Parent-level chunk length distribution |
| `parent_coverage` | Coverage measured on the parent level |
| `parent_boundary_violation_rate` | Section-boundary violations on the parent level |
| `parent_block_ref_rate` | Source-reference preservation on the parent level |

### Embedding (semantic chunker only)

| Metric | Meaning |
|---|---|
| `embedding_seconds` | Time spent inside the embedding model during the timed run |
| `embedding_calls` | Number of `embed_documents` calls (batch count) |
| `embedding_segments` | Number of candidate segments embedded |
| `embedding_segments_per_second` | Embedding throughput |
| `embedding_mean_batch_size` / `embedding_max_batch_size` | Batch sizes used |

## Corpus

15 documents parsed from the parser fixture suites: PDF (PyMuPDF), HTML
(BeautifulSoup), DOCX (python-docx), PPTX (python-pptx), XLSX (openpyxl) and
CSV (stdlib). The corpus is intentionally small (synthetic, designed for
regression testing); treat results as directional, not as production
baselines.

## Run

```bash
python benchmarks/chunkers/benchmark.py                  # all five strategies, 1000/200, 3 repeats
python benchmarks/chunkers/benchmark.py --chunk-size 800 --overlap 100
python benchmarks/chunkers/benchmark.py --repeats 5
python benchmarks/chunkers/benchmark.py --chunkers fixed structure_aware
python benchmarks/chunkers/benchmark.py --semantic-embedding local   # real cached model
python benchmarks/chunkers/benchmark.py --no-archive
```

Fixtures that cannot be parsed (missing parser or fixture file) are reported
as `skipped_fixtures` with a reason; nothing is fabricated.

### Semantic embedding backends

- `--semantic-embedding fake` (default) uses the deterministic, dependency-free
  `embeddings.deterministic.DeterministicEmbeddingModel`. This keeps the
  algorithm comparison byte-for-byte reproducible and needs no model weights.
- `--semantic-embedding local` uses `embeddings.local.LocalTransformerEmbeddingModel`
  with the locally cached transformer checkpoint. It **never downloads
  weights**; if the model is not in the local Hugging Face cache the run is
  reported as skipped with the reason.

## Outputs

- `reports/benchmark_report.json` — full machine-readable report (config,
  environment, per-fixture rows, per-strategy summary, skipped fixtures).
- `reports/benchmark_results.csv` — one row per (strategy, fixture).
- `reports/benchmark_summary.csv` — one row per strategy.

Existing reports are archived to `reports/archive/` before overwriting.

## Methodology notes

- Character lengths, not token counts, are compared; the default
  configuration is `chunk_size=1000`, `overlap=200` for the fixed and
  recursive strategies, `chunk_size=1000` for structure-aware, and for
  hierarchical `parent_size=1000` / `child_size=500` / `child_overlap=200`.
- The hierarchical chunker is measured on its **children**; the parent level
  is reported under the `parent_*` metrics. Children use `child_size` as the
  size-distribution reference.
- Coverage/duplication are computed on the canonical flattened document text
  (`chunkers.contracts.flatten.flatten_document`), the same text the
  chunkers slice, so the accounting is exact.
- Word-boundary rates only apply cleanly at whitespace-separated text;
  CJK text without spaces will report low boundary rates for all strategies
  (nothing to latch onto — every character is its own boundary candidate in
  tokenizer mode, but this benchmark measures the character mode).
- Timing at this corpus size is microsecond-scale; use `--repeats` for
  stability. Parsing is excluded — only `chunk()` is timed. For the semantic
  chunker the embedding time is included in the total and additionally
  reported separately, so embedding cost is never hidden.
- `heading_context_rate` = 0.0 for fixed and recursive is expected: they
  are structure-agnostic by design and never emit heading metadata.
- `boundary_violation_rate` for structure-aware is expected to be 0.0 on
  well-formed heading hierarchies; it will rise on parsers that flatten
  structure (e.g. PyPDF) — that is a property of the input, not a bug.
- No overall quality score is computed. The metrics above describe different
  properties; combining them into one number would hide the trade-offs.
