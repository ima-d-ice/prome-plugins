# PDF Parser Benchmark — Prome RAG Plugins

Reproducible comparison of the three PDF parser implementations
(PyMuPDF, PyPDF, Docling) on synthetic fixtures. Quality and speed are
measured **separately**. No scores are hardcoded; unavailable parsers are
reported as `skipped` with the reason, and capabilities a parser does not
implement are reported as **N/A** (null in JSON, empty in CSV) — never as 0
and never as a failure.

## Regenerate fixtures

```bash
python benchmarks/parsers/pdf/generate_fixtures.py
```

Creates `fixtures/*.pdf` plus `expectations.json` with per-fixture ground
truth. Fixture bytes are deterministic (fixed metadata dates,
`reproducible=True, no_new_id=True`), so SHA-256 hashes are stable across
regeneration runs and are recorded in the report. `encrypted.pdf` is the only
exception (random encryption key) and is used by the tests only.

`encrypted.pdf` is not part of the scored corpus.

### Corpus

| Fixture | Challenge |
|---|---|
| `simple_digital.pdf` | baseline digital text |
| `multipage_structured.pdf` | multi-page headings + paragraphs |
| `tables.pdf` | ruled table + cells/header content |
| `blank_page.pdf` | blank page preservation |
| `multi_section.pdf` | multi-section reading order |
| `two_column.pdf` | two columns with distinct reading order |
| `headers_footers.pdf` | running header/footer + page numbers |
| `rotated.pdf` | rotated text (`rotate=90`) and a `/Rotate=90` page |
| `multipage_table.pdf` | 60-row table spanning two pages, repeated header |
| `images_captions.pdf` | vector diagram + inserted bitmap + captions |
| `scanned.pdf` | image-only pages, **no text layer** (`requires_ocr: true`) |
| `mixed_sizes.pdf` | title/headings/body/fine print + bullet list |

Ground truth lives in `expectations.json` (expected pages, text snippets,
headings, tables, table header/cells, per-page table cells
(`expected_table_pages`), reading-order markers, image/list expectations,
`requires_ocr`). Per-fixture `known_extraction_gaps` documents measured
parser misses (e.g. Docling's rotated-label drop); gaps only excuse snippets
in the test suite — the benchmark counts them as missed.

## Capability matrix

| Capability | PyMuPDF | PyPDF | Docling |
|---|---|---|---|
| Text extraction | ✓ | ✓ | ✓ |
| Structure (headings/blocks) | ✓ | — | ✓ |
| Tables | ✓ (`find_tables`) | — | ✓ (TableFormer, markdown export) |
| Images surfaced as blocks | ✓ | — | ✓ (picture/chart) |
| Lists surfaced as blocks | — | — | ✓ (`list_item`) |
| OCR | — | — | — (`do_ocr=False`) |

A capability with no implementation yields N/A for every metric that depends
on it; those rows/cells never drag averages down and are excluded from every
mean. The benchmark score for the scanned fixture is therefore N/A for all
three parsers, not 0 — OCR remains future work.

## Run the benchmark

```bash
python benchmarks/parsers/pdf/benchmark.py                  # all available parsers, 1 repeat
python benchmarks/parsers/pdf/benchmark.py --repeats 3      # 3 steady-state repeats per fixture
python benchmarks/parsers/pdf/benchmark.py --parsers pymupdf pypdf
python benchmarks/parsers/pdf/benchmark.py --no-archive     # do not snapshot existing reports
```

Requires the repo root on `PYTHONPATH` (the script inserts it itself) and
installation of the parser extras you want to measure (`pip install -e
'.[all]'`).

Each parser gets a fresh instance. The **cold pass** (one parse per fixture,
corpus order) precedes the steady-state repeats and includes any lazy
one-time initialization (Docling loads its layout models on the first
convert, ~8-10 s on Apple M-series; ~0.4-1.1 s per document afterwards).
Existing report files are archived to `reports/archive/<name>.<UTC
timestamp>.<ext>` before being overwritten, so results are never lost.

## Outputs (in `reports/`)

- `benchmark_report.json` — full machine-readable report: config (parsers,
  repeats, fixture hashes, capability matrix, score definitions),
  environment (python/platform/library versions), per-fixture rows,
  `per_fixture` (parser-vs-parser per fixture), `capabilities`
  (per-capability comparison per parser), per-parser `summary`, skipped
  list, and `archived_previous`.
- `benchmark_results.csv` — one row per (parser, fixture).
- `benchmark_summary.csv` — one row per parser.

## Metrics

| Metric | Definition |
|---|---|
| `extraction_success_rate` | ok / (total − unsupported) |
| `failure_rate` | failed / (total − unsupported) |
| `fixtures_unsupported` | rows where a required capability (e.g. OCR) is missing |
| `page_coverage` | `min(1, parsed_pages / expected_pages)` |
| `text_coverage` | fraction of expected snippets found in extracted text |
| `heading_preservation` | fraction of expected headings found in heading blocks |
| `table_cell_coverage` | fraction of expected table cells found in **table-block content** |
| `table_header_coverage` | fraction of expected header cells found in table-block content |
| `multipage_table_continuity` | per-page cell markers found on their expected page |
| `table_text_duplication` | fraction of table cells also present in non-table text (informational; 0 = no duplicated noise) |
| `structure_preservation` | mean of the applicable structure subscores (heading + table cells); N/A for parsers without a structure capability |
| `reading_order` | 1.0 iff all order markers appear in ascending position; N/A for image-only fixtures |
| `image_detection` / `list_detection` | fraction of expected (image/list) items found as image/list blocks; N/A otherwise |
| `quality_score` | mean of applicable non-N/A components: text, structure, reading order, image, list |
| `performance_score` | steady-state pages/second relative to the fastest non-skipped parser (1.0 = fastest) |
| `cold_start_seconds` | fresh parser construction + first document |
| `first_document_seconds` | time of the first cold-pass parse |
| `subsequent_document_mean_seconds` | mean steady-state (warm) time of the remaining fixtures |
| `cold_total_seconds` / `total_time_seconds` | cold pass total; total incl. construction + warm repeats |
| `throughput_pages_per_second` | steady-state pages / mean warm time |
| `time_seconds_cold` (row) | cold-pass time for this fixture |

Row `status`: `ok` (parsed and scored), `failed` (parse raised — applicable
components score 0), `unsupported` (missing capability — all metrics N/A).

## Interpreting results

- **PyMuPDF** is the fast baseline for digital PDFs; headings are detected by
  a font-size heuristic (>=13pt and >=1.2x the 10th-percentile body size),
  tables via `page.find_tables()`. Reading order is column-aware: narrow
  blocks are grouped left-to-right of a centered gutter; full-width blocks
  (headers/footers/banners) act as barriers splitting horizontal bands.
- **PyPDF** is the lightest option; it has no structure capability, so
  structure/heading/table metrics are N/A (its `quality_score` is over text
  and reading order only).
- **Docling** is layout-aware (headings, tables as markdown, lists) but much
  slower; converter (model) setup is reused across parses per parser
  instance, so warm parses are far cheaper than cold ones. It excludes
  running headers/footers from the body flow by design (they are tagged as
  page boilerplate) — measured, reported, and excluded from the test-suite
  text assertion via `known_extraction_gaps`. With OCR disabled it drops
  vertical rotated text and misreads rotated pages.
- Numbers on 1-page fixtures are noisy; use `--repeats` for anything serious.
  Fixtures are tiny synthetic documents — use them for regression comparison,
  not as a production quality guarantee.

## Known limitations

- OCR is intentionally disabled in all three parsers; the scanned fixture
  exists to expose and report that limitation (status `unsupported`), not to
  score it. OCR is future work.
- Structure metrics are N/A for PyPDF, not 0.
- Docling converter reuse is per parser instance and guarded by a lock;
  instances are independent. Reusing converters across threads is supported
  only through this lock; concurrent converters may contend on model memory.
- The corpus is synthetic and small; add real-world documents before drawing
  production conclusions.
