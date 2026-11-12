# HTML parser benchmark

Compares the two HTML parsers (`beautifulsoup`, `trafilatura`) on a synthetic
local fixture corpus. No network access is performed; fixtures are generated
and stored under `fixtures/`.

Both parsers are scored against the same ground truth
(`expectations.json`) but through different **extraction scopes**:

| Parser | Scope | Meaning |
|---|---|---|
| beautifulsoup | `full_dom` | Expected to keep nav/header/footer as authored; boilerplate retention is reported, never penalized |
| trafilatura | `main_content` | Expected to isolate the main article; `boilerplate_removal` earns credit and counts toward the quality score |

## Metrics

Quality (mean of applicable, non-N/A components):
`text_coverage`, `structure_preservation` (headings + table cells),
`reading_order` (DOM-order proxy), `metadata_completeness`,
`list_detection`, `link_preservation`, `image_detection`
(beautifulsoup only — trafilatura surfaces no images),
`boilerplate_removal` (trafilatura scope only), `forbidden_text_isolation`
(scripts/styles must not leak).

Also reported: `heading_level_accuracy`, table cell/header coverage,
boilerplate retention/removal, per-fixture cold and steady-state timings,
cold-start vs first-document time, throughput (docs/s, blocks/s, bytes/s),
parser and library versions, fixture bytes/sha256, repeat count.

Unsupported capabilities are recorded as **N/A** (null in JSON, empty in
CSV) and are excluded from all averages — never scored as 0 and never
invented.

## Usage

```bash
python benchmarks/parsers/html/benchmark.py
python benchmarks/parsers/html/benchmark.py --parsers beautifulsoup
python benchmarks/parsers/html/benchmark.py --repeats 5
python benchmarks/parsers/html/benchmark.py --output-dir reports --no-archive
```

Outputs (written to this directory's `reports/`):

- `benchmark_report.json` — full report (config, environment, rows, per-fixture comparison, capability comparison, summary)
- `benchmark_results.csv` — one row per (parser, fixture)
- `benchmark_summary.csv` — one row per parser

Before overwriting, any existing report files are copied into
`reports/archive/` with a UTC timestamp. Previous reports are never
destroyed.

## Ground truth

`expectations.json` encodes, per fixture: expected text snippets, expected
headings and their levels, expected tables (count, header cells, cell
contents), expected list items, expected link hrefs, expected image alts,
DOM-order markers, required metadata keys, boilerplate markers
(`boilerplate_text`) and forbidden markers (`forbidden_text`).

Snippet expectations were authored from the fixtures, not from parser
output, and are case-insensitive substring matches. A parser that cannot
perform a capability (e.g. trafilatura and images) records N/A there by
design.
