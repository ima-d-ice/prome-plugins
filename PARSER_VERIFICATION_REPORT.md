# Parser Verification Report

Audit date: 2026-11-12
Project: `~/Desktop/prome-rag-plugins`
Scope: every implemented parser plugin, the automated test suite, the deterministic
benchmark fixtures, and a new manual verification CLI.

Status legend (used consistently):

- **PASS** — parser executed and meaningful output was verified on real fixtures and/or
  manual runs.
- **FAIL** — parser executed but produced incorrect/incomplete output on at least one
  fixture (details below; not a crash).
- **SKIP** — cannot execute: missing optional dependencies or explicitly unavailable
  requirements. The implementation exists but was never run; it is NOT claimed to work.
- **NOT IMPLEMENTED** — feature does not exist.
- **NOT VERIFIED** — implementation exists but its behavior has not been exercised.
- **N/A** — capability does not exist for that parser; recorded as N/A, never as zero.

## 1. Headline numbers

| Measure | Value |
|---|---|
| CLI ids in `--list` | 34 (33 runnable, 1 not implemented) |
| Distinct parser implementation plugins (classes) | 23 |
| Plugins whose optional dependencies import cleanly | 22 |
| Plugins executed and returning a valid `ParsedDocument` | 22 |
| Plugins executed but never successfully (all fixtures failed) | 0 |
| Plugins skipped (dependency missing) | 1 — `images.paddleocr` |
| Plugins not implemented | 1 — `pdf.ocr` |
| Automated tests (`tests/parsers`) | **337 passed, 5 skipped** |
| Full suite (`python -m pytest`) as of report time | **691 passed, 17 skipped** (includes a concurrently added `tests/chunkers` — see §12) |
| `pip check` | **exit 1** — `faster-whisper 1.2.1 requires onnxruntime, which is not installed` |
| Fixture × parser checks (Phase 5 harness) | 145 checks: **131 PASS, 7 FAIL, 7 N/A** |

`--list` shows 34 ids because it exposes the parsers plus configuration variants:
3 JSON modes (`json.auto/json.json/json.ndjson`), 7 tree-sitter grammar pins
(`code.tree_sitter.<lang>`), 2 video frame strategies (`video.ffmpeg`, `video.ffmpeg.scene`),
and `pdf.ocr` as NOT IMPLEMENTED. The 23 distinct plugins:

PDF 3 (PyMuPDF, pypdf, Docling) · HTML 2 (BeautifulSoup, Trafilatura) ·
Office 4 (python-docx, Docling DOCX, python-pptx, Docling PPTX) ·
Spreadsheets 3 (openpyxl, pandas, CSV) · Email 2 (EML, MBOX) · JSON 1 (stdlib) ·
XML 2 (ElementTree, lxml) · Code 2 (Python AST, Tree-sitter) ·
Images 2 (Tesseract, PaddleOCR) · Audio 1 (faster-whisper) · Video 1 (PyAV/FFmpeg)

## 2. Automated test results (Phase 2)

Commands run (project venv):

```bash
.venv/bin/python -m pytest tests/parsers -q     # 337 passed, 5 skipped in ~120s
.venv/bin/python -m pytest -q                   # 337 passed, 5 skipped at the time; 691 passed, 17 skipped after a concurrent chunkers suite appeared
.venv/bin/python -m pip check                   # exit 1 (see below)
```

The reported baseline (337 passed, 5 skipped) is accurate. The 5 skips are intentional:

- `tests/parsers/html/test_beautifulsoup.py:165` — skipped because lxml is installed
  (an honest either/or test path, not a hidden failure)
- 4 × `tests/parsers/pdf/test_new_fixtures.py` — pypdf capability skips (no table
  detection, no images as blocks, no lists, no headings). These are documented
  capability gaps, not failures.

`pip check` fails with:

```
faster-whisper 1.2.1 requires onnxruntime, which is not installed.
```

faster-whisper imports and transcribes fine without it (it is needed for the Silero VAD
path; the parser defaults to `vad_filter=False`). Investigated, not fixed, and not
installed automatically (per audit rules). See §10.

No automated test failed. No test was weakened or converted to a skip.

## 3. Per-plugin verification matrix

### PDF

| Plugin | Import | Executed | Fixture result | Status |
|---|---|---|---|---|
| `pdf.pymupdf` | OK | Yes | 12/12 fixtures pass expectations (tables, headings, reading order, rotated/headers-footers markers all found) | **PASS** |
| `pdf.pypdf` | OK | Yes | Text extraction passes; tables/headings/images/lists = **N/A** (capability does not exist; recorded as N/A, not zero) | **PASS** |
| `pdf.docling` | OK | Yes | Digital PDFs OK; headers/footers + rotated marker gaps **confirmed** (documented in expectations.json). `rotated.pdf`: rotated text misclassified as a TABLE and `ROTATED-MARKER-90` missing | **FAIL** (on rotated.pdf); otherwise PASS |
| `pdf.ocr` | n/a | n/a | Not implemented (`parsers/pdf/ocr/` is a placeholder; Docling runs `do_ocr=False`) | **NOT IMPLEMENTED** |

PDF edge behavior verified: `encrypted.pdf` → `EncryptedDocumentError` for all three
parsers (never a silent pass). `scanned.pdf` → PyMuPDF returns 2 IMAGE blocks, zero
text; pypdf/docling return page warnings, zero fabricated text.

### HTML

| Plugin | Executed | Result | Status |
|---|---|---|---|
| `html.beautifulsoup` | Yes | 11/12 fixtures pass; `malformed.html` → 1 heading block swallows the unclosed `<h1>`'s siblings; TABLE (Name/Value, Alpha/Beta) lost | **FAIL** (malformed.html only); otherwise PASS |
| `html.trafilatura` | Yes | 11/12 fixtures pass; `malformed.html` → heading not detected, table lost | **FAIL** (malformed.html only); otherwise PASS |

Publication dates: verified NOT fabricated — `simple_article.html` (no date) yields no
date; `metadata_og_jsonld.html` yields `2026-09-14` from its JSON-LD `datePublished`.
`extensive_date_search` is off by default. Date fabrication concern is not present.

### Office

| Plugin | Executed | Result | Status |
|---|---|---|---|
| `office.docx.python_docx` | Yes | All fixtures pass (text, tables with dimensions, headings, metadata incl. core properties) | **PASS** |
| `office.docx.docling` | Yes | Content extracted correctly, but **core properties not surfaced**: `simple_text.docx` yields no `title`/`author` (expected `Quarterly Operations Brief` / `A. Example`) | **FAIL** (strict metadata check); content extraction PASS |
| `office.pptx.python_pptx` | Yes | All fixtures pass (slides, shapes, tables, images, speaker notes) | **PASS** |
| `office.pptx.docling` | Yes | Content extracted; **speaker notes not extracted**: `images_notes.pptx` misses `Remind the audience about the roadmap.` (benchmark quality there is 0.9167) | **FAIL** (strict text check); otherwise PASS |

### Spreadsheets

| Plugin | Executed | Result | Status |
|---|---|---|---|
| `spreadsheets.xlsx.openpyxl` | Yes | All fixtures pass (merged cells, multi-sheet, formulas, cell types, `row_cells`) | **PASS** |
| `spreadsheets.xlsx.pandas` | Yes | All fixtures pass (dtypes, missing-value stats; no formulas/merged spans by design) | **PASS** |
| `spreadsheets.csv` | Yes | All fixtures pass (sniffed `;`, `|`, quoting, unicode, large file) | **PASS** |

### Email

| Plugin | Executed | Result | Status |
|---|---|---|---|
| `email.eml` | Yes | All 5 fixtures pass (headers, bodies, `is_thread_reply`, attachment with filename/content_type/size) | **PASS** |
| `email.mbox` | Yes | `archive.mbox` → 3 messages, pages 1–3, headers/bodies pass | **PASS** |

### JSON

| Plugin | Executed | Result | Status |
|---|---|---|---|
| `json.json` / `json.ndjson` / `json.auto` | Yes | All pass; NDJSON via explicit mode and via auto fallback (`$[i]` paths, RECORD blocks per line); malformed input → `CorruptDocumentError` | **PASS** |

### XML

| Plugin | Executed | Result | Status |
|---|---|---|---|
| `xml.elementtree` | Yes | Executes; content correct BUT **known bug**: every path segment is doubled (`/catalog/book[1]/book/title/title` instead of `/catalog/book[1]/title`) — see §9 | **FAIL** (xpath metadata); text extraction PASS |
| `xml.lxml` | Yes | Content correct; emits namespace-prefixed paths (`/root/dc:title`), so the unqualified `expected_xpaths` in `expectations.json` (`/root/title`) do not match | **FAIL** (expectations mismatch); content extraction PASS |

### Code

| Plugin | Executed | Result | Status |
|---|---|---|---|
| `code.python_ast` | Yes | `sample.py` pass (definitions, signatures, decorators, docstrings, line ranges). Non-Python input → clean `CorruptDocumentError` (by design) | **PASS** |
| `code.tree_sitter` | Yes | All 7 grammars verified individually via explicit pins (`python`, `javascript`, `typescript`, `java`, `c`, `cpp`, `go`) on the `sample.*` fixtures; extension inference verified too | **PASS** |

### Images

| Plugin | Executed | Result | Status |
|---|---|---|---|
| `images.tesseract` | Yes | `clean_text.png`, `receipt.png` pass; `blank.png` → `EmptyDocumentError`; `corrupt.png` → `CorruptDocumentError` (system tesseract 5.5.4, langs `eng`/`osd`) | **PASS** |
| `images.paddleocr` | **No** | Not installed (heavy dependency, deliberately excluded from default deps). Reported SKIP by CLI; benchmark report records it as skipped. Never verified. | **SKIP** |

### Audio

| Plugin | Executed | Result | Status |
|---|---|---|---|
| `audio.faster_whisper` | Yes (model `tiny`, cached) | `speech_en.wav/.mp3`, `speech_fr.wav`, `long_speech.wav` pass; `silence.wav` → 0 blocks + warning (no invented text); `corrupt.wav` → `CorruptDocumentError`; language probabilities surfaced; no diarization (never fabricated). Project-reported speech_en WER 0.0714 (not independently re-measured) | **PASS** |

### Video

| Plugin | Executed | Result | Status |
|---|---|---|---|
| `video.ffmpeg` | Yes | `testsrc.mp4`: h264, 320×240 @ 10 fps, AAC audio stream, FRAME blocks at 0/1/2 s. Scene strategy on `scene_cut.mp4`: frame captured at **1.5 s** (the hard cut). `audio_only.mp4`/`corrupt.mp4` → `CorruptDocumentError`/`EmptyDocumentError` by design. `extract_audio_to_wav()` bridge tested end-to-end (WAV → faster-whisper, non-speech tone → 0 segments + warning, correct). `frame_dir` export tested: 3 valid JPEGs with frame references in block metadata. | **PASS** |

## 4. Fixture verification results (Phase 5)

A disposable harness (kept in `/tmp`, **not** in the repo) ran every available plugin
on every relevant deterministic fixture and checked the machine-checkable parts of
`expectations.json` (text presence, page/sheet/message/attachment counts, table counts
and cells, heading detection, reading order, metadata keys, xpaths, video/audio
metadata, `min_blocks`). Strict capability handling: pypdf table/heading/image/list
expectations were scored as N/A, not failed.

145 fixture×parser checks: **131 PASS, 7 FAIL, 7 N/A** (N/A = scanned-PDF OCR ×1 +
python_ast on 6 non-Python fixtures; plus many pypdf capability N/A notes).

The 7 failures (all detailed in §3):

1. `pdf.docling` on `rotated.pdf` (rotated text → TABLE misclassification; `ROTATED-MARKER-90` lost; reading order violated)
2. `html.beautifulsoup` on `malformed.html` (unclosed `<h1>` swallows content; table lost)
3. `html.trafilatura` on `malformed.html` (heading/table lost)
4. `office.docx.docling` on `simple_text.docx` (core properties not surfaced)
5. `office.pptx.docling` on `images_notes.pptx` (speaker notes not extracted)
6. `xml.elementtree` on `catalog.xml` (xpath doubling bug)
7. `xml.lxml` on `namespaces.xml` (prefix-qualified vs expected unqualified paths)

Important: the benchmark suites in `benchmarks/parsers/*/reports/` remain green as
committed (e.g. HTML 0.975/0.967, docling_pptx 0.9167) because benchmarks score
text-coverage/heading/table heuristics, not strict all-or-nothing assertions.
Fixtures with documented `known_extraction_gaps` (headers_footers.pdf, rotated.pdf for
docling) were verified against those gaps: the missing markers are exactly the
documented ones (plus the docling rotated-PDF table misclassification, which is worse
than documented and should be reported upstream).

No benchmark score was used as proof of manual correctness. Fixtures passed/failed on
their own merits. Capabilities that do not exist were recorded as N/A. Fixtures that
cannot exercise a feature (e.g. `empty.eml` is 0-byte and rejected by input
validation; `malformed.eml` parses leniently by design) are called out as such rather
than counted.

## 5. Manual verification CLI (Phase 3)

New file: `scripts/manual_parser_check.py`. Nothing in `parsers/` was modified.

Features: `--list` (all 34 ids with dependency status), `--parser`, `--input`,
`--output` (full `ParsedDocument.as_dict()` JSON), `--show-blocks N`, `--preview-chars N`,
`--allow-model-download` (required for docling/faster-whisper/paddleocr first-use model
downloads), `--traceback`. Type-aware previews for tables (dimensions/headers/rows),
records (keys, path), key-value leaves (path/value/type), definitions (kind/name/
signature/parent/lines), segments (timestamps, language, avg_logprob, no_speech_prob),
frames (number/timestamp/dimensions/export path), attachments (filename, content type,
size). Pageless documents display `Pages: n/a (pageless document)`. Exit codes:
0 success · 1 parse failure · 2 missing deps · 3 model download not permitted · 4 bad
input · 5 not implemented · 6 unknown id. A parser is never silently converted to
success.

Examples:

```bash
python scripts/manual_parser_check.py --list
python scripts/manual_parser_check.py --parser pdf.pymupdf --input /path/to/document.pdf
python scripts/manual_parser_check.py --parser pdf.docling --input /path/to/document.pdf --allow-model-download
python scripts/manual_parser_check.py --parser html.trafilatura --input /path/to/page.html
python scripts/manual_parser_check.py --parser office.docx.python_docx --input /path/to/document.docx
python scripts/manual_parser_check.py --parser spreadsheets.xlsx.openpyxl --input /path/to/workbook.xlsx
python scripts/manual_parser_check.py --parser email.eml --input /path/to/message.eml
python scripts/manual_parser_check.py --parser audio.faster_whisper --input /path/to/audio.wav --allow-model-download
python scripts/manual_parser_check.py --parser video.ffmpeg.scene --input /path/to/clip.mp4
python scripts/manual_parser_check.py --parser pdf.pymupdf --input /path/to/document.pdf --output /tmp/result.json
```

Full walkthrough: `MANUAL_PARSER_VERIFICATION.md`.

## 6. Docling model usage during this audit

`--allow-model-download` was passed for Docling/faster-whisper runs; **no weights were
downloaded** — the Hugging Face cache (`~/.cache/huggingface`) already contains
`docling-project/docling-models`, `docling-project/docling-layout-heron` and
`Systran/faster-whisper-tiny`. Loading from cache was observed in the logs
(`Loading weights: 100%` with no fetch). PaddleOCR (which would download ~20–50 MB and
pull in paddlepaddle) was NOT run.

## 7. Known limitations and unsupported capabilities

- **PDF OCR: not implemented.** Scanned PDFs yield images/warnings, never text.
- **pypdf**: no tables, headings, lists, or image blocks (all N/A, by design).
- **Docling**: no OCR; loses headers/footers and rotated text (documented gaps,
  confirmed live); rotated text misclassified as TABLE; DOCX core properties not
  surfaced; PPTX speaker notes not extracted; first use downloads models.
- **BeautifulSoup / Trafilatura on malformed HTML**: structure degrades to whatever
  the lenient HTML parser nests (an unclosed `<h1>` captures the rest of the document).
  Text is preserved; block structure is lost.
- **ElementTree XML**: path doubling bug (§9). Benchmarks never asserted xpaths, so it
  shipped silently.
- **lxml XML**: paths are namespace-prefix qualified (`/root/dc:title`); the
  unprefixed `expected_xpaths` in expectations.json cannot match as written.
- **Trafilatura**: no date fabricated; no date when none discoverable.
- **pandas XLSX**: no formulas, no merged-cell spans, no formatting.
- **CSV**: delimiter sniffing can misfire on ambiguous files; `.txt` accepted.
- **faster-whisper**: no diarization; `silence.wav` yields `nn` language with low
  probability (documented); fixture speech is synthesized (regression signal, not an
  accuracy measurement); WER 0.0714 reported by the project, not re-measured here.
- **FFmpegVideoParser**: scene detection = luma heuristic (hard cuts);
  `frame_count_estimate` is container-reported and may be inexact; audio-only video
  intentionally raises `CorruptDocumentError`.
- **tree-sitter**: TSX parsed with the plain TypeScript grammar.
- **Malformed files** behave by design: raise `CorruptDocumentError`/`EmptyDocumentError`
  (verified for JSON, XML, CSV/XLSX, DOCX/PPTX, images, audio, video, PDF).

## 8. Automated tests vs manual verification

| Suite | Result |
|---|---|
| `tests/parsers` | 337 passed, 5 skipped |
| Full `pytest` at report time | 691 passed, 17 skipped (includes concurrently added `tests/chunkers`; see §12) |
| `pip check` | exit 1 (onnxruntime missing for faster-whisper) |
| Benchmark reports (committed) | all 11 suites pass; `paddleocr` recorded as skipped |
| Manual fixture harness | 131 PASS / 7 FAIL / 7 N/A (145 checks) |
| Manual CLI on fixtures | every parser executed (except paddleocr) |

## 9. Bugs and issues found (production code NOT modified)

### Bug A — `xml.elementtree` doubles every path segment

`parsers/xml/elementtree/parser.py`, `_walk_et`: the parent passes an already-indexed
path (`f"{current_path}/{c_local}{idx_suffix}"`) as `path`, and the callee then appends
its own tag again (`current_path = f"{path}/{local}"`).

- Expected: `/catalog/book[1]/title`, actual: `/catalog/book[1]/book/title/title`
- Expected: `/catalog/book[1] {'id': …}`, actual: `/catalog/book[1]/book {'id': …}`

Minimal proposed correction (NOT applied):

```python
def _walk_et(elem, path=""):
    local, _ = _split_tag(elem.tag)
    current_path = path if path else f"/{local}"   # was: f"{path}/{local}" if path else f"/{local}"
    ...
```

Text content is unaffected; only `metadata["xpath"]` (and the RECORD content prefix)
is wrong. Tests assert `any("/catalog/book" in xp)` — a substring check that tolerates
the doubling; benchmarks never check xpaths. Left untouched per audit rules; fix is
a one-line change for you to approve.

### Bug B — Docling misclassifies rotated rotated text as a TABLE

On `rotated.pdf`, the rotated paragraph is exported as a single TABLE block;
`ROTATED-MARKER-90` is lost entirely. Worse than the documented `known_extraction_gaps`.
Recommend reporting upstream or extending the documented gaps. Parser code untouched.

### Gap C — Docling DOCX metadata

`simple_text.docx` core properties (`title`, `author`) are not exposed by Docling.
Content extraction is otherwise correct. Either accept as a Docling limitation or
metadata should be read via python-docx; not changed.

### Gap D — Docling PPTX speaker notes

`images_notes.pptx` notes are not extracted. Content otherwise correct. `python-pptx`
is the reference path for notes. Not changed.

### Issue E — `pip check` fails

`faster-whisper 1.2.1 requires onnxruntime`. Not installed by this audit. The parser
works with default settings (`vad_filter=False`). If VAD is wanted, `pip install
onnxruntime` — your call (it was not auto-installed per audit rules).

### Issue F — malformed-HTML structure loss

Beautifulsoup (and trafilatura) lose all nested blocks under an unclosed heading on
`malformed.html`. Likely an inherent consequence of lenient HTML parsing plus the
deliberate "heading emits inline text, no nested walk" design. Text coverage is
preserved; block structure is not. Candidate improvement (not applied): walk heading
children for nested blocks too — but that changes parser behavior and needs your call.

### Expectations mismatch — `xml` xpaths

`benchmarks/parsers/xml/expectations.json` expects unqualified paths; lxml qualifies
with prefixes and ElementTree doubles them. Either the expectations or the parsers
should be aligned. Not changed.

## 10. What was intentionally not done

- No production parser code was modified (rules 1–2).
- No tests, benchmarks, fixtures, expectations, or reports were modified (rule 8).
- PaddleOCR was not installed and was never run (rules 5–6).
- No model weights were downloaded; first-use downloads were gated behind
  `--allow-model-download` (rule: explicit permission).
- Nothing local was uploaded or transmitted (rule 7).
- Failures were not hidden or converted to skips (rule 9): see the 7 fixture failures.
- `pip check` failure documented, not silenced by installing onnxruntime.

## 11. Changes made during this verification task

New files (only):

1. `scripts/manual_parser_check.py` — the manual verification CLI (34 listed entries).
2. `MANUAL_PARSER_VERIFICATION.md` — your manual testing workflow + per-family checklist.
3. `PARSER_VERIFICATION_REPORT.md` — this report.

The Phase-5 fixture harness was written to and run from `/tmp`
(`/var/folders/.../opencode/fixture_harness.py`, results in `fixture_results.json`) —
not part of the repository.

Also observed, NOT created by this audit: a concurrent **chunkers** milestone appeared
in this repository during the session (`chunkers/`, `tests/chunkers/`, `README.md` and
`pyproject.toml` updated). It is out of scope here and was left untouched.

## 12. Concurrent changes detected (important)

While this audit ran, a separate process modified the repository:
`pyproject.toml` now includes `chunkers*` in packages, `README.md` documents chunking,
and `chunkers/` + `tests/chunkers/` (354 passed, 12 skipped) were added at ~10:35.
The parser suite remains 337 passed / 5 skipped, and no parser code was rebuilt.
Report your local test numbers accordingly.

## 13. What still needs your attention

1. **Approve or reject the one-line ElementTree path fix** (Bug A) — it is the only
   confirmed outright bug, and it is metadata-only.
2. **Decide on the malformed-HTML behavior** (Issue F) — change the heading branch to
   walk nested blocks, or document the loss.
3. **Align `expectations.json` xpaths** with the lxml/ElementTree path styles, or fix
   the parsers (Issue: expectations mismatch).
4. **Decide how to present docling's known gaps** — add `rotated.pdf` table
   misclassification to `known_extraction_gaps` or handle upstream (Bug B).
5. **Decide whether docling-docx/docling-pptx gaps are acceptable** (metadata core
   props, speaker notes) or whether expectations should be relaxed to match reality.
6. **`pip check`**: install `onnxruntime` if you want VAD / a clean `pip check`.
7. **PaddleOCR**: actually run and benchmark it if/when you install it — its row in
   every report is SKIP, i.e. its "implementation works" is currently unproven.
8. **PDF OCR** remains unimplemented — `scanned.pdf` will never yield text until it is.
9. **Idempotency of generated reports**: re-running benchmarks archives the current
   `reports/` into `reports/archive/`; review before committing if that is unwanted.
10. **Concurrent chunkers milestone**: verify its provenance — it was not created by
    this audit, and `pyproject.toml`/`README.md` changed under us.

## 14. Exact commands

### Automated tests

```bash
cd ~/Desktop/prome-rag-plugins
source .venv/bin/activate
python -m pytest tests/parsers -v        # 337 passed, 5 skipped
python -m pytest                         # full suite (691 passed, 17 skipped at report time)
python -m pip check                      # exit 1: faster-whisper needs onnxruntime
```

### List available parsers

```bash
python scripts/manual_parser_check.py --list
```

### Manually test PDF, DOCX, Excel, HTML, image, audio, video

```bash
python scripts/manual_parser_check.py --parser pdf.pymupdf            --input /path/to/document.pdf --show-blocks 10
python scripts/manual_parser_check.py --parser pdf.docling            --input /path/to/document.pdf --allow-model-download
python scripts/manual_parser_check.py --parser office.docx.python_docx --input /path/to/document.docx
python scripts/manual_parser_check.py --parser spreadsheets.xlsx.openpyxl --input /path/to/workbook.xlsx
python scripts/manual_parser_check.py --parser html.beautifulsoup     --input /path/to/page.html
python scripts/manual_parser_check.py --parser html.trafilatura       --input /path/to/page.html
python scripts/manual_parser_check.py --parser images.tesseract       --input /path/to/scan.png
python scripts/manual_parser_check.py --parser audio.faster_whisper   --input /path/to/audio.wav --allow-model-download
python scripts/manual_parser_check.py --parser video.ffmpeg           --input /path/to/clip.mp4
python scripts/manual_parser_check.py --parser video.ffmpeg.scene     --input /path/to/scene_cuts.mp4
```

### Export results to JSON

```bash
python scripts/manual_parser_check.py --parser pdf.pymupdf --input /path/to/document.pdf --output /tmp/result.json
```

## 15. What actually worked (verified by execution, not by import)

PyMuPDF · pypdf · Docling PDF (digital) · BeautifulSoup · Trafilatura · python-docx ·
Docling DOCX (content) · python-pptx · Docling PPTX (content) · openpyxl · pandas ·
CSV · EML · MBOX · JSON/NDJSON (all modes) · ElementTree · lxml · Python AST ·
tree-sitter (all 7 grammars) · Tesseract · faster-whisper (tiny) · PyAV/FFmpeg video
(metadata, interval frames, scene-cut frames, frame export, audio-extraction bridge) —
**22 of 23 implementation plugins execute and produce verified output.**

What did not work / is not proven:

- PDF OCR — **NOT IMPLEMENTED**
- PaddleOCR — **SKIP** (not installed, never run, unproven)
- ElementTree xpath metadata (Bug A), Docling rotated-PDF TABLE (Bug B), Docling
  DOCX/PPTX metadata gaps (C/D) — documented FAIL rows
- `pip check` — **exit 1**
