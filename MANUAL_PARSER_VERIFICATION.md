# Manual Parser Verification

How to manually verify every Prome RAG parser plugin on your own documents,
using the deterministic fixtures as a starting point and then your real files.

All commands assume the project root is `~/Desktop/prome-rag-plugins` and use the
project virtual environment.

## 1. Activate the project virtual environment

```bash
cd ~/Desktop/prome-rag-plugins
source .venv/bin/activate
```

Or invoke the interpreter explicitly without activating:

```bash
.venv/bin/python scripts/manual_parser_check.py --list
```

## 2. List available parsers and dependency status

```bash
python scripts/manual_parser_check.py --list
```

Every parser entry shows one of:

- `available` — its optional dependencies import cleanly (does **not** prove it works)
- `missing deps: ...` — name(s) of missing requirements
- `NOT IMPLEMENTED` — reserved/documented but no implementation

Exit codes used by the CLI:

| Code | Meaning |
|------|---------|
| 0 | Parser executed and returned a `ParsedDocument` |
| 1 | Parser executed but failed (exception or invalid result) |
| 2 | Skipped: missing optional dependencies or external requirements |
| 3 | Skipped: first-use model download not permitted (pass `--allow-model-download`) |
| 4 | Input validation failed (file missing / not a file / empty) |
| 5 | Parser not implemented |
| 6 | Unknown parser id |

## 3. Check dependency availability

```bash
python scripts/manual_parser_check.py --list          # per-parser status
python -m pip check                                    # environment consistency
python -c "import pymupdf, pypdf, docling, bs4, trafilatura, docx, pptx, openpyxl, pandas, lxml"
```

System-level requirements:

- `tesseract` binary for `images.tesseract` (e.g. `brew install tesseract`). Check with `tesseract --version`.
- `ffmpeg` binary is **not** required for `video.ffmpeg` (PyAV wheels bundle ffmpeg).
- Hugging Face cache: `~/.cache/huggingface` (Docling models, faster-whisper models).

## 4. Manually test each parser

General form:

```bash
python scripts/manual_parser_check.py --parser <parser-id> --input /path/to/file [options]
```

Options:

- `--show-blocks N` — max representative blocks to display (default 10, `0` = none)
- `--preview-chars N` — max characters per text preview (default 200)
- `--output PATH` — write the complete serialized `ParsedDocument` as JSON
- `--allow-model-download` — explicitly permit documented first-use model downloads
- `--traceback` — print a full traceback on failure

### PDF

```bash
python scripts/manual_parser_check.py --parser pdf.pymupdf --input /path/to/document.pdf
python scripts/manual_parser_check.py --parser pdf.pypdf   --input /path/to/document.pdf
python scripts/manual_parser_check.py --parser pdf.docling --input /path/to/document.pdf --allow-model-download
```

PDF OCR is **NOT IMPLEMENTED** (`pdf.ocr`). Scanned PDFs yield image blocks (PyMuPDF)
or page warnings (pypdf/docling) — never fabricated text.

### HTML

```bash
python scripts/manual_parser_check.py --parser html.beautifulsoup --input /path/to/page.html
python scripts/manual_parser_check.py --parser html.trafilatura --input /path/to/page.html
```

### Office (DOCX / PPTX)

```bash
python scripts/manual_parser_check.py --parser office.docx.python_docx --input /path/to/document.docx
python scripts/manual_parser_check.py --parser office.docx.docling     --input /path/to/document.docx --allow-model-download
python scripts/manual_parser_check.py --parser office.pptx.python_pptx --input /path/to/slides.pptx
python scripts/manual_parser_check.py --parser office.pptx.docling     --input /path/to/slides.pptx --allow-model-download
```

### Spreadsheets

```bash
python scripts/manual_parser_check.py --parser spreadsheets.xlsx.openpyxl --input /path/to/workbook.xlsx
python scripts/manual_parser_check.py --parser spreadsheets.xlsx.pandas   --input /path/to/workbook.xlsx
python scripts/manual_parser_check.py --parser spreadsheets.csv           --input /path/to/data.csv
```

### Email

```bash
python scripts/manual_parser_check.py --parser email.eml  --input /path/to/message.eml
python scripts/manual_parser_check.py --parser email.mbox --input /path/to/mailbox.mbox
```

### JSON / NDJSON

```bash
python scripts/manual_parser_check.py --parser json.auto   --input /path/to/data.json
python scripts/manual_parser_check.py --parser json.json   --input /path/to/data.json
python scripts/manual_parser_check.py --parser json.ndjson --input /path/to/events.ndjson
```

### XML

```bash
python scripts/manual_parser_check.py --parser xml.elementtree --input /path/to/catalog.xml
python scripts/manual_parser_check.py --parser xml.lxml        --input /path/to/catalog.xml
```

### Code

```bash
python scripts/manual_parser_check.py --parser code.python_ast        --input /path/to/module.py
python scripts/manual_parser_check.py --parser code.tree_sitter       --input /path/to/source.ts   # language inferred from extension
python scripts/manual_parser_check.py --parser code.tree_sitter.python     --input /path/to/module.py
python scripts/manual_parser_check.py --parser code.tree_sitter.javascript --input /path/to/app.js
python scripts/manual_parser_check.py --parser code.tree_sitter.typescript --input /path/to/app.ts
python scripts/manual_parser_check.py --parser code.tree_sitter.java       --input /path/to/App.java
python scripts/manual_parser_check.py --parser code.tree_sitter.c          --input /path/to/main.c
python scripts/manual_parser_check.py --parser code.tree_sitter.cpp        --input /path/to/main.cpp
python scripts/manual_parser_check.py --parser code.tree_sitter.go         --input /path/to/main.go
```

Supported grammars: python, javascript (.js/.mjs/.jsx), typescript (.ts/.tsx — both
parsed with the non-TSX grammar), java, c (.c/.h), cpp (.cpp/.cc/.cxx/.hpp), go.

### Images

```bash
python scripts/manual_parser_check.py --parser images.tesseract --input /path/to/scan.png
python scripts/manual_parser_check.py --parser images.paddleocr --input /path/to/scan.png --allow-model-download
```

`images.paddleocr` is NOT installed in the default environment and is a heavy
dependency; it may report SKIP. It downloads model weights on first use.

### Audio

```bash
python scripts/manual_parser_check.py --parser audio.faster_whisper --input /path/to/lecture.wav --allow-model-download
```

Default model is `tiny`. Larger models are NOT downloaded automatically — pass
`model_size="base"`/`"small"` etc. by instantiating the parser directly (note: this
downloads the corresponding weights on first use):

```bash
python -c "from parsers.audio.faster_whisper import FasterWhisperParser; \
print(FasterWhisperParser(model_size='base').parse('/path/to/lecture.wav').text)"
```

### Video

```bash
python scripts/manual_parser_check.py --parser video.ffmpeg       --input /path/to/clip.mp4
python scripts/manual_parser_check.py --parser video.ffmpeg.scene --input /path/to/clip.mp4
```

Metadata + selective frame extraction are covered by `parse()`. Extracting the
audio stream to WAV is a separate method (not part of `parse()`); it can then be
fed to the audio parser:

```bash
python - <<'EOF'
from parsers.video.ffmpeg import FFmpegVideoParser
from parsers.audio.faster_whisper import FasterWhisperParser

video = FFmpegVideoParser()
wav = video.extract_audio_to_wav("/path/to/clip.mp4", "/tmp/clip_audio.wav", sample_rate=16000)
print("WAV:", wav)
print(FasterWhisperParser(model_size="tiny").parse(wav).text)
EOF
```

To save extracted frames as JPEG files, pass a `frame_dir`:

```bash
python - <<'EOF'
from parsers.video.ffmpeg import FFmpegVideoParser
doc = FFmpegVideoParser(frame_strategy="interval", interval_seconds=1.0, max_frames=10,
                        frame_dir="/tmp/clip_frames").parse("/path/to/clip.mp4")
for block in doc.blocks:
    print(block.metadata["frame_number"], block.metadata["timestamp"], block.metadata["frame_path"])
EOF
```

## 5. Select real local documents

Use your own files — the CLI never uploads or transmits anything; everything runs
locally. Suggested starter set:

- One digitally generated PDF with tables (e.g. an invoice export)
- One scanned PDF (expect: no fabricated text, image blocks or warnings)
- One encrypted PDF (expect: `EncryptedDocumentError`, not a crash)
- One DOCX with a table and one PPTX with speaker notes
- One XLSX with merged cells and formulas, one CSV with semicolons
- One `.eml` with an attachment, one `.mbox` archive
- One `.ndjson` stream
- One source file per language you care about
- One PNG/JPEG scan for OCR, one audio file for transcription (expect a model download)
- One video with hard scene cuts (expect FRAME blocks at cut points)

Verify file size and format first:

```bash
ls -lh /path/to/document.pdf
python scripts/manual_parser_check.py --parser pdf.pymupdf --input /path/to/document.pdf --show-blocks 3
```

## 6. Inspect extracted content

The summary shows: parser, version, status, duration, page count (or `n/a (pageless)`),
block count, block-type distribution, total text length, warnings, and
representative blocks. Block previews are type-aware:

- Tables: sheet/dimensions, headers, first rows (`row_cells` for XLSX/CSV, line rows for PDF)
- Records: `json_path`, keys, key count
- Key/value leaves: `path = value (data_type)`
- Definitions: kind, name, signature, parent class, line range
- Segments: start/end timestamps, duration, language, avg_logprob, no_speech_prob
- Frames: frame number, timestamp, dimensions, export location
- Attachments: filename, content type, size in bytes

Limit what is printed:

```bash
python scripts/manual_parser_check.py --parser pdf.pymupdf --input /path/to/document.pdf --show-blocks 20 --preview-chars 400
python scripts/manual_parser_check.py --parser xml.lxml --input /path/to/catalog.xml --show-blocks 0   # summary only
```

## 7. Export results to JSON

```bash
python scripts/manual_parser_check.py --parser pdf.pymupdf --input /path/to/document.pdf --output /tmp/result.json
python -m json.tool /tmp/result.json | less
```

The JSON is `ParsedDocument.as_dict()`: `source`, `parser_name`, `parser_version`,
`metadata`, `pages[]` (paged formats) or `blocks[]` (pageless formats), `warnings`.
The file is written locally; nothing is uploaded.

## 8. Distinguish errors from missing optional dependencies

| Symptom | Meaning | Action |
|---------|---------|--------|
| `SKIP ... missing requirements` (exit 2) | Optional dependency/module or system binary missing | Install the extra or the binary; re-run |
| `SKIP ... model download` (exit 3) | Parser wants first-use model weights | Re-run with `--allow-model-download` |
| `ERROR: input file ...` (exit 4) | File missing/not a file/0 bytes | Check the path |
| `FAIL: parser ... raised <Error>` (exit 1) | Parser ran and failed | Read the error; add `--traceback` |
| `NOT IMPLEMENTED` (exit 5) | Reserved capability | Not available; see report |
| `pip check` complains | Environment inconsistency | See `PARSER_VERIFICATION_REPORT.md` |

Common traps:

- `images.tesseract` needs the system `tesseract` binary — Python package alone is not enough.
- `faster-whisper` vs `av`: see `IMPLEMENTATION_PROGRESS.md` (av>=19 incompatibility handled
  inside the parser; do not downgrade av blindly).
- Encrypted PDFs raise `EncryptedDocumentError` by design — that is a FAIL you should see,
  not a silent pass.

## 9. Media parsers that require model downloads

| Parser | Model | Downloaded when | Cache |
|--------|-------|-----------------|-------|
| `pdf.docling` | Docling layout/table models | First converter initialization | `~/.cache/huggingface` |
| `office.docx.docling` / `office.pptx.docling` | Docling models | First converter initialization | `~/.cache/huggingface` |
| `audio.faster_whisper` | Whisper model (default `tiny`) | First `parse()` | `~/.cache/huggingface` |
| `images.paddleocr` | Detection/recognition weights | First run | paddle cache dir |

All of the above are gated behind `--allow-model-download` in the CLI. The default
verification pass does NOT trigger downloads; the `tiny` faster-whisper model and the
Docling models are already present in this machine's HF cache.

Not a model download, but a system dependency: `tesseract` (installed via brew here,
`eng` and `osd` language data only).

## 10. Known limitations and unsupported capabilities

- PDF OCR: **not implemented** (`parsers/pdf/ocr/` is a placeholder; Docling runs with `do_ocr=False`).
- pypdf: no table detection, no heading detection, no image blocks, no lists (by design).
- PyMuPDF table detection and heading detection are heuristics; two-column reading
  order uses a gutter heuristic.
- Docling: no OCR; misses headers/footers and rotated text (documented
  `known_extraction_gaps` in `benchmarks/parsers/pdf/expectations.json`); DOCX core
  properties (`title`/`author`) are not surfaced; PPTX speaker notes are not extracted;
  first use downloads model weights.
- beautifulsoup/trafilatura on malformed HTML: structure degrades to whatever the
  lenient HTML parser nests (e.g. an unclosed `<h1>` swallows following blocks).
- trafilatura: no date is fabricated — a page without a discoverable publication date
  yields none (`extensive_date_search` off by default).
- ElementTree XML parser doubles every path segment (`/catalog/book[1]/book/...`);
  lxml emits namespace-prefixed paths (`/root/dc:title`). Both are documented in
  `PARSER_VERIFICATION_REPORT.md`.
- pandas XLSX parser: no formula preservation, no merged-cell spans (use openpyxl for those).
- CSV parser: sniffs delimiters; `.txt` is accepted but ambiguous files may misdetect.
- faster-whisper: no diarization (never fabricated); quality bounded by model size;
  synthesized speech fixtures are a regression signal, not an accuracy benchmark.
- FFmpegVideoParser scene detection: hard cuts only (global luma heuristic).
- `frame_count_estimate` is container-reported and may be missing/inexact.
- tree-sitter `.tsx` files are parsed with the plain TypeScript grammar.
- `video.ffmpeg` corrupt/audio-only/empty files raise errors by design (see `--traceback`).
- Email parsers are deliberately lenient: malformed MIME is extracted on a best-effort basis.

## 11. Manual testing checklist

### PDF
- [ ] `pdf.pymupdf` on a digital PDF: headings/tables detected, page numbers preserved
- [ ] `pdf.pypdf` on the same PDF: text extracted, tables absent (expected)
- [ ] `pdf.docling` on the same PDF: structure extracted (`--allow-model-download`)
- [ ] pymupdf/pypdf/docling on `benchmarks/parsers/pdf/fixtures/encrypted.pdf`: `EncryptedDocumentError`
- [ ] Scanned PDF: no fabricated text (`scanned.pdf`)
- [ ] Two-column PDF: reading order check (`two_column.pdf`)
- [ ] JSON export: `--output /tmp/out.json` opens with `python -m json.tool`

### HTML
- [ ] beautifulsoup on `simple_article.html`: headings, lists, links preserved
- [ ] trafilatura on the same: boilerplate removed, metadata present
- [ ] Date not fabricated on a page without dates (`simple_article.html` has none)
- [ ] Malformed HTML handled without crash (`malformed.html`) — verify what is lost

### Office
- [ ] python-docx `tables.docx`: 2 table blocks with dimensions
- [ ] python-pptx `multi_slide.pptx`: 3 pages, heading/text blocks
- [ ] Docling DOCX/PPTX vs native parsers on the same files
- [ ] Speaker notes: only python-pptx extracts them (docling_pptx does not)
- [ ] DOCX core properties (`title`/`author`): only python-docx surfaces them

### Spreadsheets
- [ ] openpyxl `merged_cells.xlsx`: merged ranges listed in block metadata
- [ ] openpyxl `multi_sheet.xlsx`: one page per sheet, `row_cells` populated
- [ ] pandas same file: `dtypes` and `missing_values` present in metadata
- [ ] CSV `semicolon.csv` / `pipe_delimited.csv` / `quoted_multiline.csv`: sniffed correctly

### Email
- [ ] `multipart_attachment.eml`: ATTACHMENT block with filename/content_type/size
- [ ] `thread_reply.eml`: `is_thread_reply` true in metadata
- [ ] `archive.mbox`: 3 messages, pages 1-3

### JSON
- [ ] `nested_object.json`: KEY_VALUE blocks carry `json_path` (`$.company`, ...)
- [ ] `stream.ndjson` with `json.ndjson`: RECORD per line, `$[i]` paths
- [ ] `json.auto` on the NDJSON stream: detects/falls back correctly

### XML
- [ ] `catalog.xml` with lxml: `xpath` metadata sane (`/catalog/book[1]/title`)
- [ ] `catalog.xml` with elementtree: note the doubled path segments (known bug)
- [ ] `namespaces.xml`: prefix-qualified paths from lxml
- [ ] XPath-style expectations from `expectations.json` not enforced by benchmarks

### Code
- [ ] python_ast on `sample.py`: DEFINITION blocks with signatures and line ranges
- [ ] tree_sitter on all 7 grammars (see commands above): no crash, definitions found
- [ ] python_ast on non-Python input: clean `CorruptDocumentError`

### Images
- [ ] tesseract on `clean_text.png` / `receipt.png`: text + bbox blocks
- [ ] tesseract on `blank.png`: `EmptyDocumentError`; `corrupt.png`: `CorruptDocumentError`
- [ ] paddleocr: SKIP unless deliberately installed (do not install just for a check)

### Audio
- [ ] `speech_en.wav` (`--allow-model-download`): segment(s) with timestamps, language `en`
- [ ] `silence.wav`: 0 blocks + warning, no invented text
- [ ] `corrupt.wav`: `CorruptDocumentError`

### Video
- [ ] `testsrc.mp4`: h264, 320x240 @10fps, 1 AAC audio stream, ~3 FRAME blocks at 1s intervals
- [ ] `scene_cut.mp4` (`video.ffmpeg.scene`): frame at ~1.5s (the hard cut)
- [ ] `audio_only.mp4`: `CorruptDocumentError` (no video stream)
- [ ] `corrupt.mp4`: `CorruptDocumentError`
- [ ] `extract_audio_to_wav` + faster-whisper on the WAV: runs end to end
- [ ] `frame_dir` export: JPEGs written with frame references in block metadata
