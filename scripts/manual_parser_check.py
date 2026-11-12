#!/usr/bin/env python3
"""Manual verification CLI for the Prome RAG parser plugins.

Lists every implemented parser plugin, checks its optional dependencies,
and runs a single parser against a local file, printing a readable summary
plus representative blocks.

Exit codes:
    0  parser executed successfully (output produced and summarized)
    1  parser executed but failed (exception or invalid result)
    2  parser skipped: optional dependencies or external requirements missing
    3  parser skipped: first-use model download not permitted (pass --allow-model-download)
    4  input validation failed (file missing, not a file, or empty)
    5  requested parser is not implemented
    6  unknown parser id (see --list)
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parsers.contracts.models import BlockType, ParsedDocument  # noqa: E402


@dataclass(frozen=True)
class ParserSpec:
    id: str
    family: str
    description: str
    import_path: str | None
    class_name: str | None
    ctor_kwargs: dict[str, Any] = field(default_factory=dict)
    optional_dependencies: tuple[str, ...] = ()  # importable module names
    system_dependencies: tuple[str, ...] = ()  # executables on PATH
    model_download: bool = False  # first use may download model weights
    model_download_note: str = ""
    input_extensions: tuple[str, ...] = ()
    not_implemented: bool = False
    notes: str = ""


PARSER_SPECS: list[ParserSpec] = [
    # --- PDF ---
    ParserSpec(
        id="pdf.pymupdf",
        family="pdf",
        description="PyMuPDF: fast extraction for digitally generated PDFs (tables, headings, reading order)",
        import_path="parsers.pdf.pymupdf",
        class_name="PyMuPDFParser",
        optional_dependencies=("pymupdf",),
        input_extensions=(".pdf",),
    ),
    ParserSpec(
        id="pdf.pypdf",
        family="pdf",
        description="pypdf: pure-Python PDF text extraction (no table/heading detection)",
        import_path="parsers.pdf.pypdf",
        class_name="PyPDFParser",
        optional_dependencies=("pypdf",),
        input_extensions=(".pdf",),
    ),
    ParserSpec(
        id="pdf.docling",
        family="pdf",
        description="Docling: layout-aware PDF extraction with document structure (OCR disabled)",
        import_path="parsers.pdf.docling",
        class_name="DoclingParser",
        optional_dependencies=("docling",),
        model_download=True,
        model_download_note="Docling downloads layout/table model weights on first use (cached in ~/.cache/huggingface).",
        input_extensions=(".pdf",),
    ),
    ParserSpec(
        id="pdf.ocr",
        family="pdf",
        description="PDF OCR parsers (Tesseract/PaddleOCR-based)",
        import_path=None,
        class_name=None,
        not_implemented=True,
        notes="No OCR integration for PDFs is implemented. Docling's OCR pipeline is intentionally disabled.",
    ),
    # --- HTML ---
    ParserSpec(
        id="html.beautifulsoup",
        family="html",
        description="BeautifulSoup: full-document HTML parsing preserving DOM structure",
        import_path="parsers.html.beautifulsoup",
        class_name="BeautifulSoupParser",
        optional_dependencies=("bs4",),
        input_extensions=(".html", ".htm"),
    ),
    ParserSpec(
        id="html.trafilatura",
        family="html",
        description="Trafilatura: main-content extraction with boilerplate removal and metadata",
        import_path="parsers.html.trafilatura",
        class_name="TrafilaturaParser",
        optional_dependencies=("trafilatura",),
        input_extensions=(".html", ".htm"),
    ),
    # --- Office ---
    ParserSpec(
        id="office.docx.python_docx",
        family="office",
        description="python-docx: paragraphs, headings, tables, lists, hyperlinks, core metadata",
        import_path="parsers.office.docx.python_docx",
        class_name="PythonDocxParser",
        optional_dependencies=("docx",),
        input_extensions=(".docx",),
    ),
    ParserSpec(
        id="office.docx.docling",
        family="office",
        description="Docling DOCX: structure-preserving Word extraction via Docling",
        import_path="parsers.office.docx.docling",
        class_name="DoclingDocxParser",
        optional_dependencies=("docling",),
        model_download=True,
        model_download_note="Docling model weights are downloaded on first use (cached in ~/.cache/huggingface).",
        input_extensions=(".docx",),
    ),
    ParserSpec(
        id="office.pptx.python_pptx",
        family="office",
        description="python-pptx: slides, shapes, tables, notes, images",
        import_path="parsers.office.pptx.python_pptx",
        class_name="PythonPptxParser",
        optional_dependencies=("pptx",),
        input_extensions=(".pptx",),
    ),
    ParserSpec(
        id="office.pptx.docling",
        family="office",
        description="Docling PPTX: structure-preserving PowerPoint extraction via Docling",
        import_path="parsers.office.pptx.docling",
        class_name="DoclingPptxParser",
        optional_dependencies=("docling",),
        model_download=True,
        model_download_note="Docling model weights are downloaded on first use (cached in ~/.cache/huggingface).",
        input_extensions=(".pptx",),
    ),
    # --- Spreadsheets ---
    ParserSpec(
        id="spreadsheets.xlsx.openpyxl",
        family="spreadsheets",
        description="openpyxl: formulas, merged cells, cell types, per-sheet tables, workbook metadata",
        import_path="parsers.spreadsheets.xlsx.openpyxl",
        class_name="OpenpyxlParser",
        optional_dependencies=("openpyxl",),
        input_extensions=(".xlsx",),
    ),
    ParserSpec(
        id="spreadsheets.xlsx.pandas",
        family="spreadsheets",
        description="pandas: DataFrame-oriented extraction with dtypes and missing-value stats",
        import_path="parsers.spreadsheets.xlsx.pandas",
        class_name="PandasXlsxParser",
        optional_dependencies=("pandas", "openpyxl"),
        input_extensions=(".xlsx",),
    ),
    ParserSpec(
        id="spreadsheets.csv",
        family="spreadsheets",
        description="stdlib csv: delimiter sniffing (comma/semicolon/tab/pipe), encoding fallbacks, streaming",
        import_path="parsers.spreadsheets.csv",
        class_name="CsvParser",
        optional_dependencies=(),
        input_extensions=(".csv", ".tsv", ".txt"),
    ),
    # --- Email ---
    ParserSpec(
        id="email.eml",
        family="email",
        description="stdlib email: RFC 822/MIME headers, bodies, attachments, thread relations",
        import_path="parsers.email.eml",
        class_name="EmlParser",
        optional_dependencies=(),
        input_extensions=(".eml",),
    ),
    ParserSpec(
        id="email.mbox",
        family="email",
        description="stdlib mailbox: multi-message MBOX archives (one page per message)",
        import_path="parsers.email.mbox",
        class_name="MboxParser",
        optional_dependencies=(),
        input_extensions=(".mbox",),
    ),
    # --- JSON ---
    ParserSpec(
        id="json.auto",
        family="json",
        description="stdlib json: auto-detect JSON vs NDJSON/JSONL",
        import_path="parsers.json",
        class_name="JsonParser",
        ctor_kwargs={"mode": "auto"},
        optional_dependencies=(),
        input_extensions=(".json", ".ndjson", ".jsonl"),
    ),
    ParserSpec(
        id="json.json",
        family="json",
        description="stdlib json: strict JSON documents with JSON-path preservation",
        import_path="parsers.json",
        class_name="JsonParser",
        ctor_kwargs={"mode": "json"},
        optional_dependencies=(),
        input_extensions=(".json",),
    ),
    ParserSpec(
        id="json.ndjson",
        family="json",
        description="stdlib json: NDJSON/JSONL streams (one RECORD block per line)",
        import_path="parsers.json",
        class_name="JsonParser",
        ctor_kwargs={"mode": "ndjson"},
        optional_dependencies=(),
        input_extensions=(".ndjson", ".jsonl"),
    ),
    # --- XML ---
    ParserSpec(
        id="xml.elementtree",
        family="xml",
        description="stdlib ElementTree: path-preserved XML hierarchy extraction",
        import_path="parsers.xml.elementtree",
        class_name="ElementTreeXmlParser",
        optional_dependencies=(),
        input_extensions=(".xml",),
    ),
    ParserSpec(
        id="xml.lxml",
        family="xml",
        description="lxml: path-preserved XML hierarchy extraction (XXE-safe, no external entities)",
        import_path="parsers.xml.lxml",
        class_name="LxmlXmlParser",
        optional_dependencies=("lxml",),
        input_extensions=(".xml",),
    ),
    # --- Code ---
    ParserSpec(
        id="code.python_ast",
        family="code",
        description="stdlib ast: Python definitions, signatures, docstrings, decorators, positions",
        import_path="parsers.code.python_ast",
        class_name="PythonAstParser",
        optional_dependencies=(),
        input_extensions=(".py",),
    ),
    ParserSpec(
        id="code.tree_sitter",
        family="code",
        description="tree-sitter: multi-language definitions/imports (language inferred from extension)",
        import_path="parsers.code.tree_sitter",
        class_name="TreeSitterParser",
        optional_dependencies=("tree_sitter",),
        input_extensions=(".py", ".js", ".mjs", ".jsx", ".ts", ".tsx", ".java", ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".go"),
        notes="Grammar packages are required per language; see code.tree_sitter.<lang> entries.",
    ),
]

_TREE_SITTER_LANGS = {
    "python": (("tree_sitter", "tree_sitter_python"), (".py",)),
    "javascript": (("tree_sitter", "tree_sitter_javascript"), (".js", ".mjs", ".jsx")),
    "typescript": (("tree_sitter", "tree_sitter_typescript"), (".ts", ".tsx")),
    "java": (("tree_sitter", "tree_sitter_java"), (".java",)),
    "c": (("tree_sitter", "tree_sitter_c"), (".c", ".h")),
    "cpp": (("tree_sitter", "tree_sitter_cpp"), (".cpp", ".cc", ".cxx", ".hpp")),
    "go": (("tree_sitter", "tree_sitter_go"), (".go",)),
}
for _lang, (_deps, _exts) in _TREE_SITTER_LANGS.items():
    PARSER_SPECS.append(
        ParserSpec(
            id=f"code.tree_sitter.{_lang}",
            family="code",
            description=f"tree-sitter with the '{_lang}' grammar explicitly selected",
            import_path="parsers.code.tree_sitter",
            class_name="TreeSitterParser",
            ctor_kwargs={"language": _lang},
            optional_dependencies=_deps,
            input_extensions=_exts,
        )
    )

PARSER_SPECS += [
    # --- Images ---
    ParserSpec(
        id="images.tesseract",
        family="images",
        description="Tesseract OCR: text, bounding boxes and confidence scores from raster images",
        import_path="parsers.images.tesseract",
        class_name="TesseractParser",
        optional_dependencies=("pytesseract", "PIL"),
        system_dependencies=("tesseract",),
        input_extensions=(".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"),
        notes="Requires the system tesseract binary (e.g. brew install tesseract).",
    ),
    ParserSpec(
        id="images.paddleocr",
        family="images",
        description="PaddleOCR: deep-learning OCR with bounding polygons and reading order",
        import_path="parsers.images.paddleocr",
        class_name="PaddleOCRParser",
        optional_dependencies=("paddleocr",),
        model_download=True,
        model_download_note="PaddleOCR downloads detection/recognition model weights on first run.",
        input_extensions=(".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"),
        notes="Heavy optional dependency; intentionally not part of the default install set.",
    ),
    # --- Audio ---
    ParserSpec(
        id="audio.faster_whisper",
        family="audio",
        description="faster-whisper: transcription into SEGMENT blocks with timestamps (no diarization)",
        import_path="parsers.audio.faster_whisper",
        class_name="FasterWhisperParser",
        optional_dependencies=("faster_whisper", "av", "numpy"),
        model_download=True,
        model_download_note="faster-whisper downloads the Whisper model on first use (default: 'tiny', cached in ~/.cache/huggingface).",
        input_extensions=(".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".aiff", ".aif", ".caf", ".webm"),
    ),
    # --- Video ---
    ParserSpec(
        id="video.ffmpeg",
        family="video",
        description="PyAV/ffmpeg: container metadata + selective FRAME blocks (interval strategy, 1s)",
        import_path="parsers.video.ffmpeg",
        class_name="FFmpegVideoParser",
        optional_dependencies=("av", "numpy"),
        input_extensions=(".mp4", ".m4v", ".mkv", ".webm", ".mov", ".avi", ".mpg", ".mpeg", ".ts"),
        notes="PyAV wheels bundle ffmpeg; a system ffmpeg binary is not required. extract_audio_to_wav() is a separate method, see MANUAL_PARSER_VERIFICATION.md.",
    ),
    ParserSpec(
        id="video.ffmpeg.scene",
        family="video",
        description="PyAV/ffmpeg: FRAME blocks selected by scene-cut detection (threshold 0.08)",
        import_path="parsers.video.ffmpeg",
        class_name="FFmpegVideoParser",
        ctor_kwargs={"frame_strategy": "scene"},
        optional_dependencies=("av", "numpy"),
        input_extensions=(".mp4", ".m4v", ".mkv", ".webm", ".mov", ".avi", ".mpg", ".mpeg", ".ts"),
    ),
]

SPECS_BY_ID = {spec.id: spec for spec in PARSER_SPECS}


def check_dependencies(spec: ParserSpec) -> list[str]:
    """Return a list of missing optional/system dependency descriptions."""
    missing: list[str] = []
    for module in spec.optional_dependencies:
        try:
            importlib.import_module(module)
        except Exception:
            missing.append(f"python module '{module}'")
    for binary in spec.system_dependencies:
        if shutil.which(binary) is None:
            missing.append(f"system executable '{binary}' (not on PATH)")
    return missing


def load_parser_class(spec: ParserSpec):
    module = importlib.import_module(spec.import_path)
    return getattr(module, spec.class_name)


def list_parsers() -> int:
    print(f"Available parser plugins ({len(PARSER_SPECS)} entries):\n")
    current_family = None
    for spec in PARSER_SPECS:
        if spec.family != current_family:
            current_family = spec.family
            print(f"== {current_family} ==")
        if spec.not_implemented:
            status = "NOT IMPLEMENTED"
        else:
            missing = check_dependencies(spec)
            status = "available" if not missing else f"missing deps: {', '.join(missing)}"
        print(f"  {spec.id:32} {status}")
        print(f"      {spec.description}")
        if spec.model_download:
            print(f"      model download on first use: YES — {spec.model_download_note}")
        if spec.notes:
            print(f"      notes: {spec.notes}")
    print()
    print("Status legend: 'available' = dependencies importable; it does NOT prove the parser works.")
    print("Run: python scripts/manual_parser_check.py --parser <id> --input <file>")
    return 0


def _block_location(block) -> str:
    parts = []
    if block.page_number is not None:
        parts.append(f"page={block.page_number}")
    if block.bbox is not None:
        b = block.bbox
        parts.append(f"bbox=({b.x0:.1f},{b.y0:.1f},{b.x1:.1f},{b.y1:.1f})")
    return " ".join(parts)


def _preview(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def format_block(block, preview_chars: int) -> list[str]:
    bt = block.block_type.value if isinstance(block.block_type, BlockType) else str(block.block_type)
    meta = block.metadata or {}
    loc = _block_location(block)
    header = f"[{bt}] {loc}".rstrip()
    lines = [header]

    def kv(label: str, value: Any) -> None:
        if value is not None and value != "":
            lines.append(f"    {label}: {value}")

    if bt == "table":
        sheet = meta.get("sheet_name")
        rows, cols = meta.get("rows"), meta.get("cols")
        dims = f"{rows}x{cols}" if rows and cols else None
        kv("sheet", sheet)
        kv("dimensions", dims)
        headers = meta.get("headers")
        if headers:
            kv("headers", " | ".join(str(h) for h in headers))
        row_cells = meta.get("row_cells")
        if isinstance(row_cells, list) and row_cells:
            for i, row in enumerate(row_cells[:4]):
                cells = " | ".join(_preview(str(c), 40) for c in row)
                lines.append(f"    row {i}: {cells}")
            if len(row_cells) > 4:
                lines.append(f"    … ({len(row_cells) - 4} more rows)")
        elif block.content:
            for i, content_line in enumerate(block.content.splitlines()[:6]):
                lines.append(f"    row {i}: {_preview(content_line, preview_chars)}")
            if len(block.content.splitlines()) > 6:
                lines.append(f"    … ({len(block.content.splitlines()) - 6} more lines)")
    elif bt == "record":
        try:
            payload = json.loads(block.content)
        except Exception:
            payload = None
        kv("json_path", meta.get("json_path"))
        if isinstance(payload, dict):
            kv("key_count", len(payload))
            for i, (key, value) in enumerate(payload.items()):
                if i >= 6:
                    lines.append(f"    … ({len(payload) - 6} more keys)")
                    break
                lines.append(f"    {key}: {_preview(json.dumps(value, ensure_ascii=False, default=str), preview_chars)}")
        elif isinstance(payload, list):
            kv("item_count", len(payload))
            for i, item in enumerate(payload[:6]):
                lines.append(f"    [{i}]: {_preview(json.dumps(item, ensure_ascii=False, default=str), preview_chars)}")
            if len(payload) > 6:
                lines.append(f"    … ({len(payload) - 6} more items)")
        else:
            lines.append(f"    content: {_preview(block.content, preview_chars)}")
    elif bt == "key_value":
        kv("path", meta.get("json_path") or meta.get("xpath"))
        kv("key", meta.get("key"))
        kv("value", _preview(str(meta.get("value")), preview_chars))
        kv("data_type", meta.get("data_type"))
    elif bt == "definition":
        kind, name = meta.get("kind"), meta.get("name")
        sig = meta.get("signature") or ""
        if sig and name and not sig.startswith(str(name)):
            shown = f"{kind or 'def'} {name}{sig}"
        elif sig:
            shown = f"{kind or 'def'} {sig}"  # parser signature already includes the name
        else:
            shown = f"{kind or 'def'} {name or '?'}"
        kv("definition", shown)
        kv("parent_class", meta.get("parent_class"))
        kv("node_type", meta.get("node_type"))
        start, end = meta.get("start_line"), meta.get("end_line")
        if start is not None:
            kv("lines", f"{start}–{end}")
        kv("docstring", _preview(str(meta.get("docstring") or ""), preview_chars) if meta.get("docstring") else None)
    elif bt == "segment":
        kv("timestamps", f"{meta.get('start')}s – {meta.get('end')}s (duration {meta.get('duration')}s)")
        kv("language", meta.get("language"))
        kv("avg_logprob", meta.get("avg_logprob"))
        kv("no_speech_prob", meta.get("no_speech_prob"))
        lines.append(f"    text: {_preview(block.content, preview_chars)}")
    elif bt == "frame":
        kv("frame_number", meta.get("frame_number"))
        kv("timestamp", f"{meta.get('timestamp')}s")
        kv("dimensions", f"{meta.get('width')}x{meta.get('height')}")
        kv("strategy", meta.get("strategy"))
        kv("exported_to", meta.get("frame_path") or "(not saved; pass frame_dir=... to save JPEGs)")
    elif bt == "attachment":
        kv("filename", meta.get("filename"))
        kv("content_type", meta.get("content_type"))
        kv("size_bytes", meta.get("size_bytes"))
        kv("content_disposition", meta.get("content_disposition"))
    elif bt == "image":
        dims = f"{meta.get('width')}x{meta.get('height')}" if meta.get("width") else None
        kv("dimensions", dims)
        kv("ext", meta.get("ext"))
        kv("alt", meta.get("alt"))
        if block.content:
            lines.append(f"    content: {_preview(block.content, preview_chars)}")
    elif bt == "link":
        kv("href", meta.get("href"))
        lines.append(f"    text: {_preview(block.content, preview_chars)}")
    elif bt == "list":
        lines.append(f"    content: {_preview(block.content, preview_chars)}")
    else:
        lines.append(f"    content: {_preview(block.content, preview_chars)}")
    return lines


def summarize(doc: ParsedDocument, source: Path, duration: float, args) -> str:
    all_blocks = list(doc.blocks) + [b for p in doc.pages for b in p.blocks]
    distribution: dict[str, int] = {}
    total_text = 0
    for b in all_blocks:
        key = b.block_type.value if isinstance(b.block_type, BlockType) else str(b.block_type)
        distribution[key] = distribution.get(key, 0) + 1
        total_text += len(b.content or "")

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"Parser:         {doc.parser_name}")
    lines.append(f"Parser version: {doc.parser_version}")
    lines.append(f"Input:          {source}")
    lines.append(f"Status:         PASS (parser executed and returned a ParsedDocument)")
    lines.append(f"Parse duration: {duration:.2f}s")
    lines.append(f"Pages:          {doc.page_count if doc.pages else 'n/a (pageless document)'}")
    lines.append(f"Blocks:         {len(all_blocks)}")
    if distribution:
        dist = ", ".join(f"{k}={v}" for k, v in sorted(distribution.items(), key=lambda kv: (-kv[1], kv[0])))
        lines.append(f"Block types:    {dist}")
    lines.append(f"Total text:     {total_text} chars")
    lines.append("Metadata:")
    display_meta = {}
    for key, value in (doc.metadata or {}).items():
        try:
            encoded = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            encoded = str(value)
        if len(encoded) > 400:
            display_meta[key] = f"<{len(encoded)} chars — truncated in summary; use --output for full JSON>"
        else:
            display_meta[key] = value
    try:
        meta_json = json.dumps(display_meta, indent=2, ensure_ascii=False, default=str)
    except Exception:
        meta_json = str(display_meta)
    for meta_line in meta_json.splitlines():
        lines.append(f"    {meta_line}")
    if doc.warnings:
        lines.append("Warnings:")
        for w in doc.warnings:
            lines.append(f"    - {w}")
    else:
        lines.append("Warnings:       none")

    if args.show_blocks > 0:
        lines.append(f"\nRepresentative blocks (showing up to {args.show_blocks} of {len(all_blocks)}):")
        for i, block in enumerate(all_blocks[: args.show_blocks], start=1):
            lines.append(f"\n  Block #{i}")
            for line in format_block(block, args.preview_chars):
                lines.append(f"  {line}")
        remaining = len(all_blocks) - args.show_blocks
        if remaining > 0:
            lines.append(f"\n  … and {remaining} more block(s) (use --show-blocks to display more)")
    lines.append("=" * 72)
    return "\n".join(lines)


def run_parser(args) -> int:
    source = Path(args.input).expanduser()
    if not source.exists():
        print(f"ERROR: input file does not exist: {source}", file=sys.stderr)
        return 4
    if not source.is_file():
        print(f"ERROR: input path is not a file: {source}", file=sys.stderr)
        return 4
    if source.stat().st_size == 0:
        print(f"ERROR: input file is empty (0 bytes): {source}", file=sys.stderr)
        return 4

    spec = SPECS_BY_ID.get(args.parser)
    if spec is None:
        print(f"ERROR: unknown parser id '{args.parser}'. See --list for valid ids.", file=sys.stderr)
        return 6
    if spec.not_implemented:
        print(f"NOT IMPLEMENTED: {spec.id} — {spec.description}")
        if spec.notes:
            print(f"  {spec.notes}")
        return 5

    missing = check_dependencies(spec)
    if missing:
        print(f"SKIP: parser '{spec.id}' cannot execute — missing requirements:")
        for item in missing:
            print(f"    - {item}")
        if spec.optional_dependencies:
            mods = " ".join(spec.optional_dependencies)
            print(f"  Install the relevant extra, e.g.: pip install 'prome-rag-plugins[<extra>]'")
            print(f"  (importable modules required: {mods})")
        return 2

    if spec.model_download and not args.allow_model_download:
        print(f"SKIP: parser '{spec.id}' may download model weights on first use.")
        print(f"  {spec.model_download_note}")
        print("  Re-run with --allow-model-download to explicitly permit this.")
        print("  If the model is already cached (e.g. ~/.cache/huggingface), no download occurs,")
        print("  but the flag is still required by this tool's safety policy.")
        return 3

    try:
        parser_cls = load_parser_class(spec)
        parser = parser_cls(**spec.ctor_kwargs)
    except Exception as exc:
        print(f"FAIL: could not instantiate parser '{spec.id}' ({spec.import_path}.{spec.class_name}):")
        print(f"  {type(exc).__name__}: {exc}")
        if args.traceback:
            traceback.print_exc()
        return 1

    started = time.perf_counter()
    try:
        doc = parser.parse(str(source))
    except Exception as exc:
        duration = time.perf_counter() - started
        print(f"FAIL: parser '{spec.id}' raised {type(exc).__name__} after {duration:.2f}s:")
        print(f"  {exc}")
        if args.traceback:
            traceback.print_exc()
        return 1
    duration = time.perf_counter() - started

    if not isinstance(doc, ParsedDocument):
        print(f"FAIL: parser '{spec.id}' returned {type(doc).__name__}, expected ParsedDocument.")
        return 1

    print(summarize(doc, source, duration, args))
    if args.output:
        out_path = Path(args.output).expanduser()
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(doc.as_dict(), indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            print(f"JSON written to: {out_path}")
        except Exception as exc:
            print(f"ERROR: could not write JSON output to {out_path}: {exc}", file=sys.stderr)
            return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manually verify Prome RAG parser plugins on a local file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python scripts/manual_parser_check.py --list\n"
               "  python scripts/manual_parser_check.py --parser pdf.pymupdf --input /path/to/document.pdf\n"
               "  python scripts/manual_parser_check.py --parser audio.faster_whisper --input /path/to/audio.wav --allow-model-download\n",
    )
    parser.add_argument("--list", action="store_true", help="list all parser plugins and dependency status")
    parser.add_argument("--parser", metavar="ID", help="parser id (see --list)")
    parser.add_argument("--input", metavar="PATH", help="local input file to parse")
    parser.add_argument("--output", metavar="PATH", help="write the full ParsedDocument as JSON to this path")
    parser.add_argument("--show-blocks", type=int, default=10, metavar="N",
                        help="maximum number of representative blocks to display (default: 10; 0 = none)")
    parser.add_argument("--preview-chars", type=int, default=200, metavar="N",
                        help="maximum characters per text preview (default: 200)")
    parser.add_argument("--allow-model-download", action="store_true",
                        help="explicitly permit a parser's documented first-use model download")
    parser.add_argument("--traceback", action="store_true",
                        help="print a full traceback when the parser fails")
    args = parser.parse_args()

    if args.list:
        return list_parsers()
    if not args.parser:
        print("ERROR: --parser is required (or use --list).", file=sys.stderr)
        return 6
    if not args.input:
        print("ERROR: --input is required when --parser is given.", file=sys.stderr)
        return 4
    return run_parser(args)


if __name__ == "__main__":
    sys.exit(main())
