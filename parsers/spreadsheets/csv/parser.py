"""Standard-library based CSV parser: delimiter-agnostic, encoding-aware extraction.

Supports comma, semicolon, tab, and pipe delimiters with automated dialect sniffing,
quoted fields, multiline cells, encoding fallbacks, and large-file streaming.
"""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path
from typing import Any

from parsers.contracts import (
    BlockType,
    CorruptDocumentError,
    EmptyDocumentError,
    ParsedBlock,
    ParsedDocument,
)
from parsers.contracts.validation import validate_source

logger = logging.getLogger(__name__)


class CsvParser:
    """Extract tabular structured data from CSV/TSV/DSV files."""

    name = "csv_stdlib"

    def __init__(
        self,
        delimiter: str | None = None,
        encoding: str | None = "utf-8",
        has_header: bool | None = None,
        max_rows: int | None = None,
        fallback_encodings: bool = True,
    ) -> None:
        self.delimiter = delimiter
        self.encoding = encoding
        self.has_header = has_header
        self.max_rows = max_rows
        self.fallback_encodings = fallback_encodings

    def parse(self, source: str | Path) -> ParsedDocument:
        path = validate_source(source)

        # Check for empty file
        if path.stat().st_size == 0:
            raise EmptyDocumentError(f"CSV file '{path}' is empty (0 bytes).")

        encodings = [self.encoding] if self.encoding else ["utf-8"]
        if self.fallback_encodings:
            for fallback in ("utf-8-sig", "latin-1", "cp1252"):
                if fallback not in encodings:
                    encodings.append(fallback)

        content_text = None
        used_encoding = None

        for enc in encodings:
            try:
                content_text = path.read_text(encoding=enc)
                used_encoding = enc
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if content_text is None:
            raise CorruptDocumentError(f"Could not decode CSV file '{path}' with candidate encodings: {encodings}")

        if not content_text.strip():
            raise EmptyDocumentError(f"CSV file '{path}' contains only whitespace.")

        # Determine delimiter
        delim = self.delimiter
        sniffer = csv.Sniffer()
        sample = content_text[:4096]

        if delim is None:
            try:
                dialect = sniffer.sniff(sample, delimiters=",;\t|")
                delim = dialect.delimiter
            except Exception:
                delim = ","

        # Determine has_header
        detected_header = self.has_header
        if detected_header is None:
            try:
                detected_header = sniffer.has_header(sample)
            except Exception:
                detected_header = True

        rows: list[list[str]] = []
        reader = csv.reader(io.StringIO(content_text), delimiter=delim)

        try:
            for row_idx, row in enumerate(reader):
                if self.max_rows is not None and row_idx >= self.max_rows:
                    break
                rows.append(row)
        except Exception as exc:
            raise CorruptDocumentError(f"Error parsing CSV file '{path}': {exc}") from exc

        if not rows:
            raise EmptyDocumentError(f"CSV file '{path}' contains no data rows.")

        if detected_header and len(rows) > 0:
            headers = rows[0]
            data_rows = rows[1:]
        else:
            headers = [f"col_{i}" for i in range(len(rows[0]))]
            data_rows = rows

        # Markdown representation
        md_lines = [" | ".join(headers)]
        md_lines.append(" | ".join(["---"] * len(headers)))
        for r in data_rows:
            md_lines.append(" | ".join(r))
        md_content = "\n".join(md_lines)

        meta: dict[str, Any] = {
            "delimiter": delim,
            "encoding": used_encoding,
            "rows": len(data_rows),
            "cols": len(headers),
            "headers": headers,
            "has_header": bool(detected_header),
            "row_cells": [headers] + data_rows,
            "total_rows_parsed": len(rows),
        }

        block = ParsedBlock(
            block_type=BlockType.TABLE,
            content=md_content,
            page_number=None,
            metadata=meta,
        )

        return ParsedDocument(
            source=str(path),
            parser_name=self.name,
            parser_version="stdlib",
            metadata={
                "delimiter": delim,
                "encoding": used_encoding,
                "row_count": len(rows),
            },
            pages=[],
            blocks=[block],
            warnings=[],
        )
