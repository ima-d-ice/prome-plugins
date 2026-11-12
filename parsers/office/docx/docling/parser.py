"""Docling-based DOCX parser: structure-preserving Word document extraction."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from parsers.contracts import (
    BlockType,
    CorruptDocumentError,
    EmptyDocumentError,
    ParsedBlock,
    ParsedDocument,
    ParserDependencyError,
    ParserError,
)
from parsers.contracts.validation import validate_source

logger = logging.getLogger(__name__)

_LABEL_MAP = {
    "title": BlockType.HEADING,
    "section_header": BlockType.HEADING,
    "subtitle": BlockType.HEADING,
    "text": BlockType.TEXT,
    "paragraph": BlockType.TEXT,
    "caption": BlockType.TEXT,
    "footnote": BlockType.TEXT,
    "code": BlockType.CODE,
    "list_item": BlockType.LIST,
    "table": BlockType.TABLE,
    "picture": BlockType.IMAGE,
    "chart": BlockType.IMAGE,
}


def _load_docling():
    try:
        from docling.document_converter import DocumentConverter
        import docling

        return docling, DocumentConverter
    except ImportError as exc:
        raise ParserDependencyError(
            "Docling is not installed. Install it with:\n"
            "  pip install 'prome-rag-plugins[docling]'\n"
            "or:\n"
            "  pip install docling"
        ) from exc


def _label_value(item: Any) -> str:
    label = getattr(item, "label", None)
    value = getattr(label, "value", label)
    return str(value).lower() if value is not None else ""


def _block_type(item: Any) -> BlockType:
    return _LABEL_MAP.get(_label_value(item), BlockType.OTHER)


class DoclingDocxParser:
    """Extract structured DOCX content using Docling."""

    name = "docling_docx"

    def __init__(self) -> None:
        self._converter = None
        self._lock = threading.Lock()

    def _get_converter(self):
        if self._converter is not None:
            return self._converter
        with self._lock:
            if self._converter is not None:
                return self._converter
            _, DocumentConverter = _load_docling()
            try:
                converter = DocumentConverter()
            except Exception as exc:
                raise ParserError(
                    f"Docling converter initialization failed: {type(exc).__name__}: {exc}"
                ) from exc
            self._converter = converter
            return self._converter

    def parse(self, source: str | Path) -> ParsedDocument:
        docling_mod, _ = _load_docling()
        path = validate_source(source)
        converter = self._get_converter()

        try:
            result = converter.convert(str(path))
        except Exception as exc:
            # Check for corrupt document or conversion error
            err_name = type(exc).__name__
            if "ConversionError" in err_name or "BadZipFile" in err_name or "PackageNotFoundError" in err_name:
                raise CorruptDocumentError(f"Could not open '{path}' as a DOCX file: {exc}") from exc
            raise ParserError(f"Docling failed to convert DOCX '{path}': {exc}") from exc

        doc = getattr(result, "document", None)
        if doc is None:
            raise EmptyDocumentError(f"DOCX '{path}' produced no document structure.")

        blocks: list[ParsedBlock] = []
        doc_order = 0

        for item, level in doc.iterate_items():
            b_type = _block_type(item)
            content = ""
            meta: dict[str, Any] = {
                "document_order": doc_order,
                "docling_level": level,
                "docling_label": _label_value(item),
            }

            if b_type == BlockType.TABLE:
                try:
                    content = item.export_to_markdown(doc).strip()
                except Exception:
                    content = str(getattr(item, "text", "") or "").strip()
                data = getattr(item, "data", None)
                if data is not None:
                    meta["rows"] = getattr(data, "num_rows", None)
                    meta["cols"] = getattr(data, "num_cols", None)
                    grid = getattr(data, "grid", None)
                    if grid:
                        meta["headers"] = [getattr(cell, "text", str(cell)) for cell in grid[0]]
                        meta["row_cells"] = [
                            [getattr(cell, "text", str(cell)) for cell in row]
                            for row in grid
                        ]
            elif b_type == BlockType.IMAGE:
                content = f"[image {getattr(item, 'self_ref', '')}]"
            else:
                content = str(getattr(item, "text", "") or "").strip()
                if b_type == BlockType.HEADING:
                    meta["heading_level"] = max(1, level - 1)

            if not content and b_type != BlockType.IMAGE:
                continue

            doc_order += 1
            blocks.append(
                ParsedBlock(
                    block_type=b_type,
                    content=content,
                    page_number=None,
                    metadata=meta,
                )
            )

        if not blocks:
            raise EmptyDocumentError(f"DOCX '{path}' contains no extractable content.")

        metadata: dict[str, Any] = {
            "docling_version": str(getattr(docling_mod, "__version__", "unknown")),
        }
        origin = getattr(doc, "origin", None)
        if origin is not None:
            for attr in ("filename", "mimetype", "binary_hash"):
                val = getattr(origin, attr, None)
                if val:
                    metadata[attr] = str(val)

        return ParsedDocument(
            source=str(path),
            parser_name=self.name,
            parser_version=str(getattr(docling_mod, "__version__", "unknown")),
            metadata=metadata,
            pages=[],
            blocks=blocks,
            warnings=[],
        )
