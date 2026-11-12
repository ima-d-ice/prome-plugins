"""Docling-based PPTX parser: layout-aware PowerPoint presentation extraction."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from parsers.contracts import (
    BlockType,
    BoundingBox,
    CorruptDocumentError,
    EmptyDocumentError,
    ParsedBlock,
    ParsedDocument,
    ParsedPage,
    ParserDependencyError,
    ParserError,
)
from parsers.contracts.validation import validate_source

logger = logging.getLogger(__name__)

EMU_PER_PT = 12700.0

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


def _to_bbox(item: Any) -> BoundingBox | None:
    prov = getattr(item, "prov", None)
    if not prov:
        return None
    raw_bbox = getattr(prov[0], "bbox", None)
    if raw_bbox is None:
        return None
    try:
        left = float(raw_bbox.l) / EMU_PER_PT
        right = float(raw_bbox.r) / EMU_PER_PT
        top = float(raw_bbox.t) / EMU_PER_PT
        bottom = float(raw_bbox.b) / EMU_PER_PT
        y0, y1 = min(top, bottom), max(top, bottom)
        return BoundingBox(x0=left, y0=y0, x1=right, y1=y1)
    except Exception:
        return None


class DoclingPptxParser:
    """Extract structured PPTX content using Docling."""

    name = "docling_pptx"

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
            msg = str(exc).lower()
            if "0 pages" in msg or "empty" in msg:
                raise EmptyDocumentError(f"PPTX '{path}' contains no slides: {exc}") from exc
            if "could not load document" in msg or "conversion failed" in msg or "badzip" in msg:
                raise CorruptDocumentError(f"Could not open '{path}' as a PPTX file: {exc}") from exc
            raise ParserError(f"Docling failed to convert PPTX '{path}': {exc}") from exc

        doc = getattr(result, "document", None)
        if doc is None or not getattr(doc, "pages", None):
            raise EmptyDocumentError(f"PPTX '{path}' produced no slides.")

        # Prepare pages by slide number
        pages_by_num: dict[int, ParsedPage] = {}
        for page_no, page_obj in doc.pages.items():
            size = getattr(page_obj, "size", None)
            w = float(size.width) / EMU_PER_PT if size and hasattr(size, "width") else None
            h = float(size.height) / EMU_PER_PT if size and hasattr(size, "height") else None
            pages_by_num[int(page_no)] = ParsedPage(page_number=int(page_no), width=w, height=h, blocks=[])

        total_blocks = 0
        doc_order = 0

        for item, level in doc.iterate_items():
            b_type = _block_type(item)
            content = ""
            meta: dict[str, Any] = {
                "document_order": doc_order,
                "docling_level": level,
                "docling_label": _label_value(item),
            }

            p_no = 1
            prov = getattr(item, "prov", None)
            if prov and getattr(prov[0], "page_no", None) is not None:
                p_no = int(prov[0].page_no)

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
                    meta["heading_level"] = 1

            if not content and b_type != BlockType.IMAGE:
                continue

            doc_order += 1
            total_blocks += 1
            block = ParsedBlock(
                block_type=b_type,
                content=content,
                page_number=p_no,
                bbox=_to_bbox(item),
                metadata=meta,
            )

            if p_no not in pages_by_num:
                pages_by_num[p_no] = ParsedPage(page_number=p_no, blocks=[])
            pages_by_num[p_no].blocks.append(block)

        if total_blocks == 0:
            raise EmptyDocumentError(f"PPTX '{path}' contains no extractable content.")

        sorted_pages = [pages_by_num[k] for k in sorted(pages_by_num.keys())]

        metadata: dict[str, Any] = {
            "docling_version": str(getattr(docling_mod, "__version__", "unknown")),
            "slide_count": len(sorted_pages),
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
            pages=sorted_pages,
            blocks=[],
            warnings=[],
        )
