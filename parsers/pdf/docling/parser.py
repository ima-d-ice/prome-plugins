"""Docling-based PDF parser: layout-aware extraction with document structure."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from parsers.contracts import (
    BlockType,
    BoundingBox,
    CorruptDocumentError,
    EmptyDocumentError,
    EncryptedDocumentError,
    ParsedBlock,
    ParsedDocument,
    ParsedPage,
    ParserDependencyError,
    ParserError,
)
from parsers.contracts.validation import validate_source

logger = logging.getLogger(__name__)

_LABEL_MAP = {
    "title": BlockType.HEADING,
    "section_header": BlockType.HEADING,
    "field_heading": BlockType.HEADING,
    "text": BlockType.TEXT,
    "paragraph": BlockType.TEXT,
    "caption": BlockType.TEXT,
    "footnote": BlockType.TEXT,
    "document_index": BlockType.TEXT,
    "reference": BlockType.TEXT,
    "code": BlockType.TEXT,
    "list_item": BlockType.LIST,
    "table": BlockType.TABLE,
    "picture": BlockType.IMAGE,
    "chart": BlockType.IMAGE,
}


def _load_docling():
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel import base_models
        import docling

        return docling, DocumentConverter, PdfFormatOption, PdfPipelineOptions, base_models
    except ImportError as exc:
        raise ParserDependencyError(
            "Docling is not installed. Install it with:\n"
            "  pip install 'prome-rag-plugins[docling]'\n"
            "or:\n"
            "  pip install docling\n"
            "Note: Docling is a large dependency and downloads model weights on first use."
        ) from exc


def _label_value(item: Any) -> str:
    label = getattr(item, "label", None)
    value = getattr(label, "value", label)
    return str(value).lower() if value is not None else ""


def _block_type(item: Any) -> BlockType:
    return _LABEL_MAP.get(_label_value(item), BlockType.OTHER)


def _cell_bbox_origin_is_bottom_left(bbox: Any) -> bool:
    origin = getattr(bbox, "coord_origin", None) or getattr(bbox, "origin", None)
    return origin is not None and "bottom" in str(origin).lower()


def _to_top_left(bbox: Any, page_height: float | None) -> BoundingBox | None:
    try:
        left, right = float(bbox.l), float(bbox.r)
        top, bottom = float(bbox.t), float(bbox.b)
    except (TypeError, ValueError, AttributeError):
        return None
    if _cell_bbox_origin_is_bottom_left(bbox) and page_height:
        return BoundingBox(x0=left, y0=page_height - top, x1=right, y1=page_height - bottom)
    # For top-left origin, "t" is the smaller y and "b" the larger one.
    y0, y1 = min(top, bottom), max(top, bottom)
    return BoundingBox(x0=left, y0=y0, x1=right, y1=y1)


def _item_content(item: Any, doc: Any) -> str:
    block_type = _block_type(item)
    if block_type is BlockType.TABLE:
        try:
            return item.export_to_markdown(doc).strip()
        except Exception as exc:
            logger.debug("Table markdown export failed, falling back to data scrape: %s", exc)
            data = getattr(item, "data", None)
            return str(data) if data is not None else ""
    if block_type is BlockType.IMAGE:
        return ""
    return str(getattr(item, "text", "") or "").strip()


def _document_metadata(result: Any, doc: Any, docling_version: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    origin = getattr(doc, "origin", None)
    if origin is not None:
        for attr in ("filename", "mimetype", "binary_hash"):
            value = getattr(origin, attr, None)
            if value:
                metadata[attr] = str(value)
    metadata["schema_name"] = str(getattr(doc, "schema_name", ""))
    metadata["docling_version"] = docling_version
    return metadata


class DoclingParser:
    name = "docling"

    def __init__(self) -> None:
        # The converter is created lazily on first use and reused for the
        # lifetime of this parser instance. Recreating it per parse reloads
        # the model weights every time. Each parser instance owns its own
        # converter; nothing is shared at module level. The lock guards
        # converter initialization and use against concurrent callers.
        self._converter = None
        self._lock = threading.Lock()
        self.converter_init_seconds: float | None = None
        self.converter_init_count = 0

    def _get_converter(self):
        if self._converter is not None:
            return self._converter
        with self._lock:
            if self._converter is not None:
                return self._converter
            docling, DocumentConverter, PdfFormatOption, PdfPipelineOptions, base_models = _load_docling()
            # OCR is intentionally disabled: this parser targets digitally generated PDFs.
            start = time.perf_counter()
            try:
                options = PdfPipelineOptions(do_ocr=False)
                converter = DocumentConverter(
                    format_options={
                        base_models.InputFormat.PDF: PdfFormatOption(pipeline_options=options)
                    }
                )
            except Exception as exc:
                raise ParserError(
                    f"Docling converter initialization failed: {type(exc).__name__}: {exc}"
                ) from exc
            self.converter_init_seconds = time.perf_counter() - start
            self.converter_init_count += 1
            self._converter = converter
            return self._converter

    def parse(self, source: str | Path) -> ParsedDocument:
        docling = _load_docling()[0]
        path = validate_source(source)
        converter = self._get_converter()

        try:
            with self._lock:
                result = converter.convert(str(path))
        except Exception as exc:
            message = str(exc).lower()
            if "password" in message or "encrypt" in message:
                raise EncryptedDocumentError(
                    f"PDF '{path}' is password-protected; encrypted PDFs are not supported."
                ) from exc
            raise CorruptDocumentError(f"Could not convert '{path}' as a PDF: {exc}") from exc

        doc = result.document
        page_heights: dict[int, float | None] = {}
        pages: list[ParsedPage] = []
        for raw_page_no in sorted(getattr(doc, "pages", {}) or {}):
            page_no = int(raw_page_no)
            size = getattr(doc.pages[raw_page_no], "size", None)
            width = float(size.width) if size is not None else None
            height = float(size.height) if size is not None else None
            page_heights[page_no] = height
            pages.append(ParsedPage(page_number=page_no, width=width, height=height, blocks=[]))
        if not pages:
            raise EmptyDocumentError(f"PDF '{path}' contains no pages.")
        page_index = {page.page_number: page for page in pages}

        warnings: list[str] = []
        missing_prov = 0
        for item, _level in doc.iterate_items():
            block_type = _block_type(item)
            content = _item_content(item, doc)
            if block_type is not BlockType.IMAGE and not content:
                continue

            provs = list(getattr(item, "prov", []) or [])
            prov = provs[0] if provs else None
            page_number = int(getattr(prov, "page_no", 0) or 0) if prov else 0
            if page_number < 1 or page_number not in page_index:
                missing_prov += 1
                page_number = 1 if 1 in page_index else pages[0].page_number

            page_height = page_heights.get(page_number)
            bbox = _to_top_left(prov.bbox, page_height) if prov is not None and getattr(prov, "bbox", None) else None

            page_index[page_number].blocks.append(
                ParsedBlock(
                    block_type=block_type,
                    content=content,
                    page_number=page_number,
                    bbox=bbox,
                    metadata={"docling_label": _label_value(item), "self_ref": str(getattr(item, "self_ref", ""))},
                )
            )

        if missing_prov:
            warnings.append(f"{missing_prov} element(s) had no page provenance and were assigned to page 1.")
        for page in pages:
            if not page.blocks:
                warnings.append(f"Page {page.page_number} has no extractable content.")

        return ParsedDocument(
            source=str(path),
            parser_name=self.name,
            parser_version=str(getattr(docling, "__version__", "unknown")),
            metadata=_document_metadata(result, doc, str(getattr(docling, "__version__", "unknown"))),
            pages=pages,
            warnings=warnings,
        )
