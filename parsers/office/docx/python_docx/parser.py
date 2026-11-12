"""python-docx based DOCX parser: native Word document extraction.

Preserves heading hierarchy, lists (where style names identify them),
tables (as structured blocks with row/cell grids), hyperlinks, headers,
footers, sections and core document metadata. Document order is preserved.
Pageless output: blocks land on ``doc.blocks`` and ``doc.pages == []``
(Word paginates at render time; python-docx cannot see page boundaries).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from parsers.contracts import (
    BlockType,
    CorruptDocumentError,
    EmptyDocumentError,
    ParsedBlock,
    ParsedDocument,
    ParserDependencyError,
)
from parsers.contracts.validation import validate_source

logger = logging.getLogger(__name__)

_W_P = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"
_W_TBL = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tbl"
_W_HYPERLINK = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}hyperlink"
_R_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"

_LIST_STYLE = re.compile(r"^list\s+(bullet|number)(?:\s+(\d+))?$", re.IGNORECASE)
_HEADING_STYLE = re.compile(r"^heading\s+(\d+)$", re.IGNORECASE)
_TITLE_STYLE = re.compile(r"^(title|subtitle)$", re.IGNORECASE)


def _load_docx():
    try:
        import docx

        return docx
    except ImportError as exc:
        raise ParserDependencyError(
            "python-docx is not installed. Install it with:\n"
            "  pip install 'prome-rag-plugins[docx]'\n"
            "or:\n"
            "  pip install python-docx"
        ) from exc


def _iter_body_items(document):
    """Yield ('paragraph'|'table', object) in document order."""
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in document.element.body.iterchildren():
        if child.tag == _W_P:
            yield "paragraph", Paragraph(child, document)
        elif child.tag == _W_TBL:
            yield "table", Table(child, document)


def _core_metadata(core) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    fields = {
        "title": core.title,
        "author": core.author,
        "subject": core.subject,
        "keywords": core.keywords,
        "category": core.category,
        "comments": core.comments,
        "last_modified_by": core.last_modified_by,
        "language": core.language,
        "identifier": core.identifier,
        "version": core.version,
    }
    for key, value in fields.items():
        if value not in (None, ""):
            metadata[key] = str(value)
    if core.revision is not None:
        metadata["revision"] = int(core.revision)
    for key in ("created", "modified", "printed"):
        value = getattr(core, key, None)
        if value is not None:
            metadata[key] = value.isoformat() if hasattr(value, "isoformat") else str(value)
    return metadata


def _sections_metadata(document) -> list[dict[str, Any]]:
    sections = []
    for index, section in enumerate(document.sections):
        orientation = getattr(section.orientation, "name", str(section.orientation))
        sections.append(
            {
                "index": index,
                "start_type": getattr(section.start_type, "name", str(section.start_type)),
                "orientation": orientation.lower() if isinstance(orientation, str) else orientation,
                "page_width_emu": int(section.page_width) if section.page_width else None,
                "page_height_emu": int(section.page_height) if section.page_height else None,
                "header_distance_emu": int(section.header_distance) if section.header_distance else None,
                "footer_distance_emu": int(section.footer_distance) if section.footer_distance else None,
            }
        )
    return sections


def _paragraph_hyperlinks(paragraph) -> list[dict[str, str]]:
    """Resolve external hyperlink relationships found in a paragraph."""
    links = []
    for hyperlink in paragraph._p.iter(_W_HYPERLINK):
        r_id = hyperlink.get(_R_ID)
        href = None
        if r_id:
            try:
                rel = paragraph.part.rels[r_id]
                href = getattr(rel, "target_ref", None) or str(getattr(rel, "_target", ""))
            except KeyError:
                href = None
        anchor = hyperlink.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}anchor")
        text = "".join(node.text or "" for node in hyperlink.iter() if node.tag.endswith("}t"))
        if href or anchor:
            links.append({"text": text.strip(), "href": href, "anchor": anchor})
    return links


class PythonDocxParser:
    """Extract structured DOCX content with python-docx."""

    name = "python_docx"

    def parse(self, source: str | Path) -> ParsedDocument:
        docx = _load_docx()
        path = validate_source(source)
        try:
            document = docx.Document(str(path))
        except Exception as exc:
            raise CorruptDocumentError(f"Could not open '{path}' as a DOCX file: {exc}") from exc

        metadata = _core_metadata(document.core_properties)
        metadata["sections"] = _sections_metadata(document)

        blocks: list[ParsedBlock] = []
        order = {"i": 0}

        def emit(block_type: BlockType, content: str, block_metadata: dict[str, Any] | None = None) -> None:
            content = content.strip()
            if not content and block_type not in (BlockType.IMAGE,):
                return
            meta = {"document_order": order["i"]}
            if block_metadata:
                meta.update(block_metadata)
            order["i"] += 1
            blocks.append(ParsedBlock(block_type=block_type, content=content, page_number=None, metadata=meta))

        def emit_paragraph(paragraph, *, region: str | None = None) -> None:
            text = paragraph.text.strip()
            style = (paragraph.style.name if paragraph.style is not None else "") or ""
            meta: dict[str, Any] = {"style": style}
            if region:
                meta["region"] = region
            heading = _HEADING_STYLE.match(style.strip())
            title = _TITLE_STYLE.match(style.strip())
            list_match = _LIST_STYLE.match(style.strip())
            if heading:
                meta["heading_level"] = int(heading.group(1))
                emit(BlockType.HEADING, text, meta)
            elif title:
                meta["heading_level"] = 1 if title.group(1).lower() == "title" else 2
                emit(BlockType.HEADING, text, meta)
            elif list_match:
                ordered = list_match.group(1).lower() == "number"
                level = int(list_match.group(2)) if list_match.group(2) else 1
                meta.update({"list_type": "ol" if ordered else "ul", "ordered": ordered, "list_level": level})
                emit(BlockType.LIST, text, meta)
            elif text:
                emit(BlockType.TEXT, text, meta)
            for link in _paragraph_hyperlinks(paragraph):
                if link["href"]:
                    emit(BlockType.LINK, link["text"] or link["href"], {"href": link["href"], **({"region": region} if region else {})})

        # Body in document order.
        for kind, item in _iter_body_items(document):
            if kind == "paragraph":
                emit_paragraph(item)
                continue
            table = item
            rows: list[list[str]] = []
            for row in table.rows:
                rows.append([cell.text.strip() for cell in row.cells])
            if not rows:
                continue
            header = rows[0]
            emit(
                BlockType.TABLE,
                "\n".join(" | ".join(cell for cell in row) for row in rows),
                {
                    "rows": len(rows),
                    "cols": max(len(r) for r in rows),
                    "headers": header,
                    "has_header": bool(header and any(header)),
                    "row_cells": rows,
                },
            )

        # Inline images (in document order is not trivially available across
        # runs; report them after the body with index references).
        for index, shape in enumerate(document.inline_shapes):
            meta: dict[str, Any] = {"inline_index": index}
            try:
                meta["width_emu"] = int(shape.width)
                meta["height_emu"] = int(shape.height)
            except Exception:
                pass
            try:
                image_part = shape._inline.docPr  # noqa: SLF001 - python-docx has no public filename API
                if image_part is not None:
                    meta["name"] = image_part.get("name")
            except Exception:
                pass
            meta["shape_type"] = str(getattr(shape, "type", "picture"))
            emit(BlockType.IMAGE, f"[image {index + 1}]", meta)

        # Headers/footers per section.
        for index, section in enumerate(document.sections):
            for region_name, container in (("header", section.header), ("footer", section.footer)):
                for paragraph in container.paragraphs:
                    if paragraph.text.strip():
                        emit_paragraph(paragraph, region=f"section_{index}_{region_name}")

        if not blocks:
            raise EmptyDocumentError(f"DOCX '{path}' contains no extractable content.")

        import docx as _docx

        return ParsedDocument(
            source=str(path),
            parser_name=self.name,
            parser_version=str(getattr(_docx, "__version__", "unknown")),
            metadata=metadata,
            pages=[],
            blocks=blocks,
            warnings=[],
        )
