"""Trafilatura-based HTML parser: extract the primary content of a page.

Unlike the BeautifulSoup plugin, this parser is optimized for isolating the
main article/document from boilerplate (navigation, headers, sidebars,
footers). It never downloads URLs; it parses HTML already held in memory.

Structure that survives Trafilatura's extraction (headings, paragraphs,
lists, tables, code, inline links) is emitted as blocks. Elements
Trafilatura does not retain (images, arbitrary inline markup) are reported
through the absence of blocks, not fabricated.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from parsers.contracts import (
    BlockType,
    DocumentParseError,
    EmptyDocumentError,
    ParsedBlock,
    ParsedDocument,
    ParserDependencyError,
)
from parsers.contracts.validation import validate_source

logger = logging.getLogger(__name__)

_HEADINGS = {f"h{level}": level for level in range(1, 7)}


def _load_trafilatura():
    try:
        import trafilatura

        return trafilatura
    except ImportError as exc:
        raise ParserDependencyError(
            "Trafilatura is not installed. Install it with:\n"
            "  pip install 'prome-rag-plugins[trafilatura]'\n"
            "or:\n"
            "  pip install trafilatura"
        ) from exc


def _read_html(source: str | Path | bytes) -> tuple[bytes, str]:
    """Return (html_bytes, source_label) for a path or raw HTML string/bytes.

    Raw HTML strings are recognised by the presence of a '<' character;
    anything else is treated as a filesystem path, mirroring the PDF parsers.
    """
    if isinstance(source, bytes):
        if not source.strip():
            raise EmptyDocumentError("HTML input is empty (0 bytes).")
        return source, "<bytes>"
    if isinstance(source, Path):
        path = validate_source(source)
        return path.read_bytes(), str(path)
    text = str(source)
    if not text.strip():
        raise EmptyDocumentError("HTML input is empty or whitespace-only.")
    if "<" not in text:
        path = validate_source(text)
        return path.read_bytes(), str(path)
    return text.encode("utf-8"), "<string>"


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _element_text(element) -> str:
    return _collapse("".join(element.itertext()))


def _render_list(list_element, depth: int = 0):
    ordered = (list_element.get("rend") or "").strip().lower() == "ol"
    lines: list[str] = []
    items: list[str] = []
    max_depth = depth
    index = 0
    for item in list_element:
        if item.tag != "item":
            continue
        index += 1
        marker = f"{index}." if ordered else "-"
        own_parts: list[str] = []
        if item.text:
            own_parts.append(item.text)
        for child in item:
            if child.tag == "list":
                if child.tail:
                    own_parts.append(child.tail)
                continue
            own_parts.append("".join(child.itertext()))
            if child.tail:
                own_parts.append(child.tail)
        own_text = _collapse("".join(own_parts))
        lines.append("  " * depth + marker + " " + own_text)
        items.append(own_text)
        for child in item:
            if child.tag == "list":
                sub_lines, sub_items, sub_depth = _render_list(child, depth + 1)
                lines.extend(sub_lines)
                items.extend(sub_items)
                max_depth = max(max_depth, sub_depth)
    return lines, items, max_depth


def _table_rows(table_element) -> tuple[list[list[str]], list[str], bool]:
    rows: list[list[str]] = []
    header_cells: list[str] = []
    for row in table_element:
        if row.tag != "row":
            continue
        row_cells: list[str] = []
        is_header = False
        for cell in row:
            if cell.tag != "cell":
                continue
            row_cells.append(_element_text(cell))
            if (cell.get("role") or "").lower() == "head":
                is_header = True
        if not row_cells:
            continue
        rows.append(row_cells)
        if is_header:
            header_cells = row_cells
    return rows, header_cells, bool(header_cells)


def _table_content(rows: list[list[str]]) -> str:
    width = max((len(r) for r in rows), default=0)
    lines = []
    for row in rows:
        cells = row + [""] * (width - len(row))
        lines.append(" | ".join(cells))
    return "\n".join(lines).strip()


def _document_metadata(document, source_url: str | None) -> dict[str, Any]:
    keys = (
        "title",
        "author",
        "date",
        "description",
        "sitename",
        "hostname",
        "language",
        "pagetype",
        "license",
        "image",
        "categories",
        "tags",
    )
    metadata: dict[str, Any] = {}
    for key in keys:
        value = getattr(document, key, None)
        if value in (None, "", [], {}):
            continue
        metadata[key] = value
    # "date" is the publication date discovered from the page itself.
    # Trafilatura's `filedate` field (the search upper bound, i.e. today) is
    # deliberately not reported: it is not an ingestion timestamp and is not
    # a publication date. Callers that need an ingestion timestamp must stamp
    # it at ingestion time.
    url = source_url or getattr(document, "url", None)
    if url:
        metadata["source_url"] = str(url)
    return metadata


def _trafilatura_version(trafilatura) -> str:
    return str(getattr(trafilatura, "__version__", "unknown"))


class TrafilaturaParser:
    """Main-content HTML extraction (boilerplate removal) via Trafilatura."""

    name = "trafilatura"

    def __init__(
        self,
        *,
        favor_precision: bool = False,
        favor_recall: bool = False,
        include_comments: bool = False,
        include_tables: bool = True,
        include_formatting: bool = True,
        include_links: bool = True,
        include_images: bool = False,
        target_language: str | None = None,
        extensive_date_search: bool = False,
    ) -> None:
        if favor_precision and favor_recall:
            raise ValueError(
                "favor_precision and favor_recall are mutually exclusive; "
                "choose at most one."
            )
        self.favor_precision = favor_precision
        self.favor_recall = favor_recall
        self.include_comments = include_comments
        self.include_tables = include_tables
        self.include_formatting = include_formatting
        self.include_links = include_links
        self.include_images = include_images
        self.target_language = target_language
        # htmldate's extensive free-text search invents dates out of unrelated
        # text (e.g. "2000 docs/min" -> 2000-01-01). It is off by default so a
        # page without a real, discoverable publication date yields no date.
        self.extensive_date_search = extensive_date_search

    def parse(self, source: str | Path | bytes, source_url: str | None = None) -> ParsedDocument:
        trafilatura = _load_trafilatura()
        html_bytes, source_label = _read_html(source)

        try:
            result = trafilatura.bare_extraction(
                html_bytes,
                url=source_url,
                fast=False,
                favor_precision=self.favor_precision,
                favor_recall=self.favor_recall,
                include_comments=self.include_comments,
                output_format="xml",
                target_language=self.target_language,
                include_tables=self.include_tables,
                include_images=self.include_images,
                include_formatting=self.include_formatting,
                include_links=self.include_links,
                deduplicate=False,
                with_metadata=True,
                date_extraction_params={
                    "original_date": True,
                    "extensive_search": self.extensive_date_search,
                },
            )
        except Exception as exc:
            raise DocumentParseError(
                f"Could not parse '{source_label}' as HTML with trafilatura: {exc}"
            ) from exc

        body = getattr(result, "body", None) if result is not None else None

        warnings: list[str] = []
        if result is None or body is None:
            metadata = {}
            try:
                fallback = trafilatura.extract_metadata(html_bytes, default_url=source_url)
            except Exception as exc:
                logger.debug("Metadata-only fallback failed for '%s': %s", source_label, exc)
                fallback = None
            if fallback is not None:
                metadata = _document_metadata(fallback, source_url)
            warnings.append("No main content could be identified in the document.")
            return ParsedDocument(
                source=source_label,
                parser_name=self.name,
                parser_version=_trafilatura_version(trafilatura),
                metadata=metadata,
                pages=[],
                blocks=[],
                warnings=warnings,
            )

        metadata = _document_metadata(result, source_url)

        blocks: list[ParsedBlock] = []
        state = {"order": 0}

        def emit(block_type: BlockType, content: str, tag_name: str, metadata_extra: dict[str, Any] | None = None):
            content = content.strip()
            if not content:
                return
            block_meta = {"tag": tag_name, "dom_order": state["order"]}
            if metadata_extra:
                block_meta.update(metadata_extra)
            state["order"] += 1
            blocks.append(ParsedBlock(block_type=block_type, content=content, page_number=None, metadata=block_meta))

        def emit_refs(element):
            for ref in element.iter("ref"):
                href = str(ref.get("target") or "").strip()
                if not href:
                    continue
                content = _element_text(ref) or href
                emit(BlockType.LINK, content, "ref", {"href": href})

        def walk(element):
            for child in element:
                tag = (child.tag or "").lower() if isinstance(child.tag, str) else ""
                if tag == "head":
                    rend = (child.get("rend") or "").strip().lower()
                    level = _HEADINGS.get(rend)
                    emit(
                        BlockType.HEADING,
                        _element_text(child),
                        tag,
                        {"heading_level": level} if level else {},
                    )
                    emit_refs(child)
                    continue
                if tag == "p":
                    emit(BlockType.TEXT, _element_text(child), tag)
                    emit_refs(child)
                    continue
                if tag == "list":
                    lines, items, depth = _render_list(child)
                    ordered = (child.get("rend") or "").strip().lower() == "ol"
                    emit(
                        BlockType.LIST,
                        "\n".join(lines),
                        tag,
                        {"list_type": "ol" if ordered else "ul", "ordered": ordered, "items": items, "depth": depth},
                    )
                    continue
                if tag == "table":
                    rows, headers, has_header = _table_rows(child)
                    rows = [row for row in rows if any(cell for cell in row)]
                    if rows:
                        emit(
                            BlockType.TABLE,
                            _table_content(rows),
                            tag,
                            {
                                "rows": len(rows),
                                "cols": max(len(r) for r in rows),
                                "headers": headers,
                                "has_header": has_header,
                                "row_cells": rows,
                            },
                        )
                    continue
                if tag == "code":
                    language = None
                    candidates = [child] + list(child.iter("code"))
                    for node in candidates:
                        for cls in (node.get("class") or "").split():
                            for prefix in ("language-", "lang-"):
                                if cls.startswith(prefix):
                                    language = cls[len(prefix):]
                    emit(
                        BlockType.CODE,
                        "".join(child.itertext()).strip("\n"),
                        tag,
                        {"language": language} if language else {},
                    )
                    continue
                if tag == "quote":
                    for sub in child:
                        walk(sub)
                    if not list(child) and (child.text or "").strip():
                        emit(BlockType.TEXT, _element_text(child), tag, {"quote": True})
                    continue
                if tag == "ref":
                    target = str(child.get("target") or "").strip()
                    if target:
                        emit(BlockType.LINK, _element_text(child) or target, tag, {"href": target})
                    continue
                # Unknown/wrapper elements: recurse to be safe.
                walk(child)

        walk(body)

        if not blocks:
            warnings.append("Main content was identified but no structured blocks could be built.")

        return ParsedDocument(
            source=source_label,
            parser_name=self.name,
            parser_version=_trafilatura_version(trafilatura),
            metadata=metadata,
            pages=[],
            blocks=blocks,
            warnings=warnings,
        )
