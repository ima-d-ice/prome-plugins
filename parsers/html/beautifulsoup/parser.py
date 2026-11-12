"""BeautifulSoup-based HTML parser: preserve the DOM structure of HTML documents.

This parser performs full-document, structure-first extraction: every
meaningful element in document order is emitted as a ParsedBlock, including
navigation, header, sidebar and footer content (controlled by the
``include_boilerplate`` flag). It does not download URLs, does not execute
scripts, and does not evaluate CSS.
"""

from __future__ import annotations

import json
import logging
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

_HEADINGS = {f"h{level}": level for level in range(1, 7)}
_SKIP_TAGS = {
    "head",  # page metadata is harvested separately
    "script",
    "style",
    "noscript",
    "template",
    "iframe",
    "canvas",
    "svg",
    "select",
    "option",
    "input",
}
_BOILERPLATE_TAGS = {"nav", "header", "footer", "aside"}
_BOILERPLATE_ROLES = {"navigation", "banner", "contentinfo", "complementary"}
_BACKENDS = {"html.parser": None, "lxml": "lxml", "html5lib": "html5lib"}
# Elements that can hold block-level content (dispatched or recursed into).
_BLOCK_CONTAINERS = set(_HEADINGS) | {
    "p", "pre", "blockquote", "figcaption", "ul", "ol", "dl", "table",
}
# Generic containers: emit their residual inline text (anything not claimed
# by a nested block/generic/skipped/boilerplate container), then recurse.
_GENERIC_CONTAINERS = {
    "div", "section", "main", "article", "header", "footer", "nav", "aside",
    "figure", "form", "fieldset", "address", "summary", "details", "body",
}


def _load_beautifulsoup(backend: str):
    backend = (backend or "html.parser").lower()
    if backend not in _BACKENDS:
        raise ValueError(
            f"Unsupported backend {backend!r}. Expected one of: {', '.join(sorted(_BACKENDS))}."
        )
    try:
        from bs4 import BeautifulSoup  # noqa: F401
    except ImportError as exc:
        raise ParserDependencyError(
            "beautifulsoup4 is not installed. Install it with:\n"
            "  pip install 'prome-rag-plugins[beautifulsoup]'\n"
            "or:\n"
            "  pip install beautifulsoup4"
        ) from exc
    module = _BACKENDS[backend]
    if module is not None:
        try:
            __import__(module)
        except ImportError as exc:
            raise ParserDependencyError(
                f"Backend {backend!r} requires the {module!r} package. Install it with:\n"
                f"  pip install {module}\n"
                "or use the default 'html.parser' backend."
            ) from exc
    return backend


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


def _replace_br(element) -> None:
    for br in element.find_all("br"):
        br.replace_with(" ")


def _inline_text(element) -> str:
    _replace_br(element)
    return _collapse(element.get_text("", strip=False))


def _item_inline_text(li) -> str:
    """Inline text of an <li>, excluding text of nested lists."""
    from bs4 import NavigableString, Tag

    parts: list[str] = []

    def _collect(node):
        for child in getattr(node, "children", []):
            if isinstance(child, NavigableString):
                parts.append(str(child))
            elif isinstance(child, Tag):
                if child.name in ("ul", "ol"):
                    continue
                if child.name in _SKIP_TAGS:
                    continue
                if child.name == "br":
                    parts.append(" ")
                    continue
                _collect(child)

    _collect(li)
    return _collapse("".join(parts))


def _render_list(list_tag, depth: int = 0):
    ordered = list_tag.name == "ol"
    lines: list[str] = []
    items: list[str] = []
    max_depth = depth
    index = 0
    for li in list_tag.find_all("li", recursive=False):
        index += 1
        marker = f"{index}." if ordered else "-"
        text = _item_inline_text(li)
        lines.append("  " * depth + marker + " " + text)
        items.append(text)
        for nested in li.find_all(["ul", "ol"], recursive=False):
            sub_lines, sub_items, sub_depth = _render_list(nested, depth + 1)
            lines.extend(sub_lines)
            items.extend(sub_items)
            max_depth = max(max_depth, sub_depth)
    return lines, items, max_depth


def _extract_row_cells(rows) -> list[list[str]]:
    extracted: list[list[str]] = []
    for row in rows:
        cells: list[str] = []
        for cell in row.find_all(["th", "td"]):
            try:
                colspan = max(1, int(cell.get("colspan", 1)))
            except (TypeError, ValueError):
                colspan = 1
            cells.extend([_inline_text(cell)] * colspan)
        extracted.append(cells)
    return extracted


def _table_parts(table) -> tuple[list[list[str]], list[str], bool]:
    """Return (all_rows, header_cells, has_header).

    Header rows are taken from <thead> when present, otherwise from a leading
    row that contains <th> cells.
    """
    thead = table.find("thead")
    header_cells: list[str] = []
    rows: list[list[str]] = []

    if thead is not None:
        header_rows = _extract_row_cells(thead.find_all("tr", recursive=True))
        if header_rows:
            header_cells = header_rows[-1]
            rows.extend(header_rows)
        body_rows = [tr for tr in table.find_all("tr") if tr.find_parent("thead") is None]
        rows.extend(_extract_row_cells(body_rows))
    else:
        all_rows = _extract_row_cells(table.find_all("tr"))
        if all_rows:
            first_tr = table.find("tr")
            first_has_th = bool(first_tr and first_tr.find("th"))
            if first_has_th:
                header_cells = all_rows[0]
            rows = all_rows
    return rows, header_cells, bool(header_cells)


def _table_content(rows: list[list[str]]) -> str:
    width = max((len(r) for r in rows), default=0)
    lines = []
    for row in rows:
        cells = row + [""] * (width - len(row))
        lines.append(" | ".join(cells))
    return "\n".join(lines).strip()


def _metadata(soup, source_url: str | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    title = soup.find("title")
    if title is not None:
        title_text = _collapse(title.get_text())
        if title_text:
            metadata["title"] = title_text
    for meta in soup.find_all("meta"):
        name = str(meta.get("name") or meta.get("property") or meta.get("http-equiv") or "").strip().lower()
        content = meta.get("content")
        if not name or content is None:
            continue
        content = _collapse(str(content))
        if not content:
            continue
        metadata.setdefault(name, content)
    canonical = soup.find("link", rel=lambda v: bool(v) and "canonical" in str(v).lower())
    if canonical is not None and canonical.get("href"):
        metadata["canonical"] = _collapse(str(canonical["href"]))
    if soup.html is not None and soup.html.get("lang"):
        metadata["lang"] = str(soup.html["lang"]).strip()
    json_ld: list[Any] = []
    for script in soup.find_all("script", type=lambda v: v and "ld+json" in str(v).lower()):
        raw = script.string if script.string is not None else script.get_text()
        try:
            json_ld.append(json.loads(raw))
        except Exception:
            json_ld.append({"raw": raw.strip(), "parse_error": True})
    if json_ld:
        metadata["json_ld"] = json_ld
    resolved = source_url or metadata.get("canonical") or metadata.get("og:url")
    if resolved:
        metadata["source_url"] = str(resolved)
    return metadata


def _is_boilerplate(tag) -> bool:
    if tag.name in _BOILERPLATE_TAGS:
        return True
    role = str(tag.get("role") or "").strip().lower()
    return role in _BOILERPLATE_ROLES


def _residual_text(element) -> str:
    """Text of an element that no nested container will claim.

    A nested block container (p/ul/table/...) emits its own text; nested
    generic containers compute their own residual. This returns only the
    strings whose nearest block/generic/skip/boilerplate ancestor is the
    element itself, so nothing is emitted twice and skipped subtrees
    (scripts, inline <style>) never leak.
    """
    names = list(_SKIP_TAGS | _BLOCK_CONTAINERS | _GENERIC_CONTAINERS | _BOILERPLATE_TAGS)
    parts = [
        str(s)
        for s in element.find_all(string=True)
        if (anc := s.find_parent(names)) is None or anc is element
    ]
    return _collapse("".join(parts))


def _code_language(code_tag) -> str | None:
    for cls in code_tag.get("class", []) or []:
        cls = str(cls)
        for prefix in ("language-", "lang-"):
            if cls.startswith(prefix):
                return cls[len(prefix):]
    return None


class BeautifulSoupParser:
    """Full-document HTML parser preserving DOM structure and ordering."""

    name = "beautifulsoup"

    def __init__(self, backend: str = "html.parser", include_boilerplate: bool = True) -> None:
        self.backend = backend
        self.include_boilerplate = include_boilerplate

    def parse(self, source: str | Path | bytes, source_url: str | None = None) -> ParsedDocument:
        backend = _load_beautifulsoup(self.backend)
        html_bytes, source_label = _read_html(source)

        from bs4 import BeautifulSoup, Tag

        try:
            soup = BeautifulSoup(html_bytes, backend)
        except Exception as exc:
            raise CorruptDocumentError(f"Could not parse '{source_label}' as HTML: {exc}") from exc

        try:
            metadata = _metadata(soup, source_url)
        except Exception as exc:  # metadata must never abort content extraction
            logger.warning("Metadata extraction failed for '%s': %s", source_label, exc)
            metadata = {"source_url": str(source_url)} if source_url else {}

        try:
            bs4_version = __import__("bs4").__version__
        except Exception:
            bs4_version = "unknown"

        blocks: list[ParsedBlock] = []
        state = {"order": 0}

        def emit(block_type: BlockType, content: str, tag_name: str, metadata_extra: dict[str, Any] | None = None):
            content = content.strip()
            if not content and block_type is not BlockType.IMAGE:
                return
            block_meta = {"tag": tag_name, "dom_order": state["order"]}
            if metadata_extra:
                block_meta.update(metadata_extra)
            state["order"] += 1
            blocks.append(ParsedBlock(block_type=block_type, content=content, page_number=None, metadata=block_meta))

        def walk(node):
            for child in getattr(node, "children", []):
                if not isinstance(child, Tag):
                    continue
                tag = child.name.lower()
                if tag in _SKIP_TAGS:
                    continue
                if not self.include_boilerplate and _is_boilerplate(child):
                    continue
                if tag in _HEADINGS:
                    emit(BlockType.HEADING, _inline_text(child), tag, {"heading_level": _HEADINGS[tag]})
                    continue  # heading text kept verbatim; inline refs not split out
                if tag == "p":
                    emit(BlockType.TEXT, _inline_text(child), tag)
                    walk(child)  # pick up inline links/images; emits no nested TEXT
                    continue
                if tag == "pre":
                    code_child = child.find("code")
                    language = _code_language(code_child) if code_child is not None else _code_language(child)
                    emit(BlockType.CODE, child.get_text("\n", strip=False).strip("\n"), tag, {"language": language} if language else {})
                    continue
                if tag == "blockquote":
                    emit(BlockType.TEXT, _inline_text(child), tag, {"quote": True})
                    continue
                if tag == "figcaption":
                    emit(BlockType.TEXT, _inline_text(child), tag, {"figcaption": True})
                    walk(child)
                    continue
                if tag in ("ul", "ol"):
                    lines, items, depth = _render_list(child)
                    emit(
                        BlockType.LIST,
                        "\n".join(lines),
                        tag,
                        {"list_type": tag, "ordered": tag == "ol", "items": items, "depth": depth},
                    )
                    continue
                if tag == "dl":
                    lines, items = [], []
                    current_term: str | None = None
                    for entry in child.find_all(["dt", "dd"], recursive=False):
                        if entry.name == "dt":
                            current_term = _inline_text(entry)
                        else:
                            definition = _inline_text(entry)
                            lines.append(f"{current_term}: {definition}" if current_term else definition)
                            items.append(definition)
                    emit(BlockType.LIST, "\n".join(lines), tag, {"list_type": "dl", "ordered": False, "items": items, "depth": 0})
                    continue
                if tag == "table":
                    rows, headers, has_header = _table_parts(child)
                    rows = [row for row in rows if any(cell for cell in row)]
                    if not rows:
                        continue
                    caption = child.find("caption")
                    table_meta: dict[str, Any] = {
                        "rows": len(rows),
                        "cols": max(len(r) for r in rows),
                        "headers": headers,
                        "has_header": has_header,
                        "row_cells": rows,
                    }
                    if caption is not None:
                        table_meta["caption"] = _inline_text(caption)
                    emit(BlockType.TABLE, _table_content(rows), tag, table_meta)
                    continue
                if tag == "img":
                    alt = _collapse(str(child.get("alt") or ""))
                    emit(
                        BlockType.IMAGE,
                        alt,
                        tag,
                        {
                            "src": str(child.get("src") or ""),
                            "alt": alt,
                            "title": str(child.get("title") or ""),
                            "width": child.get("width"),
                            "height": child.get("height"),
                        },
                    )
                    continue
                if tag == "a":
                    href = str(child.get("href") or "").strip()
                    if href:
                        link_meta: dict[str, Any] = {"href": href}
                        if child.get("title"):
                            link_meta["title"] = _collapse(str(child["title"]))
                        if child.get("rel"):
                            rel = child["rel"]
                            link_meta["rel"] = " ".join(rel) if isinstance(rel, list) else str(rel)
                        emit(BlockType.LINK, _inline_text(child) or href, tag, link_meta)
                    walk(child)
                    continue
                if tag in _GENERIC_CONTAINERS:
                    residual = _residual_text(child)
                    if residual:
                        emit(BlockType.TEXT, residual, tag)
                    walk(child)
                    continue
                walk(child)

        start = soup.body if soup.body is not None else soup
        walk(start)

        warnings: list[str] = []
        if not blocks:
            warnings.append("No extractable content found in the document.")

        return ParsedDocument(
            source=source_label,
            parser_name=self.name,
            parser_version=str(bs4_version),
            metadata=metadata,
            pages=[],
            blocks=blocks,
            warnings=warnings,
        )
