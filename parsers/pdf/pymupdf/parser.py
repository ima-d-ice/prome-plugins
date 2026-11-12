"""PyMuPDF-based PDF parser: fast extraction for digitally generated PDFs."""

from __future__ import annotations

import logging
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
)
from parsers.contracts.validation import validate_source

logger = logging.getLogger(__name__)

_HEADING_SCALE = 1.2  # a span must be at least this much larger than the body size
_HEADING_MIN_SIZE = 13.0  # and at least this absolute size (PDF points)


def _load_pymupdf():
    try:
        import pymupdf

        return pymupdf
    except ImportError:
        try:
            import fitz  # legacy import name

            return fitz
        except ImportError as exc:
            raise ParserDependencyError(
                "PyMuPDF is not installed. Install it with:\n"
                "  pip install 'prome-rag-plugins[pymupdf]'\n"
                "or:\n"
                "  pip install pymupdf"
            ) from exc


def _bbox(raw) -> BoundingBox | None:
    try:
        x0, y0, x1, y1 = (float(v) for v in raw[:4])
        return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
    except (TypeError, ValueError):
        return None


def _block_text(block: dict[str, Any]) -> str:
    lines = []
    for line in block.get("lines", []):
        spans = [span.get("text", "") for span in line.get("spans", [])]
        line_text = "".join(spans).rstrip()
        if line_text:
            lines.append(line_text)
    return "\n".join(lines).strip()


def _max_span_size(block: dict[str, Any]) -> float:
    sizes = [span.get("size", 0.0) for line in block.get("lines", []) for span in line.get("spans", [])]
    return max(sizes, default=0.0)


def _body_span_size(page_dicts: list[dict[str, Any]]) -> float:
    """Estimate the body (most common) font size via the 10th percentile."""
    sizes = sorted(
        span.get("size", 0.0)
        for page in page_dicts
        for block in page.get("blocks", [])
        if block.get("type") == 0
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if span.get("text", "").strip()
    )
    if not sizes:
        return 0.0
    return sizes[int(0.1 * (len(sizes) - 1))]


def _point_in_rect(bbox: BoundingBox | None, rect) -> bool:
    if bbox is None:
        return False
    cx, cy = (bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2
    return rect[0] <= cx <= rect[2] and rect[1] <= cy <= rect[3]


_MIN_GUTTER = 18.0  # minimum vertical whitespace (points) that qualifies as a column gutter
_BARRIER_WIDTH_FRACTION = 0.55  # blocks wider than this are treated as full-width flow barriers


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _column_cut(blocks: list[ParsedBlock], page_width: float) -> float | None:
    """Return the x of a narrow vertical gutter that splits the page into columns.

    Only blocks noticeably narrower than the page participate in gutter
    detection; full-width flow blocks (titles, headers, footers, tables) are
    ignored here and handled as band barriers in the sort instead.
    """
    narrow = [
        b.bbox
        for b in blocks
        if b.bbox is not None and (b.bbox.x1 - b.bbox.x0) < _BARRIER_WIDTH_FRACTION * page_width
    ]
    if len(narrow) < 2:
        return None
    intervals = _merge_intervals([(b.x0, b.x1) for b in narrow])
    best: float | None = None
    for left, right in zip(intervals, intervals[1:]):
        gap_start, gap_end = left[1], right[0]
        gap = gap_end - gap_start
        center = (gap_start + gap_end) / 2
        if gap < _MIN_GUTTER:
            continue
        if not (0.25 * page_width < center < 0.75 * page_width):
            continue
        if best is None or abs(center - page_width / 2) < abs(best - page_width / 2):
            best = center
    if best is None:
        return None
    # demand column-like evidence on both sides, not just one narrow strip
    narrow_centers = [(b.x0 + b.x1) / 2 for b in narrow]
    if sum(c < best for c in narrow_centers) < 2 or sum(c > best for c in narrow_centers) < 2:
        return None
    return best


def _reading_sort_key(block: ParsedBlock, cut: float | None, barriers: list[BoundingBox]) -> tuple:
    bbox = block.bbox
    if bbox is None:
        return (float("inf"), float("inf"), 0.0)
    if cut is None:
        return (float(bbox.y0), float(bbox.x0), 0.0)
    if bbox.x0 < cut < bbox.x1:
        return (2 * barriers.index(bbox) + 1, -1.0, bbox.y0)
    band = sum(1 for barrier in barriers if barrier.y1 <= bbox.y0 + 1.0)
    column = 0.0 if (bbox.x0 + bbox.x1) / 2 < cut else 1.0
    return (2 * band, column, bbox.y0)


def _sort_reading_order(blocks: list[ParsedBlock], page_width: float) -> None:
    """Sort blocks in place into a natural reading order.

    Single-column pages keep the historical (y0, x0) order. Two-column pages
    are ordered column by column, with full-width blocks (headers, footers,
    banners) acting as flow barriers between horizontal bands.
    """
    cut = _column_cut(blocks, page_width)
    if cut is None:
        blocks.sort(key=lambda b: (b.bbox.y0 if b.bbox else float("inf"), b.bbox.x0 if b.bbox else 0.0))
        return
    barriers = sorted(
        (b.bbox for b in blocks if b.bbox is not None and b.bbox.x0 < cut < b.bbox.x1),
        key=lambda bbox: (bbox.y0, bbox.x0),
    )
    blocks.sort(key=lambda b: _reading_sort_key(b, cut, barriers))


def _table_rows_to_text(rows: list[list[Any]]) -> str:
    lines = []
    for row in rows:
        cells = ["" if cell is None else str(cell).strip() for cell in row]
        lines.append(" | ".join(cells))
    return "\n".join(lines).strip()


class PyMuPDFParser:
    name = "pymupdf"

    def parse(self, source: str | Path) -> ParsedDocument:
        pymupdf = _load_pymupdf()
        path = validate_source(source)

        try:
            doc = pymupdf.open(path, filetype="pdf")
        except Exception as exc:
            raise CorruptDocumentError(f"Could not open '{path}' as a PDF: {exc}") from exc

        try:
            if doc.needs_pass:
                if not doc.authenticate(""):
                    raise EncryptedDocumentError(
                        f"PDF '{path}' is password-protected; encrypted PDFs are not supported."
                    )
            if doc.page_count == 0:
                raise EmptyDocumentError(f"PDF '{path}' contains no pages.")

            raw_metadata = dict(doc.metadata or {})
            metadata = {k: v for k, v in raw_metadata.items() if v not in (None, "")}

            page_dicts = []
            page_tables = []
            for page in doc:
                try:
                    page_dicts.append(page.get_text("dict"))
                except Exception as exc:
                    logger.warning("Failed to extract text dict for page %s: %s", page.number + 1, exc)
                    page_dicts.append({"blocks": []})
                try:
                    finder = page.find_tables()
                    page_tables.append(list(finder.tables))
                except Exception as exc:
                    logger.debug("Table detection failed on page %s: %s", page.number + 1, exc)
                    page_tables.append([])

            body_size = _body_span_size(page_dicts)
            heading_threshold = max(_HEADING_MIN_SIZE, body_size * _HEADING_SCALE)

            pages: list[ParsedPage] = []
            warnings: list[str] = []
            for index, page in enumerate(doc):
                page_number = index + 1
                page_dict = page_dicts[index]
                tables = page_tables[index]
                table_rects = [tuple(t.bbox) for t in tables]

                blocks: list[ParsedBlock] = []
                for table in tables:
                    try:
                        rows = table.extract()
                        content = _table_rows_to_text(rows)
                    except Exception as exc:
                        logger.warning("Failed to extract table on page %s: %s", page_number, exc)
                        continue
                    blocks.append(
                        ParsedBlock(
                            block_type=BlockType.TABLE,
                            content=content,
                            page_number=page_number,
                            bbox=_bbox(table.bbox),
                            metadata={"rows": len(rows)},
                        )
                    )

                for block in page_dict.get("blocks", []):
                    bbox = _bbox(block.get("bbox"))
                    if any(_point_in_rect(bbox, rect) for rect in table_rects):
                        continue  # content already captured by the table block
                    if block.get("type") == 1:
                        blocks.append(
                            ParsedBlock(
                                block_type=BlockType.IMAGE,
                                content="",
                                page_number=page_number,
                                bbox=bbox,
                                metadata={
                                    "width": block.get("width"),
                                    "height": block.get("height"),
                                    "ext": block.get("ext"),
                                },
                            )
                        )
                        continue
                    text = _block_text(block)
                    if not text:
                        continue
                    block_type = (
                        BlockType.HEADING if _max_span_size(block) >= heading_threshold else BlockType.TEXT
                    )
                    blocks.append(
                        ParsedBlock(
                            block_type=block_type,
                            content=text,
                            page_number=page_number,
                            bbox=bbox,
                            metadata={"max_span_size": _max_span_size(block)},
                        )
                    )

                _sort_reading_order(blocks, float(page.rect.width))

                if not blocks:
                    warnings.append(f"Page {page_number} has no extractable content.")

                rect = page.rect
                pages.append(
                    ParsedPage(
                        page_number=page_number,
                        width=float(rect.width),
                        height=float(rect.height),
                        blocks=blocks,
                    )
                )

            version = getattr(pymupdf, "__version__", "unknown")
            return ParsedDocument(
                source=str(path),
                parser_name=self.name,
                parser_version=str(version),
                metadata=metadata,
                pages=pages,
                warnings=warnings,
            )
        finally:
            doc.close()
