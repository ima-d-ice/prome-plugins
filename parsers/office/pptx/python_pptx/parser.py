"""python-pptx based PPTX parser: slide-structured PowerPoint extraction.

Preserves slide boundaries, titles, text boxes, bullet levels, tables,
shape information, speaker notes, and slide geometry.
Each slide is mapped to a ParsedPage with its 1-based slide index.
"""

from __future__ import annotations

import logging
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
)
from parsers.contracts.validation import validate_source

logger = logging.getLogger(__name__)

EMU_PER_PT = 12700.0  # 1 pt = 12700 EMUs (72 pt/in, 914400 emu/in)


def _load_pptx():
    try:
        import pptx
        from pptx.enum.shapes import PP_PLACEHOLDER

        return pptx, PP_PLACEHOLDER
    except ImportError as exc:
        raise ParserDependencyError(
            "python-pptx is not installed. Install it with:\n"
            "  pip install 'prome-rag-plugins[pptx]'\n"
            "or:\n"
            "  pip install python-pptx"
        ) from exc


def _core_metadata(core) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    fields = {
        "title": getattr(core, "title", None),
        "author": getattr(core, "author", None),
        "subject": getattr(core, "subject", None),
        "keywords": getattr(core, "keywords", None),
        "category": getattr(core, "category", None),
        "comments": getattr(core, "comments", None),
        "last_modified_by": getattr(core, "last_modified_by", None),
        "language": getattr(core, "language", None),
    }
    for key, value in fields.items():
        if value not in (None, ""):
            metadata[key] = str(value)
    if getattr(core, "revision", None) is not None:
        metadata["revision"] = int(core.revision)
    for key in ("created", "modified"):
        val = getattr(core, key, None)
        if val is not None:
            metadata[key] = val.isoformat() if hasattr(val, "isoformat") else str(val)
    return metadata


def _to_bbox(shape) -> BoundingBox | None:
    try:
        left = float(shape.left) / EMU_PER_PT
        top = float(shape.top) / EMU_PER_PT
        width = float(shape.width) / EMU_PER_PT
        height = float(shape.height) / EMU_PER_PT
        return BoundingBox(x0=left, y0=top, x1=left + width, y1=top + height)
    except Exception:
        return None


class PythonPptxParser:
    """Extract structured PPTX content using python-pptx."""

    name = "python_pptx"

    def parse(self, source: str | Path) -> ParsedDocument:
        pptx_mod, PP_PLACEHOLDER = _load_pptx()
        path = validate_source(source)

        try:
            prs = pptx_mod.Presentation(str(path))
        except Exception as exc:
            raise CorruptDocumentError(f"Could not open '{path}' as a PPTX file: {exc}") from exc

        metadata = _core_metadata(prs.core_properties)
        slide_w_pt = float(prs.slide_width) / EMU_PER_PT if prs.slide_width else None
        slide_h_pt = float(prs.slide_height) / EMU_PER_PT if prs.slide_height else None
        metadata["slide_width_pt"] = slide_w_pt
        metadata["slide_height_pt"] = slide_h_pt

        pages: list[ParsedPage] = []
        total_blocks = 0

        for slide_idx, slide in enumerate(prs.slides, start=1):
            slide_blocks: list[ParsedBlock] = []

            # Determine title shape if any
            title_shape = None
            try:
                title_shape = slide.shapes.title
            except Exception:
                title_shape = None

            for shape_idx, shape in enumerate(slide.shapes):
                bbox = _to_bbox(shape)
                base_meta: dict[str, Any] = {
                    "shape_index": shape_idx,
                    "shape_name": shape.name,
                    "shape_type": str(shape.shape_type),
                }

                # 1. Title shape
                if shape == title_shape and shape.has_text_frame:
                    title_text = shape.text_frame.text.strip()
                    if title_text:
                        slide_blocks.append(
                            ParsedBlock(
                                block_type=BlockType.HEADING,
                                content=title_text,
                                page_number=slide_idx,
                                bbox=bbox,
                                metadata={**base_meta, "heading_level": 1, "is_slide_title": True},
                            )
                        )
                    continue

                # 2. Table shape
                if shape.has_table:
                    table = shape.table
                    rows: list[list[str]] = []
                    for row in table.rows:
                        rows.append([c.text.strip() for c in row.cells])
                    if rows:
                        headers = rows[0]
                        md_content = "\n".join(" | ".join(c for c in r) for r in rows)
                        slide_blocks.append(
                            ParsedBlock(
                                block_type=BlockType.TABLE,
                                content=md_content,
                                page_number=slide_idx,
                                bbox=bbox,
                                metadata={
                                    **base_meta,
                                    "rows": len(rows),
                                    "cols": len(table.columns),
                                    "headers": headers,
                                    "has_header": bool(headers and any(headers)),
                                    "row_cells": rows,
                                },
                            )
                        )
                    continue

                # 3. Picture / image
                if str(shape.shape_type) in ("PICTURE (13)", "13", "MSO_SHAPE_TYPE.PICTURE"):
                    slide_blocks.append(
                        ParsedBlock(
                            block_type=BlockType.IMAGE,
                            content=f"[image: {shape.name}]",
                            page_number=slide_idx,
                            bbox=bbox,
                            metadata=base_meta,
                        )
                    )
                    continue

                # 4. Text frame (text boxes, callouts, shapes with text)
                if shape.has_text_frame:
                    for p_idx, para in enumerate(shape.text_frame.paragraphs):
                        text = para.text.strip()
                        if not text:
                            continue
                        p_meta = {
                            **base_meta,
                            "paragraph_index": p_idx,
                            "level": para.level,
                        }
                        if para.level > 0:
                            p_meta["list_type"] = "bullet"
                            p_meta["list_level"] = para.level
                            b_type = BlockType.LIST
                        else:
                            b_type = BlockType.TEXT

                        slide_blocks.append(
                            ParsedBlock(
                                block_type=b_type,
                                content=text,
                                page_number=slide_idx,
                                bbox=bbox,
                                metadata=p_meta,
                            )
                        )

            # 5. Speaker notes
            try:
                if slide.has_notes_slide:
                    notes_tf = slide.notes_slide.notes_text_frame
                    if notes_tf and notes_tf.text.strip():
                        slide_blocks.append(
                            ParsedBlock(
                                block_type=BlockType.TEXT,
                                content=notes_tf.text.strip(),
                                page_number=slide_idx,
                                bbox=None,
                                metadata={"region": "speaker_notes", "is_speaker_note": True},
                            )
                        )
            except Exception as exc:
                logger.debug("Failed to extract notes for slide %d: %s", slide_idx, exc)

            pages.append(
                ParsedPage(
                    page_number=slide_idx,
                    width=slide_w_pt,
                    height=slide_h_pt,
                    blocks=slide_blocks,
                )
            )
            total_blocks += len(slide_blocks)

        if not pages or total_blocks == 0:
            raise EmptyDocumentError(f"PPTX '{path}' contains no slides or no extractable content.")

        return ParsedDocument(
            source=str(path),
            parser_name=self.name,
            parser_version=str(getattr(pptx_mod, "__version__", "unknown")),
            metadata=metadata,
            pages=pages,
            blocks=[],
            warnings=[],
        )
