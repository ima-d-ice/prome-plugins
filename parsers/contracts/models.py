"""Lightweight common data model shared by all Prome RAG parsers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BlockType(str, Enum):
    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    IMAGE = "image"
    LIST = "list"
    CODE = "code"
    LINK = "link"
    # Structured records: spreadsheet rows, JSON/XML objects, email bodies.
    RECORD = "record"
    # A single key/value pair with its JSON/XML path (leaf nodes).
    KEY_VALUE = "key_value"
    # An email attachment (references the attachment, does not parse it).
    ATTACHMENT = "attachment"
    # A source-code definition (function/class/method) with line range.
    DEFINITION = "definition"
    # An audio transcript segment with start/end timestamps (seconds).
    SEGMENT = "segment"
    # A video frame with a timestamp (seconds) and frame number.
    FRAME = "frame"
    OTHER = "other"


@dataclass
class BoundingBox:
    """Axis-aligned bounding box in PDF points (1/72 inch), origin top-left."""

    x0: float
    y0: float
    x1: float
    y1: float

    def as_dict(self) -> dict[str, float]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


@dataclass
class ParsedBlock:
    block_type: BlockType
    content: str
    page_number: int | None = None  # 1-based for paged formats (PDF); None for pageless formats (HTML)
    bbox: BoundingBox | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "block_type": self.block_type.value,
            "content": self.content,
            "page_number": self.page_number,
            "bbox": self.bbox.as_dict() if self.bbox else None,
            "metadata": self.metadata,
        }


@dataclass
class ParsedPage:
    page_number: int  # 1-based
    width: float | None = None
    height: float | None = None
    blocks: list[ParsedBlock] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(b.content for b in self.blocks if b.content)

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "blocks": [b.as_dict() for b in self.blocks],
        }


@dataclass
class ParsedDocument:
    source: str
    parser_name: str
    parser_version: str
    metadata: dict[str, Any] = field(default_factory=dict)
    pages: list[ParsedPage] = field(default_factory=list)
    # Top-level blocks for pageless formats (e.g. HTML) that carry no page
    # structure. Paged formats (PDF) keep blocks inside pages instead.
    blocks: list[ParsedBlock] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def block_count(self) -> int:
        return len(self.blocks) + sum(len(p.blocks) for p in self.pages)

    @property
    def text(self) -> str:
        parts = [p.text for p in self.pages if p.text]
        parts.extend(b.content for b in self.blocks if b.content)
        return "\n\n".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "parser_name": self.parser_name,
            "parser_version": self.parser_version,
            "metadata": self.metadata,
            "pages": [p.as_dict() for p in self.pages],
            "blocks": [b.as_dict() for b in self.blocks],
            "warnings": self.warnings,
        }
