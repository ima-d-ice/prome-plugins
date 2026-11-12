"""Lightweight chunk data model shared by all Prome RAG chunkers.

A Chunk is a slice of a parsed document plus provenance back to the source
document and the blocks it was built from. Chunkers never mutate the
ParsedDocument; they only read from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BlockReference:
    """Points a chunk (or part of it) back at a block in the source document.

    ``block_index`` is the index of the block within its container:
    ``page.blocks`` for paged formats (with ``page_number`` set), or
    ``doc.blocks`` for pageless formats (``page_number`` is None). The pair
    ``(page_number, block_index)`` therefore identifies a block unambiguously.

    ``char_start``/``char_end`` are offsets of the referenced slice into the
    block's own ``content``; ``None`` means the whole block.
    """

    block_index: int
    page_number: int | None
    block_type: str
    char_start: int | None = None
    char_end: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "block_index": self.block_index,
            "page_number": self.page_number,
            "block_type": self.block_type,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


@dataclass
class Chunk:
    """A single chunk of a parsed document.

    ``doc_id`` is the identity of the source document; ParsedDocument has no
    dedicated id field, so ``doc_id == ParsedDocument.source``.

    ``count`` is the total number of chunks produced for the document (same
    value on every chunk of a single run). ``length`` is measured in
    ``length_unit`` (``"characters"`` by default) and is NOT claimed to be a
    model token count unless a tokenizer was configured. ``heading`` carries
    optional section context (last section heading covering this chunk);
    strategies that do not use structure leave it ``None``.

    ``metadata`` carries chunker-specific extras. All built-in chunkers set
    ``source``, ``parser_name``, ``parser_version``, ``strategy``,
    ``char_start``/``char_end`` (offsets into the flattened document text,
    see ``chunkers.contracts.flatten``) and ``pages``.
    """

    id: str
    doc_id: str
    text: str
    index: int
    count: int
    length: int
    length_unit: str
    block_refs: list[BlockReference] = field(default_factory=list)
    heading: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "text": self.text,
            "index": self.index,
            "count": self.count,
            "length": self.length,
            "length_unit": self.length_unit,
            "block_refs": [ref.as_dict() for ref in self.block_refs],
            "heading": self.heading,
            "metadata": self.metadata,
        }
