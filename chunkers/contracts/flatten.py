"""Canonical flattening of a ParsedDocument into text + block spans.

Every built-in chunker works on the same flattened representation: blocks in
reading order (paged formats iterate pages in order, pageless formats iterate
``doc.blocks``), concatenated with a blank-line separator. Each block's
``[start, end)`` offsets into the flattened text are tracked so any chunk's
character range can be mapped back to the blocks it came from.

A document with both ``pages`` and top-level ``blocks`` is flattened with
pages first (matching ``ParsedDocument.text``); blocks whose content is empty
or all whitespace carry no text and are excluded from the flattened text
(and therefore from block references).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from chunkers.contracts.models import BlockReference, Chunk
from chunkers.contracts.tokenizers import TokenCounter
from parsers.contracts.models import ParsedBlock, ParsedDocument

BLOCK_SEPARATOR = "\n\n"


@dataclass
class FlatBlock:
    """A content-bearing block with its span in the flattened text."""

    block: ParsedBlock
    block_index: int  # index within the containing list (page.blocks or doc.blocks)
    page_number: int | None
    global_index: int  # sequential across the whole document
    start: int
    end: int

    @property
    def content(self) -> str:
        return self.block.content


def _iter_content_blocks(document: ParsedDocument):
    for page in document.pages:
        for index, block in enumerate(page.blocks):
            if block.content and block.content.strip():
                yield block, index, page.page_number
    for index, block in enumerate(document.blocks):
        if block.content and block.content.strip():
            yield block, index, None


def flatten_document(document: ParsedDocument) -> tuple[str, list[FlatBlock]]:
    """Return ``(text, flat_blocks)`` for ``document``.

    ``text`` is the block contents joined by a blank line. ``flat_blocks``
    maps each block to its ``[start, end)`` span in ``text``.
    """
    parts: list[str] = []
    flats: list[FlatBlock] = []
    cursor = 0
    for block, index, page_number in _iter_content_blocks(document):
        start = cursor
        parts.append(block.content)
        cursor += len(block.content)
        flats.append(
            FlatBlock(
                block=block,
                block_index=index,
                page_number=page_number,
                global_index=len(flats),
                start=start,
                end=cursor,
            )
        )
        parts.append(BLOCK_SEPARATOR)
        cursor += len(BLOCK_SEPARATOR)
    text = "".join(parts)
    if flats:
        # Drop the trailing separator; spans recorded above stay valid.
        text = text[: flats[-1].end]
    return text, flats


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Trim leading/trailing whitespace from ``[start, end)``."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def refs_for_range(flats: list[FlatBlock], start: int, end: int) -> list[BlockReference]:
    """Build block references for every block overlapping ``[start, end)``."""
    refs: list[BlockReference] = []
    for fb in flats:
        if fb.end <= start or fb.start >= end:
            continue
        refs.append(
            BlockReference(
                block_index=fb.block_index,
                page_number=fb.page_number,
                block_type=fb.block.block_type.value,
                char_start=max(start, fb.start) - fb.start,
                char_end=min(end, fb.end) - fb.start,
            )
        )
    return refs


def chunk_digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8", errors="surrogatepass")).hexdigest()[:12]


def build_chunks(
    document: ParsedDocument,
    text: str,
    flats: list[FlatBlock],
    spans: list[tuple[int, int] | tuple[int, int, str | None]],
    *,
    strategy: str,
    counter: TokenCounter,
) -> list[Chunk]:
    """Materialize Chunk objects from ``(start, end[, heading])`` spans.

    Spans are expected in document order and must lie within ``text``.
    Whitespace-only spans are dropped; indices and ``count`` are assigned
    after filtering so the output is always a clean 0..n-1 sequence.
    """
    chunks: list[Chunk] = []
    digest = chunk_digest(document.source)
    for span in spans:
        if len(span) == 2:
            start, end = span  # type: ignore[misc]
            heading = None
        else:
            start, end, heading = span  # type: ignore[misc]
        start, end = trim_span(text, max(0, start), min(len(text), end))
        if start >= end:
            continue
        chunk_text = text[start:end]
        pages = sorted({fb.page_number for fb in flats if fb.end > start and fb.start < end and fb.page_number is not None})
        metadata: dict[str, Any] = {
            "source": document.source,
            "parser_name": document.parser_name,
            "parser_version": document.parser_version,
            "strategy": strategy,
            "char_start": start,
            "char_end": end,
            "pages": pages,
        }
        chunks.append(
            Chunk(
                id=f"{digest}-{strategy}-{len(chunks):04d}",
                doc_id=document.source,
                text=chunk_text,
                index=len(chunks),
                count=0,  # assigned below
                length=counter.count(chunk_text),
                length_unit=counter.unit,
                block_refs=refs_for_range(flats, start, end),
                heading=heading,
                metadata=metadata,
            )
        )
    total = len(chunks)
    for index, chunk in enumerate(chunks):
        chunk.index = index
        chunk.count = total
        chunk.id = f"{digest}-{strategy}-{index:04d}"
    return chunks
