"""Shared helpers for chunker coverage/provenance assertions."""

from __future__ import annotations

from chunkers.contracts.flatten import flatten_document
from parsers.contracts.models import ParsedDocument


def coverage_stats(document: ParsedDocument, chunks) -> dict:
    """Fraction of non-whitespace flattened characters covered by chunks,
    and the fraction of emitted characters that duplicate earlier coverage."""
    text, _ = flatten_document(document)
    n = len(text)
    covered = bytearray(n)
    total_chunk_chars = 0
    for chunk in chunks:
        start, end = chunk.metadata["char_start"], chunk.metadata["char_end"]
        total_chunk_chars += end - start
        for i in range(start, end):
            covered[i] = 1
    non_ws = sum(1 for i, ch in enumerate(text) if not ch.isspace())
    if non_ws == 0:
        return {"coverage": 1.0, "duplicate_rate": 0.0, "total_chunk_chars": total_chunk_chars}
    covered_ws = sum(1 for i, ch in enumerate(text) if not ch.isspace() and covered[i])
    duplicated = max(0, total_chunk_chars - n)
    return {
        "coverage": covered_ws / non_ws,
        "duplicate_rate": duplicated / total_chunk_chars if total_chunk_chars else 0.0,
        "total_chunk_chars": total_chunk_chars,
    }


def flattened_text(document: ParsedDocument) -> str:
    text, _ = flatten_document(document)
    return text


def assert_provenance(document: ParsedDocument, chunks) -> None:
    """Refs must point at real blocks with offsets inside the blocks."""
    text, _ = flatten_document(document)
    pages = {page.page_number: page.blocks for page in document.pages}
    for chunk in chunks:
        assert text[chunk.metadata["char_start"] : chunk.metadata["char_end"]] == chunk.text
        for ref in chunk.block_refs:
            if ref.page_number is None:
                blocks = document.blocks
            else:
                assert ref.page_number in pages, f"unknown page {ref.page_number}"
                blocks = pages[ref.page_number]
            assert 0 <= ref.block_index < len(blocks)
            block = blocks[ref.block_index]
            assert ref.block_type == block.block_type.value
            if ref.char_start is not None:
                assert 0 <= ref.char_start <= ref.char_end <= len(block.content)
