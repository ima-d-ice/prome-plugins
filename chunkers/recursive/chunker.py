"""Recursive chunker: split along increasingly fine separators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from chunkers._textsplit import DEFAULT_SEPARATORS, merge_spans, recursive_split
from chunkers.contracts.exceptions import ChunkerConfigError
from chunkers.contracts.flatten import build_chunks, flatten_document
from chunkers.contracts.models import Chunk
from chunkers.contracts.tokenizers import TokenCounter, ensure_counter
from chunkers.contracts.validation import validate_size_overlap
from parsers.contracts.models import ParsedDocument


@dataclass(frozen=True)
class RecursiveConfig:
    """Configuration for :class:`RecursiveChunker`.

    ``separators`` are tried in order; the first one present in a too-large
    span wins, then smaller separators are applied recursively inside the
    pieces. The default prefers document/paragraph boundaries, then lines,
    sentences, then words. ``""`` as the final separator is the hard-cut
    fallback. Limits are measured with ``tokenizer`` (default: characters).
    """

    chunk_size: int = 1000
    overlap: int = 200
    tokenizer: TokenCounter | Callable[[str], int] | None = None
    separators: tuple[str, ...] = DEFAULT_SEPARATORS

    def __post_init__(self) -> None:
        validate_size_overlap(self.chunk_size, self.overlap, prefix="recursive")
        try:
            object.__setattr__(self, "tokenizer", ensure_counter(self.tokenizer))
        except TypeError as exc:
            raise ChunkerConfigError(str(exc)) from exc
        if not isinstance(self.separators, (tuple, list)):
            raise ChunkerConfigError("separators must be a tuple of strings")
        seps = tuple(self.separators)
        for sep in seps:
            if not isinstance(sep, str):
                raise ChunkerConfigError(f"separators must be strings, got {sep!r}")
        if not seps:
            raise ChunkerConfigError("separators must not be empty; use ('',) for a hard cut only")
        # Drop interior duplicates while preserving order, keep one "" at the end if present.
        seen: set[str] = set()
        deduped: list[str] = []
        for sep in seps:
            if sep not in seen:
                seen.add(sep)
                deduped.append(sep)
        object.__setattr__(self, "separators", tuple(deduped))


class RecursiveChunker:
    """Split a ParsedDocument along hierarchical separators, then pack greedily.

    Text is recursively divided with the configured separators until every
    piece fits the limit; pieces are then merged left-to-right into chunks of
    at most ``chunk_size`` units with ``overlap`` units of trailing context
    carried into the next chunk. This is NOT hierarchical (parent/child)
    chunking: output is a flat list of chunks. Deterministic for a given
    document and config.
    """

    name = "recursive"

    def __init__(self, config: RecursiveConfig | None = None) -> None:
        self.config = config or RecursiveConfig()

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        if not isinstance(document, ParsedDocument):
            raise TypeError(f"RecursiveChunker expects a ParsedDocument, got {type(document).__name__}")
        counter = self.config.tokenizer  # always normalized by __post_init__
        text, flats = flatten_document(document)
        if not text.strip():
            return []
        atoms = recursive_split(text, counter, 0, len(text), self.config.separators, 0, self.config.chunk_size)
        spans = merge_spans(text, counter, atoms, self.config.chunk_size, self.config.overlap)
        return build_chunks(document, text, flats, spans, strategy=self.name, counter=counter)
