"""Fixed-size chunker: uniform windows over the flattened document text."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from chunkers._textsplit import merge_spans
from chunkers.contracts.exceptions import ChunkerConfigError
from chunkers.contracts.flatten import build_chunks, flatten_document
from chunkers.contracts.models import Chunk
from chunkers.contracts.tokenizers import CharacterCounter, TokenCounter, ensure_counter
from chunkers.contracts.validation import validate_size_overlap
from parsers.contracts.models import ParsedDocument

_WORD_RE = re.compile(r"\S+\s*")
# A word-boundary snap may pull the cut point back by at most this fraction
# of the window; otherwise we hard-cut at the configured size.
_MIN_WINDOW_FRACTION = 0.25


@dataclass(frozen=True)
class FixedSizeConfig:
    """Configuration for :class:`FixedSizeChunker`.

    ``chunk_size`` is measured in characters by default; pass ``tokenizer``
    (a TokenCounter or a plain ``str -> int`` callable) to size chunks in
    tokenizer units instead. A single unbreakable word larger than the
    configured size is kept whole (limits are best-effort).
    """

    chunk_size: int = 1000
    overlap: int = 200
    tokenizer: TokenCounter | Callable[[str], int] | None = None
    respect_word_boundaries: bool = True

    def __post_init__(self) -> None:
        validate_size_overlap(self.chunk_size, self.overlap, prefix="fixed")
        if not isinstance(self.respect_word_boundaries, bool):
            raise ChunkerConfigError("respect_word_boundaries must be a boolean")
        try:
            object.__setattr__(self, "tokenizer", ensure_counter(self.tokenizer))
        except TypeError as exc:
            raise ChunkerConfigError(str(exc)) from exc


class FixedSizeChunker:
    """Split a ParsedDocument into fixed-size windows.

    Character mode uses direct character offsets (with word-boundary snapping
    when enabled). Tokenizer mode packs word-like atoms until the counted
    limit is reached, re-measuring overlap in the same units. Windows may
    exceed the configured size only for a single unbreakable unit. Output is
    deterministic for a given document and config.
    """

    name = "fixed"

    def __init__(self, config: FixedSizeConfig | None = None) -> None:
        self.config = config or FixedSizeConfig()

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        if not isinstance(document, ParsedDocument):
            raise TypeError(f"FixedSizeChunker expects a ParsedDocument, got {type(document).__name__}")
        counter = self.config.tokenizer or CharacterCounter()
        text, flats = flatten_document(document)
        if not text.strip():
            return []
        if counter.unit == "characters":
            spans = self._char_spans(text)
        else:
            atoms = [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]
            spans = merge_spans(text, counter, atoms, self.config.chunk_size, self.config.overlap)
        return build_chunks(document, text, flats, spans, strategy=self.name, counter=counter)

    # ------------------------------------------------------------------
    def _char_spans(self, text: str) -> list[tuple[int, int]]:
        size = self.config.chunk_size
        overlap = self.config.overlap
        spans: list[tuple[int, int]] = []
        start = 0
        n = len(text)
        min_progress = max(1, int(size * _MIN_WINDOW_FRACTION))
        while start < n:
            end = min(start + size, n)
            if end < n and self.config.respect_word_boundaries:
                snapped = self._snap_end_to_word(text, start, end, min_progress)
                if snapped is not None:
                    end = snapped
            spans.append((start, end))
            if end >= n:
                break
            next_start = max(end - overlap, start + 1)
            while next_start < n and text[next_start].isspace():
                next_start += 1
            # Snap an overlap rewind that landed inside a word forward to the
            # next word boundary (skipping at most one short word).
            if next_start > 0 and not text[next_start - 1].isspace():
                probe = next_start
                while probe < n and not text[probe].isspace() and probe - next_start < 32:
                    probe += 1
                if probe < n and text[probe].isspace() and probe - next_start < 32:
                    while probe < n and text[probe].isspace():
                        probe += 1
                    next_start = probe
            if next_start <= start:
                next_start = start + 1
            start = next_start
        return spans

    @staticmethod
    def _snap_end_to_word(text: str, start: int, end: int, min_progress: int) -> int | None:
        window = text[start:end]
        idx = max(window.rfind(ws) for ws in (" ", "\n", "\t", "\r"))
        if idx == -1:
            return None
        cut = start + idx + 1
        if cut - start < min_progress:
            return None
        return cut
