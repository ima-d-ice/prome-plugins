"""Internal text-splitting primitives shared by chunkers.

Not part of the public contracts. Works on absolute character offsets into a
shared source string so every span maps back to the original document text.
Spans are always contiguous and cover the input range without duplication,
which keeps merged chunks as exact slices of the source text.
"""

from __future__ import annotations

from chunkers.contracts.tokenizers import TokenCounter

DEFAULT_SEPARATORS: tuple[str, ...] = (
    "\n\n",  # paragraph / block boundary
    "\n",    # line boundary
    ". ",    # sentence
    "? ",
    "! ",
    "; ",
    ", ",
    " ",
    "",     # hard character cut (last resort)
)

Span = tuple[int, int]


def hard_char_split(start: int, end: int, window: int) -> list[Span]:
    window = max(1, window)
    return [(s, min(s + window, end)) for s in range(start, end, window)]


def split_keep(text: str, start: int, end: int, sep: str) -> list[Span]:
    """Split ``[start, end)`` on ``sep`` keeping the separator attached to the
    piece it terminates. Pieces are contiguous and cover the whole range."""
    pieces: list[Span] = []
    cursor = start
    while cursor < end:
        idx = text.find(sep, cursor, end)
        if idx == -1:
            pieces.append((cursor, end))
            break
        piece_end = idx + len(sep)
        if piece_end > cursor:
            pieces.append((cursor, piece_end))
        cursor = piece_end
    return pieces


def recursive_split(
    text: str,
    counter: TokenCounter,
    start: int,
    end: int,
    separators: tuple[str, ...],
    sep_index: int,
    chunk_size: int,
) -> list[Span]:
    """Split ``[start, end)`` into atomic spans, each within ``chunk_size``.

    Prefers earlier separators (document/section breaks first, then paragraphs,
    lines, sentences, words) and only falls back to a hard cut when no
    separator exists at deeper levels.
    """
    if start >= end:
        return []
    if counter.count(text[start:end]) <= chunk_size:
        return [(start, end)]
    sep = None
    while sep_index < len(separators):
        candidate = separators[sep_index]
        if candidate == "":
            sep = ""
            break
        if candidate in text[start:end]:
            sep = candidate
            break
        sep_index += 1
    if sep is None or sep == "":
        window = chunk_size if counter.unit == "characters" else chunk_size * 4
        return hard_char_split(start, end, window)
    atoms: list[Span] = []
    for piece_start, piece_end in split_keep(text, start, end, sep):
        if counter.count(text[piece_start:piece_end]) <= chunk_size:
            atoms.append((piece_start, piece_end))
        else:
            atoms.extend(
                recursive_split(text, counter, piece_start, piece_end, separators, sep_index + 1, chunk_size)
            )
    return atoms


def merge_spans(
    text: str,
    counter: TokenCounter,
    atoms: list[Span],
    chunk_size: int,
    overlap: int,
) -> list[Span]:
    """Greedily pack atomic spans into chunks of at most ``chunk_size``.

    After a chunk is emitted, the longest trailing run of atoms whose counted
    size fits within ``overlap`` starts the next chunk. The whole emitted
    buffer is never carried over, which keeps the algorithm making progress
    even when overlap is large.
    """
    chunks: list[Span] = []
    buffer: list[Span] = []

    for atom in atoms:
        if buffer:
            candidate_end = atom[1]
            if counter.count(text[buffer[0][0] : candidate_end]) > chunk_size:
                chunks.append((buffer[0][0], buffer[-1][1]))
                buffer_end = buffer[-1][1]
                keep: list[Span] = []
                for take in range(1, len(buffer)):  # never carry the whole buffer
                    candidate = buffer[len(buffer) - take :]
                    if counter.count(text[candidate[0][0] : buffer_end]) <= overlap:
                        keep = candidate
                    else:
                        break
                buffer = keep
        buffer.append(atom)
    if buffer:
        chunks.append((buffer[0][0], buffer[-1][1]))
    return chunks
