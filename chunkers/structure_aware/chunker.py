"""Structure-aware chunker: respect the parser's extracted structure."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from chunkers._textsplit import DEFAULT_SEPARATORS, merge_spans, recursive_split
from chunkers.contracts.exceptions import ChunkerConfigError
from chunkers.contracts.flatten import build_chunks, flatten_document, trim_span
from chunkers.contracts.models import Chunk
from chunkers.contracts.tokenizers import TokenCounter, ensure_counter
from chunkers.contracts.validation import validate_size_overlap
from parsers.contracts.models import BlockType, ParsedDocument


@dataclass(frozen=True)
class StructureAwareConfig:
    """Configuration for :class:`StructureAwareChunker`.

    ``chunk_size`` is measured with ``tokenizer`` (default: characters).
    When ``close_on_heading`` is true (default), a heading whose level is the
    same as or higher than the current section (i.e. it opens a new major
    section) closes the accumulated chunk before the heading. Headings
    missing a usable ``heading_level`` in their block metadata are treated as
    level 1.
    """

    chunk_size: int = 1000
    tokenizer: TokenCounter | Callable[[str], int] | None = None
    close_on_heading: bool = True

    def __post_init__(self) -> None:
        validate_size_overlap(self.chunk_size, 0, prefix="structure_aware")
        try:
            object.__setattr__(self, "tokenizer", ensure_counter(self.tokenizer))
        except TypeError as exc:
            raise ChunkerConfigError(str(exc)) from exc
        if not isinstance(self.close_on_heading, bool):
            raise ChunkerConfigError("close_on_heading must be a boolean")


def _heading_level(block) -> int:
    raw = block.metadata.get("heading_level") if block.metadata else None
    try:
        level = int(raw)  # accepts 2 and "2"
    except (TypeError, ValueError):
        level = 1
    return max(1, min(level, 8))


def _heading_text(block) -> str:
    return block.content.strip()


class StructureAwareChunker:
    """Chunk by section while respecting the parsed block structure.

    Blocks are emitted in reading order. The chunker:

    - holds headings as *pending* and glues them to the content that follows,
      so a heading is never orphaned from its section body;
    - opens a new chunk at major section boundaries (a heading whose level is
      the same as or higher than the current section) when
      ``close_on_heading`` is enabled;
    - keeps every other block atomic where possible: lists and tables are
      never silently split or flattened;
    - closes the current chunk when adding the next block would exceed
      ``chunk_size`` (block-boundary split, never mid-block);
    - splits a single block that exceeds ``chunk_size`` by recursively
      descending separators, packing the pieces up to size;
    - records the active heading path in each chunk's metadata and preserves
      page/block provenance.

    Output is a flat list of chunks in document order; this is NOT
    parent/child (hierarchical) chunking. Deterministic.
    """

    name = "structure_aware"

    def __init__(self, config: StructureAwareConfig | None = None) -> None:
        self.config = config or StructureAwareConfig()

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        if not isinstance(document, ParsedDocument):
            raise TypeError(
                f"StructureAwareChunker expects a ParsedDocument, got {type(document).__name__}"
            )
        counter = self.config.tokenizer
        text, flats = flatten_document(document)
        if not flats:
            return []

        spans: list[tuple[int, int, str | None]] = []
        paths: list[list[dict]] = []
        heading_stack: list[tuple[int, str]] = []  # (level, text) of open sections
        current: list[tuple[int, int]] = []
        pending_headings: list[tuple[int, int]] = []
        current_heading: str | None = None
        current_path: list[dict] = []

        def section_heading() -> str | None:
            return heading_stack[-1][1] if heading_stack else None

        def section_path() -> list[dict]:
            return [{"level": level, "text": t} for level, t in heading_stack]

        def bind_context() -> None:
            nonlocal current_heading, current_path
            current_heading = section_heading()
            current_path = section_path()

        def close_current() -> None:
            nonlocal current, current_heading, current_path
            if current:
                spans.append((current[0][0], current[-1][1], current_heading))
                paths.append(list(current_path))
                current = []
                current_heading = None
                current_path = []

        def flush_pending_as_chunk() -> None:
            nonlocal pending_headings
            if not pending_headings:
                return
            bind_context()
            spans.append((pending_headings[0][0], pending_headings[-1][1], current_heading))
            paths.append(list(current_path))
            pending_headings = []

        def take_pending_into_current() -> None:
            nonlocal pending_headings
            if pending_headings:
                if not current:
                    bind_context()
                current.extend(pending_headings)
                pending_headings = []

        def emit_oversized_atoms(block_start: int, block_end: int) -> None:
            atoms = recursive_split(
                text, counter, block_start, block_end, DEFAULT_SEPARATORS, 0, self.config.chunk_size
            )
            index = 0
            while index < len(atoms):
                take_pending_into_current()
                if current:
                    atom_end = atoms[index][1]
                    if counter.count(text[current[0][0] : atom_end]) <= self.config.chunk_size:
                        current.append(atoms[index])
                        index += 1
                        continue
                    close_current()
                    continue
                # Nothing open: pack the remaining atoms as tight chunks.
                bind_context()
                for piece_start, piece_end in merge_spans(
                    text, counter, atoms[index:], self.config.chunk_size, 0
                ):
                    p_start, p_end = trim_span(text, piece_start, piece_end)
                    if p_start >= p_end:
                        continue
                    spans.append((p_start, p_end, current_heading))
                    paths.append(list(current_path))
                index = len(atoms)

        for fb in flats:
            block = fb.block
            span = (fb.start, fb.end)
            if block.block_type == BlockType.HEADING:
                level = _heading_level(block)
                if (
                    self.config.close_on_heading
                    and current
                    and heading_stack
                    and level <= heading_stack[-1][0]
                ):
                    close_current()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, _heading_text(block)))
                pending_headings.append(span)
                pending_size = counter.count(text[pending_headings[0][0] : pending_headings[-1][1]])
                if pending_size > self.config.chunk_size:
                    flush_pending_as_chunk()
                continue

            # Non-heading block.
            block_size = counter.count(text[fb.start : fb.end])
            if current:
                effective_start = current[0][0]  # includes pending headings physically
            elif pending_headings:
                effective_start = pending_headings[0][0]
            else:
                effective_start = fb.start
            if current and counter.count(text[effective_start : fb.end]) > self.config.chunk_size:
                close_current()

            if block_size > self.config.chunk_size:
                emit_oversized_atoms(fb.start, fb.end)
                continue
            take_pending_into_current()
            if not current:
                bind_context()
            current.append(span)

        # End of document: flush anything still waiting.
        if pending_headings:
            if current:
                total_end = pending_headings[-1][1]
                if counter.count(text[current[0][0] : total_end]) <= self.config.chunk_size:
                    current.extend(pending_headings)
                    pending_headings = []
                    close_current()
                else:
                    close_current()  # content without the headings
                    current = list(pending_headings)  # headings open a trailing chunk
                    bind_context()
                    pending_headings = []
                    close_current()
            else:
                take_pending_into_current()
        close_current()

        chunks = build_chunks(document, text, flats, spans, strategy=self.name, counter=counter)
        path_iter = iter(paths)
        for chunk in chunks:
            path = next(path_iter, [])
            chunk.metadata["heading_path"] = [item["text"] for item in path]
            chunk.metadata["heading_level"] = path[-1]["level"] if path else None
            chunk.metadata["block_types"] = sorted({ref.block_type for ref in chunk.block_refs})
            chunk.metadata["block_count"] = len(chunk.block_refs)
        return chunks
