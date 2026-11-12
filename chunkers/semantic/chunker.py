"""Semantic chunker: meaning-aware chunk boundaries via embeddings.

Where the structure-aware chunker uses the parser's headings and block
types to decide where a chunk may end, the semantic chunker decides by
comparing *what the text says*. Candidate segments are embedded with a
pluggable embedding model and adjacent segments whose cosine similarity
falls below a threshold are treated as belonging to different topics.

The embedding model is injected and must implement the shared
:class:`~embeddings.contracts.protocols.EmbeddingModel` interface. This
module never loads a model, never hardcodes a provider and never falls back
to another chunking strategy if embedding inference fails -- it raises
instead.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from chunkers._textsplit import DEFAULT_SEPARATORS, recursive_split, split_keep
from chunkers.contracts.exceptions import ChunkerConfigError, ChunkerError
from chunkers.contracts.flatten import build_chunks, flatten_document, trim_span
from chunkers.contracts.models import Chunk
from chunkers.contracts.tokenizers import TokenCounter, ensure_counter
from chunkers.contracts.validation import validate_size_overlap
from embeddings.contracts.models import EmbeddingInput, RepresentationType
from parsers.contracts.models import BlockType, ParsedDocument

_STRATEGY = "semantic"

# Sentence boundary: a terminator, optional closing punctuation, then
# whitespace. The whitespace belongs to the sentence it closes, so no
# character is lost when the segments are reassembled.
_SENTENCE_RE = re.compile(r"[.!?…]+[\"'’”\)\]]*\s+")

# Blocks that are meaningful as a unit and are not cut at sentence
# boundaries. Structured blocks (tables, lists, code, spreadsheet records)
# would lose their meaning if split mid-way.
_ATOMIC_BLOCK_TYPES = frozenset(
    {
        BlockType.HEADING,
        BlockType.TABLE,
        BlockType.LIST,
        BlockType.CODE,
        BlockType.RECORD,
        BlockType.KEY_VALUE,
        BlockType.ATTACHMENT,
        BlockType.IMAGE,
        BlockType.DEFINITION,
        BlockType.SEGMENT,
        BlockType.FRAME,
    }
)


class SemanticChunkerError(ChunkerError):
    """The semantic chunker could not complete (e.g. embedding failure)."""


@dataclass(frozen=True)
class SemanticConfig:
    """Configuration for :class:`SemanticChunker`.

    ``embedding_model`` is required and must implement
    :class:`~embeddings.contracts.protocols.EmbeddingModel`, i.e. provide
    ``embed_documents(inputs) -> list[EmbeddingResult]``.

    ``similarity_threshold`` is the **minimum adjacent cosine similarity**
    required for two consecutive segments to stay in the same chunk. A
    similarity *below* the threshold marks a topic change and therefore a
    chunk boundary. The threshold lives in ``[-1.0, 1.0]``; higher values
    split more aggressively, lower values keep more text together. The
    default of ``0.5`` is a conservative "clearly different topic" cut.

    ``max_chunk_size`` is the hard upper bound for a chunk and
    ``min_chunk_size`` an optional lower bound: a chunk below the minimum
    is merged into a neighbour when that neighbour is in the same section
    and the merged chunk still fits ``max_chunk_size``.

    ``overlap`` is the amount of trailing context (measured with the same
    counter) carried from one chunk into the next. Overlap is expressed in
    whole segments, so it never cuts a segment in half, and it never
    crosses a section boundary.

    ``batch_size`` is the number of segments handed to the embedding model
    per ``embed_documents`` call.

    ``respect_structure`` enables section boundaries: a heading starts a new
    chunk and is glued to the content that follows it, and segments are
    never merged across a heading.

    Sizes are measured with ``tokenizer`` (default: characters).
    """

    embedding_model: Any
    similarity_threshold: float = 0.5
    max_chunk_size: int = 1000
    min_chunk_size: int | None = None
    overlap: int = 0
    tokenizer: TokenCounter | Callable[[str], int] | None = None
    batch_size: int = 32
    respect_structure: bool = True

    def __post_init__(self) -> None:
        if self.embedding_model is None:
            raise ChunkerConfigError(
                "semantic chunking requires an embedding model; pass "
                "embedding_model=<an EmbeddingModel instance>"
            )
        if not callable(getattr(self.embedding_model, "embed_documents", None)):
            raise ChunkerConfigError(
                "embedding_model must provide an embed_documents(inputs) method; got "
                f"{type(self.embedding_model).__name__}"
            )
        if not isinstance(self.similarity_threshold, (int, float)) or isinstance(
            self.similarity_threshold, bool
        ):
            raise ChunkerConfigError(
                f"similarity_threshold must be a number, got {self.similarity_threshold!r}"
            )
        if not -1.0 <= float(self.similarity_threshold) <= 1.0:
            raise ChunkerConfigError(
                f"similarity_threshold must be within [-1.0, 1.0], got {self.similarity_threshold}"
            )
        validate_size_overlap(self.max_chunk_size, self.overlap, prefix="semantic")
        if self.min_chunk_size is not None:
            if (
                not isinstance(self.min_chunk_size, int)
                or isinstance(self.min_chunk_size, bool)
                or self.min_chunk_size < 1
            ):
                raise ChunkerConfigError(
                    f"min_chunk_size must be a positive integer or None, got {self.min_chunk_size!r}"
                )
            if self.min_chunk_size > self.max_chunk_size:
                raise ChunkerConfigError(
                    f"min_chunk_size ({self.min_chunk_size}) must not exceed "
                    f"max_chunk_size ({self.max_chunk_size})"
                )
        if (
            not isinstance(self.batch_size, int)
            or isinstance(self.batch_size, bool)
            or self.batch_size < 1
        ):
            raise ChunkerConfigError(f"batch_size must be a positive integer, got {self.batch_size!r}")
        if not isinstance(self.respect_structure, bool):
            raise ChunkerConfigError("respect_structure must be a boolean")
        try:
            object.__setattr__(self, "tokenizer", ensure_counter(self.tokenizer))
        except TypeError as exc:
            raise ChunkerConfigError(str(exc)) from exc


@dataclass(frozen=True)
class _Unit:
    """A candidate segment: the smallest unit that is embedded and compared."""

    start: int
    end: int
    block_index: int
    page_number: int | None
    block_type: str
    is_heading: bool
    section: int
    heading_text: str | None


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity of two dense vectors, in ``[-1.0, 1.0]``.

    A zero-length vector has no direction, so it is reported as ``0.0``
    similarity rather than raising: a degenerate embedding means "no
    information", not a crash.
    """
    if len(left) != len(right):
        raise SemanticChunkerError(
            f"cannot compare vectors of different dimensions: {len(left)} != {len(right)}"
        )
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


class SemanticChunker:
    """Chunk a parsed document along semantic topic boundaries.

    The algorithm:

    1. flatten the document into blocks with character offsets;
    2. extract candidate segments -- sentences for prose blocks, whole
       blocks for structured ones (tables, lists, code, records);
    3. split segments that are larger than ``max_chunk_size`` using the
       shared recursive splitter (atomic blocks are split on line
       boundaries so rows stay intact);
    4. embed all segments in batches with the injected embedding model,
       through its ``embed_documents`` operation;
    5. compare adjacent segments with cosine similarity;
    6. cut a boundary wherever the similarity drops below
       ``similarity_threshold``, at a heading, or where the size limit
       would be exceeded;
    7. merge undersized chunks into a neighbour in the same section when
       ``min_chunk_size`` is set;
    8. carry ``overlap`` of trailing context into the next chunk.

    Content is never dropped and block references stay accurate: every
    chunk is an exact slice of the flattened document text. Output is
    deterministic for a given document, configuration and embedding model.
    """

    name = "semantic"

    def __init__(self, config: SemanticConfig | None = None) -> None:
        if config is None:
            raise ChunkerConfigError(
                "SemanticChunker requires an embedding model: "
                "SemanticChunker(SemanticConfig(embedding_model=...))"
            )
        self.config = config

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        if not isinstance(document, ParsedDocument):
            raise TypeError(
                f"SemanticChunker expects a ParsedDocument, got {type(document).__name__}"
            )
        counter = self.config.tokenizer
        text, flats = flatten_document(document)
        if not flats:
            return []

        units = self._build_units(text, flats, counter)
        if not units:
            return []

        embeddings = self._embed_units(text, units)
        groups = self._group_units(text, units, embeddings, counter)
        spans = self._apply_overlap(text, groups, counter)

        # build_chunks drops whitespace-only spans; do the same filtering
        # here so groups and chunks stay aligned one-to-one.
        kept: list[tuple[int, int, list[_Unit]]] = []
        for group, (start, end) in zip(groups, spans):
            trimmed = trim_span(text, start, end)
            if trimmed[0] >= trimmed[1]:
                continue
            kept.append((trimmed[0], trimmed[1], group))

        chunks = build_chunks(
            document,
            text,
            flats,
            [(start, end, self._heading_for(group)) for start, end, group in kept],
            strategy=_STRATEGY,
            counter=counter,
        )
        self._annotate(chunks, kept, embeddings)
        return chunks

    # ------------------------------------------------------------------
    # Candidate segments
    # ------------------------------------------------------------------
    def _build_units(self, text: str, flats: list, counter: TokenCounter) -> list[_Unit]:
        units: list[_Unit] = []
        section = 0
        current_heading: str | None = None
        for fb in flats:
            block = fb.block
            is_heading = block.block_type == BlockType.HEADING
            if is_heading:
                # Sections are only tracked when the structure is respected;
                # otherwise a heading must not separate anything.
                if self.config.respect_structure:
                    section += 1
                current_heading = block.content.strip() or None
            for start, end in self._segment_spans(text, fb, block.block_type, counter):
                units.append(
                    _Unit(
                        start=start,
                        end=end,
                        block_index=fb.block_index,
                        page_number=fb.page_number,
                        block_type=block.block_type.value,
                        is_heading=is_heading,
                        section=section,
                        heading_text=current_heading,
                    )
                )
        return units

    def _segment_spans(
        self, text: str, fb, block_type: BlockType, counter: TokenCounter
    ) -> list[tuple[int, int]]:
        """Split one block into candidate segments."""
        limit = self.config.max_chunk_size
        if block_type in _ATOMIC_BLOCK_TYPES:
            return self._split_atomic(text, fb.start, fb.end, counter)

        spans: list[tuple[int, int]] = []
        for start, end in self._sentence_spans(text, fb.start, fb.end):
            if counter.count(text[start:end]) <= limit:
                spans.append((start, end))
            else:
                # Too large for one chunk: fall back to the shared
                # recursive splitter (paragraph, line, sentence, word).
                spans.extend(
                    recursive_split(text, counter, start, end, DEFAULT_SEPARATORS, 0, limit)
                )
        return spans

    def _split_atomic(
        self, text: str, start: int, end: int, counter: TokenCounter
    ) -> list[tuple[int, int]]:
        """Keep an atomic block whole; split it on line boundaries if too large.

        Rows of a table or items of a list stay intact; only a single
        oversized row may exceed the limit (same best-effort behaviour as
        the other chunkers).
        """
        limit = self.config.max_chunk_size
        if counter.count(text[start:end]) <= limit:
            return [(start, end)]
        groups: list[tuple[int, int]] = []
        current: tuple[int, int] | None = None
        for line_start, line_end in split_keep(text, start, end, "\n"):
            if current is None:
                current = (line_start, line_end)
            elif counter.count(text[current[0] : line_end]) <= limit:
                current = (current[0], line_end)
            else:
                groups.append(current)
                current = (line_start, line_end)
        if current is not None:
            groups.append(current)
        return groups

    @staticmethod
    def _sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
        """Split a prose range at sentence boundaries."""
        body = text[start:end]
        if not body.strip():
            return [(start, end)]
        spans: list[tuple[int, int]] = []
        cursor = start
        for match in _SENTENCE_RE.finditer(body):
            cut = start + match.end()
            if cut > cursor:
                spans.append((cursor, cut))
            cursor = cut
        if cursor < end:
            spans.append((cursor, end))
        return spans or [(start, end)]

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    def _embed_units(self, text: str, units: list[_Unit]) -> list[list[float]]:
        model = self.config.embedding_model
        batch_size = self.config.batch_size
        vectors: list[list[float]] = []
        for start in range(0, len(units), batch_size):
            batch = units[start : start + batch_size]
            inputs = [
                EmbeddingInput(
                    text=text[unit.start : unit.end].strip(),
                    doc_id=None,
                    chunk_id=None,
                )
                for unit in batch
            ]
            try:
                results = model.embed_documents(inputs)
            except Exception as exc:
                raise SemanticChunkerError(
                    f"embedding inference failed for segment batch {start // batch_size}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if results is None or len(results) != len(batch):
                raise SemanticChunkerError(
                    f"embedding model returned {0 if results is None else len(results)} results "
                    f"for {len(batch)} segments"
                )
            for index, result in enumerate(results):
                vectors.append(self._dense_vector(result, batch[index], index))

        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) > 1:
            raise SemanticChunkerError(
                f"embedding model returned inconsistent dimensions: {sorted(dimensions)}"
            )
        dimension = dimensions.pop() if dimensions else 0
        if dimension <= 0:
            raise SemanticChunkerError("embedding model returned empty vectors")
        return vectors

    @staticmethod
    def _dense_vector(result, unit: _Unit, index: int) -> list[float]:
        """Extract and validate the dense vector of an embedding result."""
        representation = getattr(result, "representation_type", RepresentationType.DENSE)
        if representation != RepresentationType.DENSE:
            raise SemanticChunkerError(
                f"embedding model returned a {representation.value} representation at "
                f"segment {index}; only dense embeddings support cosine similarity"
            )
        vector = getattr(result, "dense_vector", None)
        if vector is None:
            raise SemanticChunkerError(
                f"embedding model returned no dense vector for segment {index}"
            )
        vector = [float(value) for value in vector]
        if not vector:
            raise SemanticChunkerError(
                f"embedding model returned an empty vector for segment {index}"
            )
        declared = getattr(result, "dimensions", None)
        if isinstance(declared, int) and not isinstance(declared, bool) and declared != len(vector):
            raise SemanticChunkerError(
                f"embedding dimension mismatch at segment {index}: "
                f"result declares {declared}, vector has {len(vector)} elements"
            )
        for value in vector:
            if not math.isfinite(value):
                raise SemanticChunkerError(
                    f"embedding model returned a non-finite value at segment {index}"
                )
        return vector

    # ------------------------------------------------------------------
    # Grouping
    # ------------------------------------------------------------------
    def _group_units(
        self, text: str, units: list[_Unit], embeddings: list[list[float]], counter: TokenCounter
    ) -> list[list[_Unit]]:
        """Group adjacent units into chunks.

        Headings are *not* compared by similarity: a heading is a title, not
        body text, so embedding it next to a paragraph says nothing about
        topical coherence. A heading is instead glued to the content that
        follows it and, when ``respect_structure`` is enabled, it opens a
        new chunk.
        """
        max_size = self.config.max_chunk_size
        threshold = float(self.config.similarity_threshold)
        respect = self.config.respect_structure

        groups: list[list[_Unit]] = []
        current: list[_Unit] = []
        last_content_index: int | None = None

        for index, unit in enumerate(units):
            if unit.is_heading and respect and current:
                # A heading opens a new chunk and takes its content with it.
                groups.append(current)
                current = []
                last_content_index = None

            if current:
                prospective = counter.count(text[current[0].start : unit.end])
                # Never split a lone heading away from the content it
                # introduces; an oversized unit is kept whole (best effort).
                heading_only = len(current) == 1 and current[0].is_heading
                if prospective > max_size and not heading_only:
                    groups.append(current)
                    current = []
                    last_content_index = None

            if not unit.is_heading and current and last_content_index is not None:
                similarity = _cosine_similarity(
                    embeddings[last_content_index], embeddings[index]
                )
                if similarity < threshold:
                    groups.append(current)
                    current = []
                    last_content_index = None

            current.append(unit)
            if not unit.is_heading:
                last_content_index = index

        if current:
            groups.append(current)

        return self._enforce_min_size(text, groups, counter)

    def _enforce_min_size(
        self, text: str, groups: list[list[_Unit]], counter: TokenCounter
    ) -> list[list[_Unit]]:
        minimum = self.config.min_chunk_size
        if minimum is None:
            return groups

        max_size = self.config.max_chunk_size
        merged: list[list[_Unit]] = []
        index = 0
        while index < len(groups):
            group = groups[index]
            if self._span_size(text, group[0].start, group[-1].end, counter) >= minimum:
                merged.append(group)
                index += 1
                continue

            # Prefer growing the following chunk (keeps document order natural).
            if index + 1 < len(groups):
                following = groups[index + 1]
                if (
                    following[0].section == group[0].section
                    and self._span_size(text, group[0].start, following[-1].end, counter) <= max_size
                ):
                    groups[index + 1] = group + following
                    del groups[index]
                    continue

            # Otherwise grow the previously accepted chunk.
            if merged and merged[-1][-1].section == group[0].section:
                previous = merged[-1]
                if (
                    self._span_size(text, previous[0].start, group[-1].end, counter) <= max_size
                ):
                    merged[-1] = previous + group
                    del groups[index]
                    continue

            merged.append(group)
            index += 1
        return merged

    # ------------------------------------------------------------------
    # Spans and overlap
    # ------------------------------------------------------------------
    def _apply_overlap(
        self, text: str, groups: list[list[_Unit]], counter: TokenCounter
    ) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        for index, group in enumerate(groups):
            start = group[0].start
            end = group[-1].end
            if index > 0 and self.config.overlap > 0:
                previous = groups[index - 1]
                # Never carry context across a section boundary.
                if previous[-1].section == group[0].section:
                    carry = 0
                    for previous_unit in reversed(previous):
                        size = self._span_size(
                            text, previous_unit.start, previous_unit.end, counter
                        )
                        if carry + size > self.config.overlap:
                            break
                        carry += size
                        start = previous_unit.start
            spans.append((start, end))
        return spans

    @staticmethod
    def _heading_for(group: list[_Unit]) -> str | None:
        for unit in group:
            if unit.is_heading:
                return unit.heading_text
        return group[0].heading_text

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def _annotate(
        self,
        chunks: list[Chunk],
        kept: list[tuple[int, int, list[_Unit]]],
        embeddings: list[list[float]],
    ) -> None:
        model = self.config.embedding_model
        model_id = getattr(model, "model_id", None) or getattr(model, "name", None)
        dimension = len(embeddings[0]) if embeddings else None
        snapshot: dict[str, Any] = {
            "similarity_threshold": float(self.config.similarity_threshold),
            "max_chunk_size": self.config.max_chunk_size,
            "min_chunk_size": self.config.min_chunk_size,
            "overlap": self.config.overlap,
            "batch_size": self.config.batch_size,
            "respect_structure": self.config.respect_structure,
        }
        for chunk, (start, end, group) in zip(chunks, kept):
            chunk.metadata["embedding_model"] = model_id
            chunk.metadata["embedding_dimension"] = dimension
            chunk.metadata["semantic_config"] = dict(snapshot)
            chunk.metadata["semantic_threshold"] = float(self.config.similarity_threshold)
            chunk.metadata["semantic_max_chunk_size"] = self.config.max_chunk_size
            chunk.metadata["semantic_min_chunk_size"] = self.config.min_chunk_size
            chunk.metadata["semantic_overlap"] = self.config.overlap
            chunk.metadata["semantic_batch_size"] = self.config.batch_size
            chunk.metadata["semantic_respect_structure"] = self.config.respect_structure
            chunk.metadata["semantic_threshold_kind"] = "minimum_adjacent_cosine_similarity"
            chunk.metadata["section_index"] = group[0].section
            chunk.metadata["segment_count"] = len(group)
            chunk.metadata["segment_start_index"] = group[0].start
            chunk.metadata["block_types"] = sorted({unit.block_type for unit in group})

    # ------------------------------------------------------------------
    @staticmethod
    def _span_size(text: str, start: int, end: int, counter: TokenCounter) -> int:
        return counter.count(text[start:end])
