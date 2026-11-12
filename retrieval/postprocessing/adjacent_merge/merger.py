"""Merging of adjacent chunks.

A retrieved passage often ends in the middle of an explanation.  The
adjacent-chunk merger joins a hit with the chunk that follows it (and,
optionally, the one that precedes it) when they belong to the same document
and to the same section.

Three rules keep the merge honest:

* only chunks of the **same document** are merged;
* only chunks that are **consecutive** in the document are merged, so two
  unrelated sections are never glued together;
* the merged text never exceeds the configured size limit.

The merged item keeps the id of the chunk it was built from, so the original
evidence references stay intact, and it records the ids of the chunks that
contributed to it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from retrieval.contracts.models import ContextChunk, RetrievedItem
from vector_stores.contracts.models import ChunkProvenance


@dataclass(frozen=True)
class AdjacentMergeConfig:
    """Settings of :class:`AdjacentMerger`.

    Args:
        enabled: whether the merging is applied at all.
        max_merged_chars: the maximum length of a merged text.
        include_previous: whether the preceding chunk is merged too.
        max_merged_chunks: how many chunks a merged item may contain.
    """

    enabled: bool = False
    max_merged_chars: int = 2000
    include_previous: bool = False
    max_merged_chunks: int = 3

    def __post_init__(self) -> None:
        if self.max_merged_chars < 1:
            raise ValueError(
                f"max_merged_chars must be >= 1, got {self.max_merged_chars}"
            )
        if self.max_merged_chunks < 1:
            raise ValueError(
                f"max_merged_chunks must be >= 1, got {self.max_merged_chunks}"
            )


@dataclass
class AdjacentMergeReport:
    """What a merging run did."""

    merged: int = 0
    """Number of items that were merged with a neighbour."""
    skipped: int = 0
    """Number of items that could not be merged."""

    notes: list[str] = field(default_factory=list)


class AdjacentMerger:
    """Merges retrieved items with their adjacent chunks.

    Args:
        chunks: the chunks of the corpus, keyed by their chunk id.  The
            merger needs them to find the neighbour of a hit.
        config: the merging settings.
    """

    def __init__(
        self,
        chunks: Mapping[str, Any] | None = None,
        config: AdjacentMergeConfig | None = None,
    ) -> None:
        self.chunks = dict(chunks or {})
        self.config = config or AdjacentMergeConfig()

    # ------------------------------------------------------------------
    def merge(
        self, items: Sequence[RetrievedItem]
    ) -> tuple[list[RetrievedItem], AdjacentMergeReport]:
        """Merge the items with their neighbours.

        Args:
            items: the items to merge.

        Returns:
            The merged items and a report of what happened.
        """
        report = AdjacentMergeReport()
        if not self.config.enabled or not items:
            return list(items), report

        merged: list[RetrievedItem] = []
        for item in items:
            result = self._merge_one(item, report)
            merged.append(result)
        return merged, report

    # ------------------------------------------------------------------
    def _merge_one(
        self, item: RetrievedItem, report: AdjacentMergeReport
    ) -> RetrievedItem:
        """Merge one item with its neighbour, when that is possible."""
        neighbour = self._find_neighbour(item)
        if neighbour is None:
            report.skipped += 1
            return item

        parts = [item.text]
        if self.config.include_previous:
            previous = self._find_previous(item)
            if previous is not None:
                parts.insert(0, previous.text)
        parts.append(neighbour.text)

        if len(parts) > self.config.max_merged_chunks:
            parts = parts[-self.config.max_merged_chunks :]
        text = "\n\n".join(part for part in parts if part)
        if len(text) > self.config.max_merged_chars:
            report.skipped += 1
            report.notes.append(
                f"the merged text of {item.chunk_id!r} would exceed "
                f"{self.config.max_merged_chars} characters"
            )
            return item

        report.merged += 1
        contributing = [
            _chunk_id(chunk)
            for chunk in self._contributing(item, neighbour)
        ]
        metadata = dict(item.metadata)
        metadata["merged_chunk_ids"] = contributing
        return RetrievedItem(
            chunk_id=item.chunk_id,
            doc_id=item.doc_id,
            text=text,
            score=item.score,
            score_direction=item.score_direction,
            score_interpretation=item.score_interpretation,
            rank=item.rank,
            strategy=item.strategy,
            metadata=metadata,
            provenance=item.provenance,
            methods=item.methods,
            channel_ranks=dict(item.channel_ranks),
            channel_scores=dict(item.channel_scores),
            context=item.context,
        )

    def _contributing(self, item: RetrievedItem, neighbour: Any) -> list[Any]:
        """Return the chunks that contribute to the merged text."""
        chunks = []
        if self.config.include_previous:
            previous = self._find_previous(item)
            if previous is not None:
                chunks.append(previous)
        chunks.append(item)
        chunks.append(neighbour)
        return chunks

    # ------------------------------------------------------------------
    def _find_neighbour(self, item: RetrievedItem) -> Any | None:
        """Return the chunk that follows ``item`` in the same document."""
        return self._adjacent(item, offset=1)

    def _find_previous(self, item: RetrievedItem) -> Any | None:
        """Return the chunk that precedes ``item`` in the same document."""
        return self._adjacent(item, offset=-1)

    def _adjacent(self, item: RetrievedItem, *, offset: int) -> Any | None:
        """Return the chunk at ``offset`` positions from ``item``."""
        index = self._index_of(item)
        if index is None:
            return None
        target = index + offset
        for chunk in self.chunks.values():
            if self._index_of_chunk(chunk) != target:
                continue
            if getattr(chunk, "doc_id", None) != item.doc_id:
                # Never merge across documents.
                continue
            if chunk.id == item.chunk_id:
                continue
            return chunk
        return None

    def _index_of(self, item: RetrievedItem) -> int | None:
        """Return the position of ``item`` inside its document."""
        chunk = self.chunks.get(item.chunk_id)
        if chunk is not None:
            return self._index_of_chunk(chunk)
        raw = item.metadata.get("chunk_index")
        if isinstance(raw, int):
            return raw
        return None

    @staticmethod
    def _index_of_chunk(chunk: Any) -> int | None:
        """Return the position of ``chunk`` inside its document."""
        raw = getattr(chunk, "index", None)
        if isinstance(raw, int):
            return raw
        metadata = getattr(chunk, "metadata", None)
        if isinstance(metadata, dict):
            value = metadata.get("chunk_index")
            if isinstance(value, int):
                return value
        return None


def build_context_chunk(chunk: Any, *, role: str = "adjacent") -> ContextChunk:
    """Build a context chunk from a corpus chunk."""
    provenance = getattr(chunk, "provenance", None)
    if provenance is None:
        from vector_stores.contracts.models import ChunkProvenance

        provenance = ChunkProvenance.from_chunk(chunk)
    return ContextChunk(
        chunk_id=_chunk_id(chunk),
        text=str(getattr(chunk, "text", "") or ""),
        role=role,
        doc_id=str(getattr(chunk, "doc_id", "") or ""),
        metadata=dict(getattr(chunk, "metadata", {}) or {}),
        provenance=provenance,
    )


def _chunk_id(chunk: Any) -> str:
    """Return the id of a chunk, whatever it is called."""
    for attribute in ("id", "chunk_id"):
        value = getattr(chunk, attribute, None)
        if value:
            return str(value)
    return ""
