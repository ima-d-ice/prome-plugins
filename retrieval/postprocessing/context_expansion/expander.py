"""Context expansion for the hierarchical retrieval strategy.

Retrieving the right child chunk is only half of the answer: a reader also
needs to know *where* in the document the evidence sits.  This package
expands a matched child into a small, explicit context bundle:

- the **parent section** that contains it, and
- optionally the **adjacent children** of the same parent.

Two rules shape the implementation:

* the context is *additional*, never a replacement: the matched child stays
  the evidence of the result and is never overwritten by its parent;
* nothing is rewritten.  No summarization, no rewriting of the source text --
  the context is the original chunk text, with its provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from retrieval.contracts.models import ContextChunk


@dataclass(frozen=True)
class ContextExpansionConfig:
    """Settings of :class:`ContextExpander`.

    Args:
        include_parent: whether the parent section is attached.
        include_adjacent: whether neighbouring children are attached.
        max_adjacent: how many neighbouring children are attached in total.
            The nearest neighbours are taken first, so a budget of one yields
            the single closest sibling.
    """

    include_parent: bool = True
    include_adjacent: bool = False
    max_adjacent: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.max_adjacent, int) or isinstance(self.max_adjacent, bool):
            raise ValueError(
                f"max_adjacent must be an integer, got {self.max_adjacent!r}"
            )
        if self.max_adjacent < 0:
            raise ValueError(f"max_adjacent must be >= 0, got {self.max_adjacent}")


@dataclass
class ExpandedContext:
    """The context bundle of one result."""

    parent: ContextChunk | None = None
    adjacent: tuple[ContextChunk, ...] = ()
    notes: tuple[str, ...] = ()
    """Human-readable notes about what could not be expanded."""


class ContextExpander:
    """Expands a matched child into its parent section and its neighbours.

    The expander never touches the backend: it works on the records the
    retrieval stages already produced, so it can be tested on its own.

    Args:
        parents: the parent records, keyed by their chunk id.
        children: the child records, keyed by their chunk id.
    """

    def __init__(
        self,
        parents: dict[str, Any] | None = None,
        children: dict[str, Any] | None = None,
    ) -> None:
        self.parents = dict(parents or {})
        self.children = dict(children or {})

    # ------------------------------------------------------------------
    def expand(
        self,
        child: Any,
        *,
        config: ContextExpansionConfig | None = None,
    ) -> ExpandedContext:
        """Return the context bundle of ``child``.

        Args:
            child: the matched child record.
            config: the expansion settings; the defaults are used when it is
                omitted.
        """
        settings = config or ContextExpansionConfig()
        notes: list[str] = []
        parent_context: ContextChunk | None = None
        adjacent: tuple[ContextChunk, ...] = ()

        parent_id = self._parent_id(child)
        parent = self.parents.get(parent_id) if parent_id else None

        if settings.include_parent:
            if parent is None:
                notes.append(
                    f"the parent {parent_id!r} of the matched child is not available"
                )
            else:
                parent_context = self._as_context_chunk(parent, role="parent")

        if settings.include_adjacent and settings.max_adjacent > 0:
            adjacent, adjacent_notes = self._adjacent(child, parent_id, settings)
            notes.extend(adjacent_notes)

        return ExpandedContext(
            parent=parent_context,
            adjacent=adjacent,
            notes=tuple(notes),
        )

    # ------------------------------------------------------------------
    def _adjacent(
        self,
        child: Any,
        parent_id: str | None,
        settings: ContextExpansionConfig,
    ) -> tuple[tuple[ContextChunk, ...], list[str]]:
        """Return the neighbours of ``child`` inside the same parent."""
        notes: list[str] = []
        if not parent_id:
            return (), ["the matched child carries no parent reference"]
        ordered = self._siblings(parent_id)
        if not ordered:
            return (), [f"no sibling of the matched child is available for {parent_id!r}"]
        child_id = self._chunk_id(child)
        try:
            position = next(
                index for index, item in enumerate(ordered) if self._chunk_id(item) == child_id
            )
        except StopIteration:
            return (), ["the matched child is not part of its parent's child list"]

        # The nearest neighbours come first: distance 1 before distance 2, and
        # within a distance the earlier sibling wins.  The budget is a total.
        candidates = sorted(
            (
                (abs(index - position), index)
                for index in range(len(ordered))
                if index != position
            ),
        )
        selected: list[ContextChunk] = []
        seen_text = {self._text(child)}
        evidence_document = self._document_id(child)
        for _, index in candidates:
            if len(selected) >= settings.max_adjacent:
                break
            neighbour = ordered[index]
            # The expansion never leaves the document of the evidence.
            if evidence_document and self._document_id(neighbour) != evidence_document:
                continue
            text = self._text(neighbour)
            if not text or text in seen_text:
                # Never repeat the same content twice.
                continue
            seen_text.add(text)
            selected.append(self._as_context_chunk(neighbour, role="adjacent"))
        return tuple(selected), notes

    def _siblings(self, parent_id: str) -> list[Any]:
        """Return the children of ``parent_id`` in document order."""
        parent = self.parents.get(parent_id)
        if parent is not None:
            child_ids = self._metadata(parent).get("child_ids") or ()
            ordered = [self.children[cid] for cid in child_ids if cid in self.children]
            if ordered:
                return ordered
        # Fall back to the children that declare this parent, ordered by their
        # index inside the parent.
        candidates = [
            child
            for child in self.children.values()
            if self._parent_id(child) == parent_id
        ]
        candidates.sort(key=lambda item: self._metadata(item).get("child_index", 0))
        return candidates

    # ------------------------------------------------------------------
    @staticmethod
    def _as_context_chunk(record: Any, *, role: str) -> ContextChunk:
        """Turn a record into a context chunk."""
        metadata = dict(record.metadata) if hasattr(record, "metadata") else {}
        return ContextChunk(
            chunk_id=str(getattr(record, "id", "") or getattr(record, "chunk_id", "")),
            text=str(getattr(record, "text", "") or ""),
            role=role,
            doc_id=str(getattr(record, "doc_id", "") or ""),
            metadata=metadata,
            provenance=_provenance(record),
        )

    @staticmethod
    def _metadata(record: Any) -> dict[str, Any]:
        metadata = getattr(record, "metadata", None)
        if isinstance(metadata, dict):
            return metadata
        return {}

    @staticmethod
    def _chunk_id(record: Any) -> str:
        return str(getattr(record, "id", "") or getattr(record, "chunk_id", ""))

    @staticmethod
    def _text(record: Any) -> str:
        return str(getattr(record, "text", "") or "")

    @staticmethod
    def _document_id(record: Any) -> str:
        return str(getattr(record, "doc_id", "") or "")

    @staticmethod
    def _parent_id(record: Any) -> str | None:
        metadata = ContextExpander._metadata(record)
        value = metadata.get("parent_id")
        return str(value) if value is not None else None


def _provenance(record: Any) -> Any:
    """Return the provenance of ``record``, when it has one."""
    provenance = getattr(record, "provenance", None)
    if provenance is not None:
        return provenance
    from chunkers.contracts.models import Chunk
    from vector_stores.contracts.models import ChunkProvenance

    if isinstance(record, Chunk):
        return ChunkProvenance.from_chunk(record)
    return None
