"""Hierarchical chunker: parent/child (one level) chunking.

Small, specific chunks are what a retrieval system should match against, but
answering from a 300-character fragment usually needs more context. The
hierarchical chunker produces both levels from the same document:

- **parents** -- larger, section-aware units that carry the heading context;
- **children** -- smaller units carved out of a parent, suitable for
  embedding and retrieval.

Only children are meant to be embedded and indexed. A parent is the context
that a future retrieval system fetches *after* a child matched, so a vector
store is not required to hold parent vectors at all.

The implementation reuses the existing structure-aware chunker for the parent
level and the shared recursive splitter for the child level, so no splitting
logic is duplicated here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from chunkers._textsplit import DEFAULT_SEPARATORS, merge_spans, recursive_split
from chunkers.contracts.exceptions import ChunkerConfigError
from chunkers.contracts.flatten import build_chunks, chunk_digest, flatten_document
from chunkers.contracts.models import Chunk
from chunkers.contracts.tokenizers import TokenCounter, ensure_counter
from chunkers.contracts.validation import validate_size_overlap
from chunkers.structure_aware.chunker import StructureAwareChunker, StructureAwareConfig
from parsers.contracts.models import ParsedDocument

_STRATEGY = "hierarchical"


@dataclass(frozen=True)
class HierarchicalConfig:
    """Configuration for :class:`HierarchicalChunker`.

    ``parent_size`` and ``child_size`` are measured with ``tokenizer``
    (default: characters). ``child_overlap`` is the trailing context carried
    from one child into the next, in the same unit.

    ``child_size`` must not exceed ``parent_size`` and ``child_overlap``
    must be smaller than ``child_size``.
    """

    parent_size: int = 2000
    child_size: int = 500
    child_overlap: int = 50
    tokenizer: TokenCounter | Callable[[str], int] | None = None

    def __post_init__(self) -> None:
        validate_size_overlap(self.parent_size, 0, prefix="hierarchical")
        validate_size_overlap(self.child_size, self.child_overlap, prefix="hierarchical child")
        if self.child_size > self.parent_size:
            raise ChunkerConfigError(
                f"hierarchical child_size ({self.child_size}) must not exceed "
                f"parent_size ({self.parent_size})"
            )
        try:
            object.__setattr__(self, "tokenizer", ensure_counter(self.tokenizer))
        except TypeError as exc:
            raise ChunkerConfigError(str(exc)) from exc


@dataclass
class HierarchicalResult:
    """The output of :meth:`HierarchicalChunker.chunk`.

    Holds the two levels plus the parent-child relationship:

    - ``parents`` -- parent chunks in document order;
    - ``children`` -- child chunks in document order (all parents' children
      concatenated, so child indices are document-global);
    - each child carries ``metadata["parent_id"]``; each parent carries
      ``metadata["child_ids"]``.

    Only the children are intended for embedding and indexing; a parent is
    fetched later to provide context.
    """

    parents: list[Chunk] = field(default_factory=list)
    children: list[Chunk] = field(default_factory=list)
    _parents_by_id: dict[str, Chunk] = field(default_factory=dict, repr=False, init=False)
    _children_by_parent: dict[str, list[str]] = field(default_factory=dict, repr=False, init=False)

    def __post_init__(self) -> None:
        self._parents_by_id = {chunk.id: chunk for chunk in self.parents}
        by_parent: dict[str, list[str]] = {}
        for child in self.children:
            parent_id = child.metadata.get("parent_id")
            if parent_id is None:
                continue
            by_parent.setdefault(parent_id, []).append(child.id)
        self._children_by_parent = by_parent

    def parent_of(self, child: Chunk | str) -> Chunk | None:
        """Return the parent of ``child`` (a :class:`Chunk` or a child id)."""
        child_id = child.id if isinstance(child, Chunk) else child
        parent_id = self._child_parent_ids().get(child_id)
        return self._parents_by_id.get(parent_id) if parent_id else None

    def children_of(self, parent: Chunk | str) -> list[Chunk]:
        """Return the children of ``parent`` (a :class:`Chunk` or a parent id)."""
        parent_id = parent.id if isinstance(parent, Chunk) else parent
        child_ids = self._children_by_parent.get(parent_id, [])
        index = {child.id: child for child in self.children}
        return [index[child_id] for child_id in child_ids if child_id in index]

    def _child_parent_ids(self) -> dict[str, str]:
        return {
            child.id: child.metadata["parent_id"]
            for child in self.children
            if child.metadata.get("parent_id") is not None
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "parents": [chunk.as_dict() for chunk in self.parents],
            "children": [chunk.as_dict() for chunk in self.children],
        }


class HierarchicalChunker:
    """Produce parent and child chunks from a parsed document.

    The parent level is computed with the structure-aware chunker at
    ``parent_size``, so parents respect major section boundaries, keep
    their heading context and never orphan a heading from its body. Each
    parent is then split into children of at most ``child_size`` with
    ``child_overlap`` of trailing context, using the shared recursive
    splitter.

    Guarantees:

    - every child references an existing parent (``metadata["parent_id"]``);
    - every parent lists its children (``metadata["child_ids"]``);
    - children partition their parent, so nothing is dropped;
    - ids are stable and deterministic for a given document and config;
    - the input document is never mutated.
    """

    name = "hierarchical"

    def __init__(self, config: HierarchicalConfig | None = None) -> None:
        self.config = config or HierarchicalConfig()

    def chunk(self, document: ParsedDocument) -> HierarchicalResult:
        """Chunk ``document`` into parents and children."""
        if not isinstance(document, ParsedDocument):
            raise TypeError(
                f"HierarchicalChunker expects a ParsedDocument, got {type(document).__name__}"
            )
        counter = self.config.tokenizer
        text, flats = flatten_document(document)
        if not flats:
            return HierarchicalResult()

        parents = self._build_parents(document, counter)
        if not parents:
            return HierarchicalResult()

        children = self._build_children(document, text, flats, parents, counter)
        return HierarchicalResult(parents=parents, children=children)

    def chunk_children(self, document: ParsedDocument) -> list[Chunk]:
        """Return only the child chunks (the units meant for indexing)."""
        return self.chunk(document).children

    # ------------------------------------------------------------------
    def _build_parents(self, document: ParsedDocument, counter: TokenCounter) -> list[Chunk]:
        """Compute the parent level with the structure-aware chunker."""
        parent_chunker = StructureAwareChunker(
            StructureAwareConfig(
                chunk_size=self.config.parent_size,
                tokenizer=counter,
                close_on_heading=True,
            )
        )
        parents = parent_chunker.chunk(document)
        digest = chunk_digest(document.source)
        for index, parent in enumerate(parents):
            parent.id = f"{digest}-{_STRATEGY}-parent-{index:04d}"
            parent.metadata["strategy"] = _STRATEGY
            parent.metadata["level"] = "parent"
            parent.metadata["parent_id"] = parent.id
            parent.metadata["child_ids"] = []
            parent.metadata["child_count"] = 0
        return parents

    def _build_children(
        self,
        document: ParsedDocument,
        text: str,
        flats: list,
        parents: list[Chunk],
        counter: TokenCounter,
    ) -> list[Chunk]:
        """Split every parent into children and link the two levels."""
        digest = chunk_digest(document.source)
        children: list[Chunk] = []
        links: list[tuple[Chunk, list[Chunk]]] = []

        for parent in parents:
            parent_start = parent.metadata["char_start"]
            parent_end = parent.metadata["char_end"]
            atoms = recursive_split(
                text, counter, parent_start, parent_end, DEFAULT_SEPARATORS, 0, self.config.child_size
            )
            spans = merge_spans(
                text, counter, atoms, self.config.child_size, self.config.child_overlap
            )
            local = build_chunks(
                document,
                text,
                flats,
                [(start, end, parent.heading) for start, end in spans],
                strategy=_STRATEGY,
                counter=counter,
            )
            if not local:
                # Defensive: a parent always yields at least one child, but
                # never leave a parent without children (no orphan children
                # and no empty parents).
                continue
            for child_index, child in enumerate(local):
                child.metadata["level"] = "child"
                child.metadata["parent_id"] = parent.id
                child.metadata["parent_index"] = parent.index
                child.metadata["child_index"] = child_index
                if "heading_path" in parent.metadata:
                    child.metadata["heading_path"] = list(parent.metadata["heading_path"])
                if "heading_level" in parent.metadata:
                    child.metadata["heading_level"] = parent.metadata["heading_level"]
                child.metadata["block_types"] = sorted(
                    {ref.block_type for ref in child.block_refs}
                )
                child.metadata["block_count"] = len(child.block_refs)
                child.metadata["child_ids"] = []
                child.metadata["child_count"] = 0
            children.extend(local)
            links.append((parent, local))

        # Global child indexing and stable ids across the whole document.
        for index, child in enumerate(children):
            child.id = f"{digest}-{_STRATEGY}-child-{index:04d}"
            child.index = index
            child.count = len(children)
        # Now that the final child ids are known, link both directions.
        for parent, local in links:
            parent.metadata["child_ids"] = [child.id for child in local]
            parent.metadata["child_count"] = len(local)
        return children

    def iter_pairs(self, document: ParsedDocument) -> Iterator[tuple[Chunk, list[Chunk]]]:
        """Yield ``(parent, children)`` pairs in document order."""
        result = self.chunk(document)
        for parent in result.parents:
            yield parent, result.children_of(parent)
