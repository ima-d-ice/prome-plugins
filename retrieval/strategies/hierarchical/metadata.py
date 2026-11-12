"""Metadata contract of the hierarchical retrieval strategy.

The hierarchical chunker already writes the parent/child relationship into the
chunk metadata: ``level`` (``"parent"`` / ``"child"``), ``parent_id``,
``child_ids``, ``child_count`` and ``parent_index`` / ``child_index``.  This
module turns that in-memory relationship into the *filterable* metadata that
the vector store needs, without inventing a second relationship format.

Two names matter for the filters:

``chunk_level``
    ``"parent"`` or ``"child"``.  It is the only field the parent-level and
    child-level searches filter on, so a parent search can never return a
    child record and a child search can never return a parent record.

``parent_id``
    The id of the parent a child belongs to.  The child-level search filters
    on it, so the search is restricted to the children of the selected
    parents *in the store itself* — the restriction is pushed down to the
    backend instead of being applied afterwards.

The field names are prefixed to stay clear of the chunk metadata they are
derived from: the chunker writes ``level`` and ``parent_id`` into the chunk
metadata, the strategy writes ``chunk_level`` and ``parent_id`` into the
record metadata.  ``parent_id`` deliberately keeps its name: it is the same
value, and a different name for the same relationship would be worse than
reusing it.
"""

from __future__ import annotations

from chunkers.contracts.models import Chunk
from vector_stores.contracts.filters import MetadataFilter, and_, eq, is_in

#: Record metadata field that separates the two levels of the hierarchy.
CHUNK_LEVEL_FIELD = "chunk_level"

#: Record metadata field that points a child at its parent.
PARENT_ID_FIELD = "parent_id"

#: Record metadata field that carries the document id.
DOCUMENT_ID_FIELD = "document_id"

#: Record metadata field that carries the number of children of a parent.
CHILD_COUNT_FIELD = "child_count"

#: Record metadata field that carries the ordered ids of the children.
CHILD_IDS_FIELD = "child_ids"

#: Record metadata field that carries the heading path of the chunk.
HEADING_PATH_FIELD = "heading_path"

#: Record metadata field that carries the index of the chunk in its level.
CHUNK_INDEX_FIELD = "chunk_index"

#: Record metadata field that carries the index of a child inside its parent.
CHILD_INDEX_FIELD = "child_index"

#: Record metadata field that carries the source of the document.
SOURCE_FIELD = "source"

#: Value of :data:`CHUNK_LEVEL_FIELD` for a parent record.
LEVEL_PARENT = "parent"

#: Value of :data:`CHUNK_LEVEL_FIELD` for a child record.
LEVEL_CHILD = "child"

#: Chunk metadata key the hierarchical chunker uses for the level.
CHUNK_METADATA_LEVEL = "level"

#: Chunk metadata key the hierarchical chunker uses for the parent link.
CHUNK_METADATA_PARENT_ID = "parent_id"

#: Chunk metadata key the hierarchical chunker uses for the child list.
CHUNK_METADATA_CHILD_IDS = "child_ids"

#: Chunk metadata key the hierarchical chunker uses for the child count.
CHUNK_METADATA_CHILD_COUNT = "child_count"

#: Chunk metadata key the hierarchical chunker uses for the heading path.
CHUNK_METADATA_HEADING_PATH = "heading_path"

#: Chunk metadata key the hierarchical chunker uses for the child index.
CHUNK_METADATA_CHILD_INDEX = "child_index"


def chunk_level(chunk: Chunk) -> str:
    """Return the level of ``chunk`` (``"parent"`` or ``"child"``).

    Raises:
        ValueError: when the chunk carries no hierarchy metadata at all.
    """
    level = chunk.metadata.get(CHUNK_METADATA_LEVEL)
    if level not in (LEVEL_PARENT, LEVEL_CHILD):
        raise ValueError(
            f"chunk {chunk.id!r} has no hierarchy level "
            f"(expected {CHUNK_METADATA_LEVEL!r} to be {LEVEL_PARENT!r} or "
            f"{LEVEL_CHILD!r}, got {level!r}); the chunk was not produced by the "
            "hierarchical chunker"
        )
    return str(level)


def parent_id_of(chunk: Chunk) -> str | None:
    """Return the parent id of a child chunk, or ``None`` for a parent."""
    if chunk_level(chunk) == LEVEL_PARENT:
        return None
    value = chunk.metadata.get(CHUNK_METADATA_PARENT_ID)
    return str(value) if value is not None else None


def record_metadata(chunk: Chunk) -> dict[str, object]:
    """Build the filterable record metadata of ``chunk``.

    The metadata is derived from the chunk metadata the hierarchical chunker
    wrote, so the two can never drift apart.
    """
    level = chunk_level(chunk)
    metadata: dict[str, object] = {
        CHUNK_LEVEL_FIELD: level,
        DOCUMENT_ID_FIELD: chunk.doc_id,
        CHUNK_INDEX_FIELD: chunk.index,
        SOURCE_FIELD: chunk.metadata.get("source", chunk.doc_id),
    }
    heading_path = chunk.metadata.get(CHUNK_METADATA_HEADING_PATH)
    if heading_path:
        metadata[HEADING_PATH_FIELD] = list(heading_path)
    if level == LEVEL_CHILD:
        parent_id = parent_id_of(chunk)
        if parent_id is None:
            raise ValueError(
                f"child chunk {chunk.id!r} carries no {CHUNK_METADATA_PARENT_ID!r}"
            )
        metadata[PARENT_ID_FIELD] = parent_id
        # The index of the child *inside* its parent: it is what orders the
        # siblings when the adjacent context is expanded.
        child_index = chunk.metadata.get(CHUNK_METADATA_CHILD_INDEX)
        if child_index is not None:
            metadata[CHILD_INDEX_FIELD] = int(child_index)
    else:
        child_ids = chunk.metadata.get(CHUNK_METADATA_CHILD_IDS) or []
        metadata[CHILD_COUNT_FIELD] = len(child_ids)
        if child_ids:
            metadata[CHILD_IDS_FIELD] = list(child_ids)
    return metadata


def parent_filter() -> MetadataFilter:
    """Return the filter that selects **only** parent records."""
    return eq(CHUNK_LEVEL_FIELD, LEVEL_PARENT)


def child_filter(parent_ids: list[str] | tuple[str, ...]) -> MetadataFilter:
    """Return the filter that selects the children of ``parent_ids``.

    The filter is the conjunction of "this record is a child" and "this
    record belongs to one of the selected parents", so the restriction is
    applied by the backend rather than after the search.
    """
    return and_(
        eq(CHUNK_LEVEL_FIELD, LEVEL_CHILD),
        is_in(PARENT_ID_FIELD, list(parent_ids)),
    )


def child_filter_for_parent(parent_id: str) -> MetadataFilter:
    """Return the filter that selects the children of a single parent."""
    return and_(
        eq(CHUNK_LEVEL_FIELD, LEVEL_CHILD),
        eq(PARENT_ID_FIELD, parent_id),
    )


def level_filter(level: str) -> MetadataFilter:
    """Return the filter that selects the records of one level."""
    return eq(CHUNK_LEVEL_FIELD, level)
