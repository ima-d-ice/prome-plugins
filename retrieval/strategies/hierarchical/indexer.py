"""Indexing of hierarchical (parent/child) chunks into a vector store.

Both levels of the hierarchy are embedded **independently** with the same
encoder and written into the *same* collection and the *same* named dense
vector space.  What distinguishes a parent record from a child record is the
``chunk_level`` metadata field, not a second collection: a single collection
with a single embedding space keeps the two levels comparable and makes the
parent/child filters portable across every backend.

Guarantees:

- the original chunk ids are preserved;
- the parent/child relationship survives a restart of the store, because it
  is written into the record metadata (``parent_id`` on a child,
  ``child_ids`` / ``child_count`` on a parent);
- the page and block provenance of every chunk is preserved;
- the embedding-space identity of the collection is enforced;
- indexing is idempotent: re-indexing a document replaces its records
  instead of duplicating them;
- a parent and a child of *different* documents are never linked.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from chunkers.contracts.models import Chunk
from chunkers.hierarchical.chunker import HierarchicalResult
from embeddings.contracts.models import (
    EmbeddingInput,
    EmbeddingSpaceDescriptor,
    RepresentationType,
)
from embeddings.contracts.protocols import EmbeddingModel
from retrieval.contracts.exceptions import HierarchicalIntegrityError
from retrieval.strategies.hierarchical.config import HierarchicalIndexConfig
from retrieval.strategies.hierarchical.metadata import (
    CHUNK_LEVEL_FIELD,
    PARENT_ID_FIELD,
    chunk_level,
    record_metadata,
)
from vector_stores.contracts.filters import eq
from vector_stores.contracts.indexes import PayloadFieldType, PayloadIndexSpec
from vector_stores.contracts.models import (
    CollectionConfig,
    DistanceMetric,
    SpaceVectors,
    StoredRecord,
    VectorSpaceConfig,
)
from vector_stores.contracts.protocols import VectorStore


@dataclass
class IndexReport:
    """What an indexing run wrote."""

    parents: int = 0
    children: int = 0
    replaced: int = 0
    """Number of records that already existed and were replaced."""
    removed: int = 0
    """Number of stale records of the document that were removed."""

    @property
    def total(self) -> int:
        return self.parents + self.children


def build_collection_config(
    model: EmbeddingModel,
    name: str,
    *,
    space_name: str = "dense",
    description: str | None = None,
    distance: DistanceMetric = DistanceMetric.COSINE,
) -> CollectionConfig:
    """Build the collection configuration for a hierarchical collection.

    The collection declares one named dense vector space whose embedding space
    is the one of ``model``.  Parents and children share it; they are told
    apart by their metadata.

    The two fields the strategies filter on -- ``chunk_level`` and
    ``parent_id`` -- are declared as payload indexes, because a backend with a
    fixed schema (LanceDB) can only filter on the fields the collection
    declares.
    """
    descriptor = model.get_embedding_space()
    if descriptor.representation_type is not RepresentationType.DENSE:
        raise ValueError(
            "the hierarchical indexer needs a dense embedding model, got "
            f"{descriptor.representation_type.value!r}"
        )
    dimensions = descriptor.dimensions
    if not dimensions:
        raise ValueError("the embedding model does not declare its dimensions")
    space = VectorSpaceConfig(
        name=space_name,
        vector_type=RepresentationType.DENSE,
        embedding_space=descriptor,
        dimensions=dimensions,
        distance=distance,
    )
    return CollectionConfig(
        name=name,
        vector_spaces={space_name: space},
        description=description,
        payload_indexes=(
            PayloadIndexSpec(field=CHUNK_LEVEL_FIELD, field_type=PayloadFieldType.KEYWORD),
            PayloadIndexSpec(field=PARENT_ID_FIELD, field_type=PayloadFieldType.KEYWORD),
        ),
    )


def validate_hierarchy(result: HierarchicalResult) -> None:
    """Check that the parent/child links of ``result`` are consistent.

    Raises:
        HierarchicalIntegrityError: when a child references a missing parent,
            when a parent and a child belong to different documents, or when
            a chunk id is used twice.
    """
    parents_by_id: dict[str, Chunk] = {}
    for parent in result.parents:
        if parent.id in parents_by_id:
            raise HierarchicalIntegrityError(
                f"duplicate parent chunk id {parent.id!r}"
            )
        parents_by_id[parent.id] = parent

    seen_children: set[str] = set()
    for child in result.children:
        if child.id in seen_children:
            raise HierarchicalIntegrityError(f"duplicate child chunk id {child.id!r}")
        seen_children.add(child.id)
        if child.id in parents_by_id:
            raise HierarchicalIntegrityError(
                f"chunk id {child.id!r} is used by both a parent and a child"
            )
        parent_id = child.metadata.get("parent_id")
        if not parent_id:
            raise HierarchicalIntegrityError(
                f"child chunk {child.id!r} carries no 'parent_id'"
            )
        parent = parents_by_id.get(parent_id)
        if parent is None:
            raise HierarchicalIntegrityError(
                f"child chunk {child.id!r} references the unknown parent "
                f"{parent_id!r}"
            )
        if parent.doc_id != child.doc_id:
            raise HierarchicalIntegrityError(
                f"child chunk {child.id!r} belongs to document {child.doc_id!r} but "
                f"its parent {parent.id!r} belongs to document {parent.doc_id!r}; "
                "a parent and its child must share the document"
            )

    # The parent's own child list must match the children that point at it.
    for parent in result.parents:
        declared = list(parent.metadata.get("child_ids") or [])
        actual = [
            child.id
            for child in result.children
            if child.metadata.get("parent_id") == parent.id
        ]
        if declared and declared != actual:
            raise HierarchicalIntegrityError(
                f"parent {parent.id!r} declares the children {declared} but the "
                f"hierarchy contains {actual}"
            )


class HierarchicalIndexer:
    """Writes hierarchical chunks into a vector store.

    Args:
        store: the vector store to write to.
        embedding_model: the injected embedding model (never loaded here).
        collection: name of the collection to write to.
        config: indexing settings.
    """

    def __init__(
        self,
        store: VectorStore,
        embedding_model: EmbeddingModel,
        *,
        collection: str,
        config: HierarchicalIndexConfig | None = None,
    ) -> None:
        self.store = store
        self.embedding_model = embedding_model
        self.collection = collection
        self.config = config or HierarchicalIndexConfig()

    # ------------------------------------------------------------------
    # collection
    # ------------------------------------------------------------------
    def collection_config(self) -> CollectionConfig:
        """Return the collection configuration this indexer expects."""
        return build_collection_config(
            self.embedding_model,
            self.collection,
            space_name=self.config.vector_space,
            description=self.config.collection_description,
        )

    def ensure_collection(self, *, exist_ok: bool = True) -> None:
        """Create the collection when it does not exist yet."""
        from vector_stores.contracts.exceptions import CollectionExistsError

        try:
            self.store.create_collection(self.collection_config(), exist_ok=exist_ok)
        except CollectionExistsError:
            if not exist_ok:
                raise

    # ------------------------------------------------------------------
    # indexing
    # ------------------------------------------------------------------
    def index_result(
        self,
        result: HierarchicalResult,
        *,
        replace: bool = True,
    ) -> IndexReport:
        """Index the parents and the children of ``result``.

        Args:
            result: the hierarchical chunking result to index.
            replace: when true, the records of the affected documents are
                removed first, so re-indexing a document with a different
                chunking configuration does not leave stale records behind.

        Returns:
            An :class:`IndexReport` describing what was written.
        """
        validate_hierarchy(result)
        parents = list(result.parents)
        children = list(result.children)
        if not parents and not children:
            return IndexReport()

        documents = {chunk.doc_id for chunk in parents + children}
        existing: set[str] = set()
        if replace:
            for doc_id in sorted(documents):
                existing |= self._document_ids(doc_id)
                self.store.delete_by_document(self.collection, doc_id)

        parent_vectors = self._embed([chunk.text for chunk in parents])
        child_vectors = self._embed([chunk.text for chunk in children])

        records: list[StoredRecord] = []
        for chunk, vector in zip(parents, parent_vectors):
            records.append(self._record(chunk, vector))
        for chunk, vector in zip(children, child_vectors):
            records.append(self._record(chunk, vector))

        if records:
            self.store.upsert(self.collection, records)
        return IndexReport(
            parents=len(parents),
            children=len(children),
            replaced=len(existing & {record.chunk.id for record in records}),
            removed=len(existing),
        )

    def index_document(
        self,
        document: Any,
        *,
        chunker: Any | None = None,
        replace: bool = True,
    ) -> IndexReport:
        """Chunk ``document`` with the hierarchical chunker and index it."""
        if chunker is None:
            from chunkers.hierarchical.chunker import HierarchicalChunker

            chunker = HierarchicalChunker()
        return self.index_result(chunker.chunk(document), replace=replace)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _document_ids(self, doc_id: str) -> set[str]:
        """Return the ids already stored for ``doc_id``."""
        found: set[str] = set()
        offset: int | str | None = None
        while True:
            page = self.store.scroll(
                self.collection,
                limit=200,
                offset=offset,
                filter=eq("doc_id", doc_id),
            )
            for record in page.records:
                found.add(record.chunk.id)
            if page.next_offset is None:
                break
            offset = page.next_offset
        return found

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts`` with the injected model."""
        if not texts:
            return []
        inputs = [EmbeddingInput(text=text) for text in texts]
        results = self.embedding_model.embed_documents(inputs)
        if len(results) != len(inputs):
            raise ValueError(
                f"the embedding model returned {len(results)} results for "
                f"{len(inputs)} inputs"
            )
        vectors: list[list[float]] = []
        for result in results:
            vector = result.dense_vector
            if not vector:
                raise ValueError(
                    "the embedding model returned no dense vector for a chunk"
                )
            vectors.append(list(vector))
        return vectors

    def _record(self, chunk: Chunk, vector: list[float]) -> StoredRecord:
        """Build the stored record of ``chunk``."""
        descriptor: EmbeddingSpaceDescriptor = self.embedding_model.get_embedding_space()
        metadata = record_metadata(chunk)
        # The chunk keeps its own relationship metadata; the record metadata
        # adds the filterable projection of it.
        chunk.metadata.setdefault(CHUNK_LEVEL_FIELD, chunk_level(chunk))
        space = self.config.vector_space
        return StoredRecord(
            chunk=chunk,
            vectors={
                space: SpaceVectors(
                    space=space,
                    embedding_space=descriptor,
                    dense=list(vector),
                )
            },
            metadata=metadata,
        )
