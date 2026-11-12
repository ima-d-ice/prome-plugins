"""Core data models of the vector-store ecosystem.

The models deliberately reuse the contracts of the ecosystems they consume:

- a stored record keeps the whole :class:`chunkers.contracts.models.Chunk`
  (stable id, document id, text, index/count, metadata and provenance);
- a vector space is described by the embedding ecosystem's
  :class:`embeddings.contracts.models.EmbeddingSpaceDescriptor`, so the
  storage layer never invents a second embedding-space format;
- vector kinds reuse
  :class:`embeddings.contracts.models.RepresentationType`
  (``dense`` / ``sparse`` / ``multi-vector``).

What the vector store adds on top is storage-side vocabulary that does not
exist upstream: distance metrics, score direction, collection configuration,
search requests and results.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from chunkers.contracts.models import BlockReference, Chunk
from embeddings.contracts.models import (
    EmbeddingResult,
    EmbeddingSpaceDescriptor,
    Normalization,
    RepresentationType,
)
from vector_stores.contracts.filters import MetadataFilter
from vector_stores.contracts.indexes import (
    HnswIndexConfig,
    IndexDescription,
    PayloadFieldType,
    PayloadIndexSpec,
    SparseIndexConfig,
)


class DistanceMetric(str, Enum):
    """Distance / similarity metrics a vector space can use."""

    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"

    @property
    def score_direction(self) -> ScoreDirection:
        """Return whether a larger score means a better match."""
        if self in (DistanceMetric.COSINE, DistanceMetric.DOT):
            return ScoreDirection.HIGHER_IS_BETTER
        return ScoreDirection.LOWER_IS_BETTER


class ScoreDirection(str, Enum):
    """How the raw backend score must be interpreted."""

    HIGHER_IS_BETTER = "higher-is-better"
    LOWER_IS_BETTER = "lower-is-better"


@dataclass(frozen=True)
class VectorSpaceConfig:
    """Configuration of one named vector space inside a collection.

    A collection may hold several named spaces (e.g. a dense space produced by
    one model and a sparse space produced by another).  Vectors that do not
    belong to the space's embedding space are rejected by the adapter.

    Normalization is not repeated here: it lives in
    ``embedding_space.normalization`` so the model identity stays in one place.
    """

    name: str
    vector_type: RepresentationType
    embedding_space: EmbeddingSpaceDescriptor
    dimensions: int | None = None
    """Vector width. Required for dense and multi-vector spaces, ``None`` for sparse."""

    distance: DistanceMetric | None = None
    """Distance / similarity metric. ``None`` means "backend default" (cosine)."""

    hnsw: HnswIndexConfig | None = None
    """Approximate dense index configuration (``None`` = backend default)."""

    sparse: SparseIndexConfig | None = None
    """Sparse inverted index configuration (``None`` = backend default)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Free-form embedding configuration metadata (query/document instructions, ...)."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "vector_type": self.vector_type.value,
            "embedding_space": descriptor_to_dict(self.embedding_space),
            "dimensions": self.dimensions,
            "distance": self.distance.value if self.distance else None,
            "hnsw": self.hnsw.as_dict() if self.hnsw else None,
            "sparse": self.sparse.as_dict() if self.sparse else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CollectionConfig:
    """Declaration of a collection: a named group of stored chunks.

    ``vector_spaces`` maps a space name to its configuration; the name of a
    space is also the name used in search requests.
    """

    name: str
    vector_spaces: dict[str, VectorSpaceConfig]
    description: str | None = None
    payload_indexes: tuple[PayloadIndexSpec, ...] = ()
    """Metadata fields the user explicitly asked to index."""

    metadata: dict[str, Any] = field(default_factory=dict)

    def space(self, name: str) -> VectorSpaceConfig:
        """Return the configuration of the named space.

        Raises:
            KeyError: if the space does not exist.
        """
        try:
            return self.vector_spaces[name]
        except KeyError as exc:
            raise KeyError(
                f"collection {self.name!r} has no vector space {name!r}; "
                f"known spaces: {sorted(self.vector_spaces)}"
            ) from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "vector_spaces": {
                name: space.as_dict() for name, space in self.vector_spaces.items()
            },
            "payload_indexes": [spec.as_dict() for spec in self.payload_indexes],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CollectionConfig:
        """Rebuild a configuration from :meth:`as_dict` output."""
        spaces: dict[str, VectorSpaceConfig] = {}
        for name, raw in (data.get("vector_spaces") or {}).items():
            spaces[name] = vector_space_from_dict(name, raw)
        indexes = tuple(
            PayloadIndexSpec(
                field=spec["field"],
                field_type=_payload_field_type(spec.get("field_type")),
            )
            for spec in (data.get("payload_indexes") or [])
        )
        return cls(
            name=data["name"],
            vector_spaces=spaces,
            description=data.get("description"),
            payload_indexes=indexes,
            metadata=dict(data.get("metadata") or {}),
        )


def _payload_field_type(value: str | None) -> PayloadFieldType:
    """Return the payload field type for ``value`` (default: keyword)."""
    if value is None:
        return PayloadFieldType.KEYWORD
    return PayloadFieldType(value)


@dataclass
class SpaceVectors:
    """The vectors a record contributes to one named vector space.

    Only the fields matching the space's ``vector_type`` are meaningful:
    dense spaces use ``dense``, sparse spaces use
    ``sparse_indices``/``sparse_values`` and multi-vector spaces use
    ``multi_vector`` (the full token-level matrix).
    """

    space: str
    embedding_space: EmbeddingSpaceDescriptor
    dense: list[float] | None = None
    sparse_indices: list[int] | None = None
    sparse_values: list[float] | None = None
    multi_vector: list[list[float]] | None = None

    def is_empty(self) -> bool:
        """Return True when no vector of any kind is present."""
        return not (self.dense or self.sparse_indices or self.multi_vector)


@dataclass
class StoredRecord:
    """A chunk plus the vectors and metadata stored for it.

    The chunk itself carries the public identity (chunk id, document id,
    text, index/count, metadata and provenance); ``vectors`` maps a named
    vector space to the vectors stored for that space.
    """

    chunk: Chunk
    vectors: dict[str, SpaceVectors] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    """Extra metadata stored next to the chunk (filterable payload fields)."""

    @classmethod
    def from_embedding_result(
        cls,
        chunk: Chunk,
        result: EmbeddingResult,
        *,
        space: str = "default",
        vector_type: RepresentationType | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StoredRecord:
        """Build a record from a chunk and an embedding result.

        The embedding ecosystem's result already carries the model identity
        and the dense / sparse / multi-vector representations, so this is the
        bridge between the two ecosystems - no vector is copied twice.

        A named vector space holds exactly one representation kind: the kind
        selected by ``vector_type`` (default: the representation type of the
        result).  The other representations of the result are ignored, they
        belong to another space.
        """
        kind = vector_type or result.representation_type
        descriptor = EmbeddingSpaceDescriptor(
            model_id=result.model_id,
            model_revision=result.metadata.get("model_revision"),
            adapter_name=result.metadata.get("adapter_name"),
            dimensions=None if kind is RepresentationType.SPARSE else result.dimensions,
            representation_type=kind,
            normalization=result.normalization,
            query_instruction=result.metadata.get("query_instruction"),
            document_instruction=result.metadata.get("document_instruction"),
        )
        dense: list[float] | None = None
        sparse_indices: list[int] | None = None
        sparse_values: list[float] | None = None
        multi_vector: list[list[float]] | None = None
        if kind is RepresentationType.DENSE:
            dense = list(result.dense_vector or [])
        elif kind is RepresentationType.SPARSE:
            sparse_indices = list(result.sparse_indices or [])
            sparse_values = list(result.sparse_values or [])
        else:
            multi_vector = [list(vector) for vector in (result.multi_vector or [])]
        vectors = SpaceVectors(
            space=space,
            embedding_space=descriptor,
            dense=dense,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            multi_vector=multi_vector,
        )
        return cls(chunk=chunk, vectors={space: vectors}, metadata=dict(metadata or {}))


@dataclass(frozen=True)
class ChunkProvenance:
    """Where a chunk came from in the source document."""

    source: str | None = None
    """Path / name of the source document."""

    pages: tuple[int, ...] = ()
    """Page numbers the chunk covers (empty for pageless formats)."""

    block_refs: tuple[BlockReference, ...] = ()
    """Blocks the chunk was built from."""

    char_start: int | None = None
    """Offset of the chunk inside the flattened document text."""

    char_end: int | None = None
    """End offset of the chunk inside the flattened document text."""

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> ChunkProvenance:
        """Extract the provenance of ``chunk``."""
        metadata = chunk.metadata or {}
        pages = metadata.get("pages")
        if pages is None:
            pages = sorted(
                {
                    ref.page_number
                    for ref in chunk.block_refs
                    if ref.page_number is not None
                }
            )
        return cls(
            source=metadata.get("source"),
            pages=tuple(int(page) for page in pages),
            block_refs=tuple(chunk.block_refs),
            char_start=metadata.get("char_start"),
            char_end=metadata.get("char_end"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "pages": list(self.pages),
            "block_refs": [ref.as_dict() for ref in self.block_refs],
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


@dataclass
class SearchRequest:
    """A search request against one named vector space.

    Exactly one query representation must be provided, and it must match the
    vector type of the target space (dense / sparse / multi-vector).
    """

    collection: str
    space: str
    limit: int = 10
    dense: list[float] | None = None
    sparse_indices: list[int] | None = None
    sparse_values: list[float] | None = None
    multi_vector: list[list[float]] | None = None
    filter: MetadataFilter | None = None
    score_threshold: float | None = None
    params: dict[str, Any] = field(default_factory=dict)
    """Backend-specific search parameters (e.g. ``{"hnsw_ef": 128}``)."""

    with_payload: bool | Sequence[str] = True
    """Return the stored payload.

    ``True`` returns everything (text, metadata, provenance), a list of field
    names returns only those payload fields, and ``False`` returns the record
    identity only (chunk id and document id).
    """

    with_vectors: bool = False
    """Return the stored vectors as well."""

    offset: int | None = None
    """Offset into the result list."""

    def query_representation(self) -> RepresentationType:
        """Return the kind of query vector carried by this request.

        Raises:
            VectorStoreConfigError: if the request carries zero or more than
                one query representation.
        """
        from vector_stores.contracts.exceptions import VectorStoreConfigError

        present = [
            name
            for name, value in (
                ("dense", self.dense),
                ("sparse", self.sparse_indices if self.sparse_values else None),
                ("multi-vector", self.multi_vector),
            )
            if value is not None
        ]
        if not present:
            raise VectorStoreConfigError(
                "search request carries no query vector; provide one of "
                "dense / sparse_indices+sparse_values / multi_vector"
            )
        if len(present) > 1:
            raise VectorStoreConfigError(
                "search request carries more than one query vector "
                f"({', '.join(present)}); provide exactly one"
            )
        return RepresentationType(present[0])


@dataclass
class SearchResult:
    """One hit of a search.

    ``score`` is the raw value the backend returned.  It is only meaningful
    together with ``metric`` and ``score_direction``: a cosine similarity
    (higher is better) is not comparable with a euclidean distance (lower is
    better), and neither is comparable across models.
    """

    chunk_id: str
    doc_id: str
    score: float
    space: str
    metric: DistanceMetric | None = None
    score_direction: ScoreDirection | None = None
    chunk: Chunk | None = None
    """The full chunk, when the payload was requested."""

    text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: ChunkProvenance | None = None


@dataclass
class CollectionInfo:
    """Backend description of an existing collection."""

    name: str
    description: str | None = None
    vector_spaces: dict[str, VectorSpaceConfig] = field(default_factory=dict)
    point_count: int | None = None
    indexes: tuple[IndexDescription, ...] = ()
    backend: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScrollPage:
    """One page of a scroll / iteration over a collection."""

    records: list[StoredRecord] = field(default_factory=list)
    next_offset: int | str | None = None
    """Opaque cursor of the next page (``None`` when the page is the last one).

    The value must be passed back to :meth:`VectorStore.scroll` unchanged;
    its type is backend specific (an index or a point id).
    """


def descriptor_to_dict(descriptor: EmbeddingSpaceDescriptor) -> dict[str, Any]:
    """Serialize an embedding-space descriptor."""
    return {
        "model_id": descriptor.model_id,
        "model_revision": descriptor.model_revision,
        "adapter_name": descriptor.adapter_name,
        "dimensions": descriptor.dimensions,
        "representation_type": descriptor.representation_type.value,
        "normalization": descriptor.normalization.value,
        "query_instruction": descriptor.query_instruction,
        "document_instruction": descriptor.document_instruction,
        "supported_languages": (
            list(descriptor.supported_languages)
            if descriptor.supported_languages is not None
            else None
        ),
    }


def descriptor_from_dict(data: dict[str, Any]) -> EmbeddingSpaceDescriptor:
    """Rebuild an embedding-space descriptor from :func:`descriptor_to_dict`."""
    languages = data.get("supported_languages")
    return EmbeddingSpaceDescriptor(
        model_id=data["model_id"],
        model_revision=data.get("model_revision"),
        adapter_name=data.get("adapter_name"),
        dimensions=data.get("dimensions"),
        representation_type=RepresentationType(
            data.get("representation_type", RepresentationType.DENSE.value)
        ),
        normalization=Normalization(data.get("normalization", Normalization.NONE.value)),
        query_instruction=data.get("query_instruction"),
        document_instruction=data.get("document_instruction"),
        supported_languages=list(languages) if languages is not None else None,
    )


def vector_space_from_dict(name: str, data: dict[str, Any]) -> VectorSpaceConfig:
    """Rebuild a vector space configuration from :meth:`VectorSpaceConfig.as_dict`."""
    hnsw_data = data.get("hnsw")
    sparse_data = data.get("sparse")
    return VectorSpaceConfig(
        name=name,
        vector_type=RepresentationType(data["vector_type"]),
        embedding_space=descriptor_from_dict(data["embedding_space"]),
        dimensions=data.get("dimensions"),
        distance=DistanceMetric(data["distance"]) if data.get("distance") else None,
        hnsw=HnswIndexConfig(**hnsw_data) if hnsw_data else None,
        sparse=SparseIndexConfig(**sparse_data) if sparse_data else None,
        metadata=dict(data.get("metadata") or {}),
    )
