"""Index abstractions shared by vector-store backends.

Vector index algorithms and payload (metadata) indexes are usually managed by
the storage backend itself.  The contracts therefore describe *what* an index
is and *which parameters* it takes, while each adapter exposes only the index
kinds it really implements.

A shared name is not a promise of portability: two backends may both call
something "hnsw" while accepting different build parameters.  The portable
parameter names are listed per backend in
:class:`~vector_stores.contracts.capabilities.StoreCapabilities`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class IndexKind(str, Enum):
    """Kinds of index a backend may support."""

    EXACT = "exact"
    """Brute-force scan over every vector (exact results, no approximation)."""

    HNSW = "hnsw"
    """Approximate dense index based on a navigable small-world graph."""

    IVF = "ivf"
    """Approximate dense index based on inverted file partitions.

    Added for the backends that support it natively (FAISS, pgvector); the
    backends that do not simply do not report it.
    """

    SPARSE_INVERTED = "sparse-inverted"
    """Inverted index over sparse (lexical) vector dimensions."""

    PAYLOAD = "payload"
    """Index over a metadata / payload field used for filtering."""

    MULTI_VECTOR = "multi-vector"
    """Late-interaction index over a token-level vector matrix."""


class PayloadFieldType(str, Enum):
    """Portable types for metadata fields that can be indexed."""

    KEYWORD = "keyword"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"


@dataclass(frozen=True)
class PayloadIndexSpec:
    """A metadata field the user explicitly asked to index."""

    field: str
    """Payload field path (may be dotted, e.g. ``metadata.tenant_id``)."""

    field_type: PayloadFieldType = PayloadFieldType.KEYWORD
    """Declared type of the field."""

    def as_dict(self) -> dict[str, Any]:
        return {"field": self.field, "field_type": self.field_type.value}


@dataclass(frozen=True)
class HnswIndexConfig:
    """Build and query parameters of an HNSW index.

    Defaults are deliberately conservative and match the values Qdrant uses
    by default; low-level knobs are not exposed here on purpose.
    """

    m: int = 16
    """Number of neighbours per graph node (higher = better recall, more memory)."""

    ef_construct: int = 100
    """Size of the candidate list while building the graph."""

    full_scan_threshold: int = 10_000
    """Below this number of vectors the backend may prefer an exact scan."""

    max_indexing_threads: int | None = None
    """Threads used for index construction (``None`` = backend default)."""

    query_ef: int | None = None
    """Beam width used at query time (``None`` = backend default)."""

    backend_options: Mapping[str, Any] = field(default_factory=dict)
    """Backend-specific build options, isolated from the portable contract."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "m": self.m,
            "ef_construct": self.ef_construct,
            "full_scan_threshold": self.full_scan_threshold,
            "max_indexing_threads": self.max_indexing_threads,
            "query_ef": self.query_ef,
            "backend_options": dict(self.backend_options),
        }


@dataclass(frozen=True)
class SparseIndexConfig:
    """Parameters of the inverted index over sparse vectors."""

    full_scan_threshold: int = 10_000
    """Below this number of vectors the backend may scan instead of using the index."""

    backend_options: Mapping[str, Any] = field(default_factory=dict)
    """Backend-specific options, isolated from the portable contract."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "full_scan_threshold": self.full_scan_threshold,
            "backend_options": dict(self.backend_options),
        }


@dataclass(frozen=True)
class IndexConfig:
    """Portable description of an index the user wants to exist.

    ``kind`` says *what* the index is, ``target`` says *what it applies to*
    (a named vector space for vector indexes, a payload field path for
    metadata indexes).  Build/query parameters are kept as mappings so a
    backend can reject the ones it does not understand instead of ignoring
    them silently.
    """

    kind: IndexKind
    target: str
    name: str | None = None
    """Optional identifier; defaults to ``target``."""

    field_type: PayloadFieldType | None = None
    """Declared payload type (metadata indexes only)."""

    build_params: Mapping[str, Any] = field(default_factory=dict)
    """Parameters applied when the index is (re)built."""

    query_params: Mapping[str, Any] = field(default_factory=dict)
    """Parameters applied when the index is queried."""

    backend_options: Mapping[str, Any] = field(default_factory=dict)
    """Backend-specific options, isolated from the portable contract."""

    @property
    def index_name(self) -> str:
        """Return the identifier of the index (``name`` or ``target``)."""
        return self.name or self.target

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "target": self.target,
            "name": self.name,
            "field_type": self.field_type.value if self.field_type else None,
            "build_params": dict(self.build_params),
            "query_params": dict(self.query_params),
            "backend_options": dict(self.backend_options),
        }


@dataclass(frozen=True)
class IndexDescription:
    """What a backend reports about an existing index."""

    kind: IndexKind
    target: str
    name: str | None = None
    field_type: PayloadFieldType | None = None
    build_params: Mapping[str, Any] = field(default_factory=dict)
    query_params: Mapping[str, Any] = field(default_factory=dict)
    backend_options: Mapping[str, Any] = field(default_factory=dict)

    @property
    def index_name(self) -> str:
        """Return the identifier of the index (``name`` or ``target``)."""
        return self.name or self.target

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "target": self.target,
            "name": self.name,
            "field_type": self.field_type.value if self.field_type else None,
            "build_params": dict(self.build_params),
            "query_params": dict(self.query_params),
            "backend_options": dict(self.backend_options),
        }


@dataclass(frozen=True)
class IndexKindCapability:
    """Capability description for a single index kind.

    ``supported`` tells whether the backend implements the kind at all;
    ``build_parameters`` / ``query_parameters`` list the *portable* parameter
    names the backend honours for that kind.  A parameter the backend does not
    list must not be passed to it.
    """

    kind: IndexKind
    supported: bool = False
    build_parameters: tuple[str, ...] = ()
    query_parameters: tuple[str, ...] = ()
    notes: str | None = None
    """Human readable caveat (e.g. "no effect in local mode")."""
