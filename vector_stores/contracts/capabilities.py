"""Capability descriptors for vector-store backends.

Capabilities are discovered, not assumed: a user asks the adapter what it can
do (``store.capabilities``) instead of guessing from the backend name.  The
descriptor also states *which* portable filter operations and *which* index
kinds the backend honours, so an adapter never silently degrades a query.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vector_stores.contracts.filters import FilterOp
from vector_stores.contracts.indexes import (
    IndexKind,
    IndexKindCapability,
    PayloadFieldType,
)


@dataclass(frozen=True)
class StoreCapabilities:
    """What a vector-store backend can do."""

    backend: str
    """Backend identifier (e.g. ``"qdrant"``)."""

    dense_search: bool = True
    sparse_search: bool = False
    multi_vector_search: bool = False
    exact_search: bool = False
    """Brute-force search with exact results."""

    filter_ops: frozenset[FilterOp] = frozenset()
    """Metadata filter operations the backend can express."""

    payload_field_types: frozenset[PayloadFieldType] = frozenset()
    """Metadata field types the backend can index."""

    index_kinds: dict[IndexKind, IndexKindCapability] = field(default_factory=dict)
    """Supported index kinds and the portable parameters they accept."""

    persistence_modes: tuple[str, ...] = ()
    """How the backend can persist data (e.g. ``("memory", "local", "server")``)."""

    delete_by_filter: bool = True
    delete_by_document: bool = True
    filter_requires_declared_fields: bool = False
    """Only the fields declared in the collection config can be filtered on.

    Backends with a fixed schema (LanceDB) cannot push a filter down to a
    field that does not exist in it; they reject such a filter instead of
    ignoring it.
    """

    create_index: bool = False
    """The backend can create indexes after the collection exists."""

    delete_index: bool = False
    max_batch_size: int | None = None
    """Maximum number of records per upsert call (``None`` = unlimited)."""

    notes: tuple[str, ...] = ()
    """Backend specific caveats (e.g. local-mode limitations)."""

    def supports(self, kind: IndexKind) -> bool:
        """Return True when the backend supports the given index kind."""
        capability = self.index_kinds.get(kind)
        return bool(capability and capability.supported)

    def index_capability(self, kind: IndexKind) -> IndexKindCapability | None:
        """Return the capability description of an index kind."""
        return self.index_kinds.get(kind)

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "dense_search": self.dense_search,
            "sparse_search": self.sparse_search,
            "multi_vector_search": self.multi_vector_search,
            "exact_search": self.exact_search,
            "filter_ops": sorted(op.value for op in self.filter_ops),
            "payload_field_types": sorted(t.value for t in self.payload_field_types),
            "index_kinds": {
                kind.value: {
                    "supported": capability.supported,
                    "build_parameters": list(capability.build_parameters),
                    "query_parameters": list(capability.query_parameters),
                    "notes": capability.notes,
                }
                for kind, capability in self.index_kinds.items()
            },
            "persistence_modes": list(self.persistence_modes),
            "delete_by_filter": self.delete_by_filter,
            "delete_by_document": self.delete_by_document,
            "filter_requires_declared_fields": self.filter_requires_declared_fields,
            "create_index": self.create_index,
            "delete_index": self.delete_index,
            "max_batch_size": self.max_batch_size,
            "notes": list(self.notes),
        }
