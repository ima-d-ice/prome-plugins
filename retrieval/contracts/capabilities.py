"""Capability reporting for the retrieval ecosystem.

A caller that wants to know whether a retriever can do something -- filter
metadata, fuse two channels, merge adjacent chunks -- asks it, instead of
discovering the answer from an exception or from a silently degraded result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vector_stores.contracts.filters import FIELD_OPS, FilterOp


@dataclass(frozen=True)
class RetrievalCapabilities:
    """What a retriever can do.

    Attributes:
        dense: whether the retriever performs dense vector retrieval.
        lexical: whether the retriever performs lexical retrieval.
        fusion: whether the retriever fuses several channels.
        metadata_filtering: whether the retriever applies metadata filters.
        filter_ops: the filter operations the retriever can express.  An empty
            set means "no filtering"; the logical operators (``and``, ``or``,
            ``not``) are structural and are not listed.
        postprocessing: the post-processing plugins the retriever applies.
        details: backend specific notes.
    """

    dense: bool = False
    lexical: bool = False
    fusion: bool = False
    metadata_filtering: bool = False
    filter_ops: frozenset[FilterOp] = frozenset(FIELD_OPS)
    postprocessing: frozenset[str] = frozenset()
    details: Mapping[str, Any] = field(default_factory=dict)

    def supports(self, operation: FilterOp) -> bool:
        """Return whether the filter operation is supported."""
        return operation in self.filter_ops

    @property
    def supports_filtering(self) -> bool:
        """Return whether any metadata filtering is supported at all."""
        return self.metadata_filtering and bool(self.filter_ops)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly description of the capabilities."""
        return {
            "dense": self.dense,
            "lexical": self.lexical,
            "fusion": self.fusion,
            "metadata_filtering": self.metadata_filtering,
            "filter_ops": sorted(op.value for op in self.filter_ops),
            "postprocessing": sorted(self.postprocessing),
            "details": dict(self.details),
        }
