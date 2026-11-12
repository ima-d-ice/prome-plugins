"""Configuration of the hierarchical (coarse-to-fine) retrieval strategy.

The configuration stays deliberately small: it exposes *what* to retrieve and
*how much* context to attach, and nothing about the backend.  Backend
differences are handled by the vector-store contracts, not by this class.
"""

from __future__ import annotations

from dataclasses import dataclass

from retrieval.contracts.exceptions import RetrievalConfigError
from vector_stores.contracts.filters import MetadataFilter


@dataclass(frozen=True)
class HierarchicalConfig:
    """Settings of :class:`~retrieval.strategies.hierarchical.strategy.HierarchicalRetrievalStrategy`.

    Args:
        parent_top_k: how many parent sections the coarse stage may select.
        child_top_k: how many child chunks the fine stage may retrieve from
            the selected parents.
        final_top_k: how many results the strategy returns.
        parent_score_threshold: minimum score a parent must reach to be
            selected (``None`` disables the threshold).
        child_score_threshold: minimum score a child must reach to be
            returned (``None`` disables the threshold).
        filters: additional metadata filter applied to **both** stages, on
            top of the level filter.
        include_parent_context: whether the parent section text is attached
            to every result.
        expand_adjacent: whether neighbouring children of the same parent are
            attached as additional context.
        max_adjacent: how many neighbouring children are attached on each
            side of the matched child.
        vector_space: name of the named dense vector space to search.
    """

    parent_top_k: int = 8
    child_top_k: int = 40
    final_top_k: int = 10
    parent_score_threshold: float | None = None
    child_score_threshold: float | None = None
    filters: MetadataFilter | None = None
    include_parent_context: bool = True
    expand_adjacent: bool = False
    max_adjacent: int = 1
    vector_space: str = "dense"

    def __post_init__(self) -> None:
        for name in ("parent_top_k", "child_top_k", "final_top_k"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise RetrievalConfigError(f"{name} must be an integer >= 1, got {value!r}")
        if not isinstance(self.max_adjacent, int) or isinstance(self.max_adjacent, bool):
            raise RetrievalConfigError(
                f"max_adjacent must be an integer, got {self.max_adjacent!r}"
            )
        if self.max_adjacent < 0:
            raise RetrievalConfigError(
                f"max_adjacent must be >= 0, got {self.max_adjacent}"
            )
        for name in ("parent_score_threshold", "child_score_threshold"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, (int, float)):
                raise RetrievalConfigError(
                    f"{name} must be a number or None, got {value!r}"
                )
        if not isinstance(self.vector_space, str) or not self.vector_space.strip():
            raise RetrievalConfigError("vector_space must be a non-empty string")


@dataclass(frozen=True)
class HierarchicalIndexConfig:
    """Settings of :class:`~retrieval.strategies.hierarchical.indexer.HierarchicalIndexer`.

    Args:
        vector_space: name of the named dense vector space to write to.
        batch_size: how many chunks are embedded per inference call.
    """

    vector_space: str = "dense"
    batch_size: int = 32
    collection_description: str = "Hierarchical (parent/child) chunks"

    def __post_init__(self) -> None:
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool):
            raise RetrievalConfigError(
                f"batch_size must be an integer, got {self.batch_size!r}"
            )
        if self.batch_size < 1:
            raise RetrievalConfigError(f"batch_size must be >= 1, got {self.batch_size}")
        if not isinstance(self.vector_space, str) or not self.vector_space.strip():
            raise RetrievalConfigError("vector_space must be a non-empty string")
