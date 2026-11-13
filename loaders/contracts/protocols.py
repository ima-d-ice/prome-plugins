"""Loader protocol for Prome RAG Plugins."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from loaders.contracts.models import (
    LoaderResult,
    SourceType,
)


@runtime_checkable
class Loader(Protocol):
    """Minimal protocol every loader implements.

    A loader fetches raw content from a source and returns a LoadedResource
    (or a LoaderResult containing multiple resources). It does NOT perform
    parsing, chunking, embedding or vector indexing.
    """

    name: str

    def load(self, source: str | Path) -> LoaderResult:
        """Load content from the given source.

        Args:
            source: A file path, URL, or source identifier.

        Returns:
            A LoaderResult containing successfully loaded resources,
            failures, skips, and warnings.
        """
        ...

    def supports(self, source_type: SourceType) -> bool:
        """Check if this loader supports the given source type."""
        ...