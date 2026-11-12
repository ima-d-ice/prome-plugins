"""Shared chunker protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from chunkers.contracts.models import Chunk
from parsers.contracts.models import ParsedDocument


@runtime_checkable
class Chunker(Protocol):
    """Minimal protocol every Prome chunker implements.

    Chunkers accept a ParsedDocument and return the full list of Chunk
    objects (deterministic, in document order). They must not mutate the
    document.
    """

    name: str

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """Chunk ``document`` into Chunks, preserving provenance."""
        ...
