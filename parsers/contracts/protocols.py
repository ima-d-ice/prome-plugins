"""Common parser protocol shared by all Prome RAG parsers."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from parsers.contracts.models import ParsedDocument


@runtime_checkable
class Parser(Protocol):
    """Minimal protocol every Prome parser implements."""

    name: str

    def parse(self, source: str | Path) -> ParsedDocument:
        """Parse the given document and return a ParsedDocument.

        Raises a subclass of ParserError on failure.
        """
        ...
