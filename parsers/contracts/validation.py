"""Shared source validation helpers (dependency-free)."""

from __future__ import annotations

from pathlib import Path

from parsers.contracts.exceptions import CorruptDocumentError, EmptyDocumentError


def validate_source(source: str | Path) -> Path:
    """Validate that a source path exists and is a non-empty file."""
    path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    if not path.is_file():
        raise CorruptDocumentError(f"Source is not a file: {path}")
    if path.stat().st_size == 0:
        raise EmptyDocumentError(f"Document is empty (0 bytes): {path}")
    return path
