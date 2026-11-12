"""Shared configuration validation helpers."""

from __future__ import annotations

from chunkers.contracts.exceptions import ChunkerConfigError


def validate_size_overlap(chunk_size: int, overlap: int, *, prefix: str = "") -> None:
    """Validate a ``chunk_size``/``overlap`` pair, raising ChunkerConfigError."""
    label = f"{prefix} " if prefix else ""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise ChunkerConfigError(f"{label}chunk_size must be an integer, got {chunk_size!r}")
    if isinstance(overlap, bool) or not isinstance(overlap, int):
        raise ChunkerConfigError(f"{label}overlap must be an integer, got {overlap!r}")
    if chunk_size < 1:
        raise ChunkerConfigError(f"{label}chunk_size must be >= 1, got {chunk_size}")
    if overlap < 0:
        raise ChunkerConfigError(f"{label}overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ChunkerConfigError(
            f"{label}overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
        )
