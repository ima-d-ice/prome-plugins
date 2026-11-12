"""Semantic chunker: meaning-aware boundaries driven by an embedding model."""

from chunkers.semantic.chunker import (
    SemanticChunker,
    SemanticChunkerError,
    SemanticConfig,
)

__all__ = ["SemanticChunker", "SemanticChunkerError", "SemanticConfig"]
