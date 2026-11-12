"""Shared contracts for Prome RAG chunkers."""

from chunkers.contracts.exceptions import ChunkerConfigError, ChunkerError
from chunkers.contracts.flatten import (
    FlatBlock,
    build_chunks,
    chunk_digest,
    flatten_document,
    refs_for_range,
    trim_span,
)
from chunkers.contracts.models import BlockReference, Chunk
from chunkers.contracts.protocols import Chunker
from chunkers.contracts.tokenizers import (
    ApproxTokenCounter,
    CallableCounter,
    CharacterCounter,
    TokenCounter,
    ensure_counter,
)
from chunkers.contracts.validation import validate_size_overlap

__all__ = [
    "ApproxTokenCounter",
    "BlockReference",
    "CallableCounter",
    "CharacterCounter",
    "Chunk",
    "Chunker",
    "ChunkerConfigError",
    "ChunkerError",
    "FlatBlock",
    "TokenCounter",
    "build_chunks",
    "chunk_digest",
    "ensure_counter",
    "flatten_document",
    "refs_for_range",
    "trim_span",
    "validate_size_overlap",
]
