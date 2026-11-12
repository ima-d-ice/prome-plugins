"""Common exceptions for Prome RAG chunkers."""


class ChunkerError(Exception):
    """Base class for all chunker-related errors."""


class ChunkerConfigError(ChunkerError, ValueError):
    """Invalid chunker configuration (bad size, overlap, tokenizer, ...)."""
