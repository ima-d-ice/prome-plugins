"""Prome RAG Embedding Ecosystem.

Modular, plug-and-play embedding module supporting local embedding models.

See individual adapter READMEs for detailed documentation.
"""

from __future__ import annotations

from embeddings.contracts.models import EmbeddingInput, EmbeddingResult
from embeddings.contracts.protocols import EmbeddingModel
from embeddings.contracts.models import EmbeddingCapabilities, EmbeddingSpaceDescriptor
from embeddings.qwen3 import Qwen3EmbeddingModel

#: Core data models and protocol
__all__ = [
    "EmbeddingInput",
    "EmbeddingResult",
    "EmbeddingCapabilities",
    "EmbeddingSpaceDescriptor",
    "EmbeddingModel",
]

#: Adapter implementations
__all__ += [
    "Qwen3EmbeddingModel",
]