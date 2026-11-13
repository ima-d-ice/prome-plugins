"""Shared contracts for the Prome RAG Plugins generation ecosystem.

The contracts are independent of Ollama, Groq, PyTorch, Transformers, and HTTP SDKs.
They describe the data models and protocol that a generation provider implementation
must satisfy.
"""

from __future__ import annotations

from generation.contracts.exceptions import (
    GenerationConfigError,
    GenerationDependencyError,
    GenerationError,
    GenerationInferenceError,
)
from generation.contracts.models import (
    Capabilities,
    GenerationRequest,
    GenerationResponse,
    GenerationTaskType,
    ModelConfiguration,
    ProviderProtocol,
)

__all__ = [
    "Capabilities",
    "GenerationConfigError",
    "GenerationDependencyError",
    "GenerationError",
    "GenerationInferenceError",
    "GenerationRequest",
    "GenerationResponse",
    "GenerationTaskType",
    "ModelConfiguration",
    "ProviderProtocol",
]