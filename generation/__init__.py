"""Prome RAG Plugins — Generation, Provider Routing & Safety Ecosystem.

The Generation and Model Routing ecosystem for Prome RAG Plugins.
Supports local Ollama Qwen models, Groq generation with 12-key rotation,
adaptive Prompt Guard integration, optional GPT-OSS-Safeguard policy evaluation,
optional query enhancement, and model-aware rate limiting.

See generation/ for the full package structure.
"""

from __future__ import annotations

from generation.contracts.exceptions import (
    GenerationAuthenticationError,
    GenerationConfigError,
    GenerationContextLimitError,
    GenerationDependencyError,
    GenerationError,
    GenerationInferenceError,
    GenerationInvalidModelError,
    GenerationInvalidRequestError,
    GenerationKeyPoolExhaustedError,
    GenerationOllamaConnectionError,
    GenerationPromptGuardError,
    GenerationProviderUnavailableError,
    GenerationRateLimitError,
    GenerationSafeguardError,
    GenerationSafetyBlockError,
)
from generation.contracts.models import (
    Capabilities,
    GenerationRequest,
    GenerationResponse,
    GenerationTaskType,
    ModelConfiguration,
    ProviderProtocol,
)
from generation.contracts.protocols import GenerationProvider
from generation.input_management import (
    Caps,
    ContextBudget,
    TokenCounter,
    get_token_counter,
)
from generation.providers import GroqProvider, OllamaProvider
from generation.query_enhancement import QueryEnhancer
from generation.rate_limiting import (
    KeyRotationPool,
    RateLimiter,
    ResponseHeaderParser,
    TokenBudget,
)
from generation.safety import PromptGuardClassifier, SafeguardEvaluator

__all__ = [
    "Capabilities",
    "Caps",
    "ContextBudget",
    "GenerationAuthenticationError",
    "GenerationConfigError",
    "GenerationContextLimitError",
    "GenerationDependencyError",
    "GenerationError",
    "GenerationInferenceError",
    "GenerationInvalidModelError",
    "GenerationInvalidRequestError",
    "GenerationKeyPoolExhaustedError",
    "GenerationOllamaConnectionError",
    "GenerationPromptGuardError",
    "GenerationProvider",
    "GenerationProviderUnavailableError",
    "GenerationRateLimitError",
    "GenerationRequest",
    "GenerationResponse",
    "GenerationSafeguardError",
    "GenerationSafetyBlockError",
    "GenerationTaskType",
    "GroqProvider",
    "KeyRotationPool",
    "ModelConfiguration",
    "OllamaProvider",
    "PromptGuardClassifier",
    "ProviderProtocol",
    "QueryEnhancer",
    "RateLimiter",
    "ResponseHeaderParser",
    "SafeguardEvaluator",
    "TokenBudget",
    "TokenCounter",
    "get_token_counter",
]