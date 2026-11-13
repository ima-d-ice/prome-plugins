"""Common exceptions for the Prome RAG Plugins generation ecosystem.

The hierarchy mirrors the specification: a single base error
(:class:`GenerationError`) plus narrow subclasses so callers can react to a
specific failure without parsing messages.

Do not make all exceptions eligible for fallback. Only config errors,
dependency errors, and inference errors should trigger retry/fallback logic.
"""

from __future__ import annotations


class GenerationError(Exception):
    """Base class for all generation errors."""


class GenerationConfigError(GenerationError, ValueError):
    """Invalid generation configuration (bad limits, bad model, ...)."""


class GenerationDependencyError(GenerationError, ImportError):
    """An optional dependency needed by the generator is not installed.

    The message always names the extra to install.
    """


class GenerationInferenceError(GenerationError):
    """The model failed while producing generated text.

    Raised instead of returning partial or empty results so that callers
    never silently continue with degraded input.
    """


class GenerationConfigKeyError(GenerationError, KeyError):
    """A required configuration key is missing."""


class GenerationModelError(GenerationError, ValueError):
    """The requested model is unavailable or unsupported."""


class GenerationModelUnavailableError(GenerationModelError):
    """The selected model is currently unavailable (e.g. Ollama not running)."""


class GenerationRateLimitError(GenerationError):
    """Rate limit was hit for the current key/model combination."""


class GenerationKeyPoolExhaustedError(GenerationRateLimitError):
    """All keys in the pool are exhausted for the target model."""


class GenerationContextLimitError(GenerationError):
    """Input context exceeds the model's or application's limits."""


class GenerationTimeoutError(GenerationError):
    """The generation operation timed out."""


class GenerationSafetyBlockError(GenerationError):
    """The generation was blocked by a safety evaluator."""


class GenerationPromptGuardError(GenerationError):
    """Prompt Guard classification failed or returned an error."""


class GenerationSafeguardError(GenerationError):
    """Safety evaluator (GPT-OSS Safeguard) failed or returned an error."""


class GenerationOllamaConnectionError(GenerationError):
    """Connection error communicating with the Ollama backend."""


class GenerationAuthenticationError(GenerationError):
    """Authentication failure with the provider."""


class GenerationInvalidRequestError(GenerationConfigError):
    """The request parameters are invalid for the selected model."""


class GenerationInvalidModelError(GenerationConfigError):
    """The model ID is unsupported or not configured."""


class GenerationProviderUnavailableError(GenerationError):
    """The selected provider is currently unavailable."""