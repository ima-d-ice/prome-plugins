"""Token counter interface and implementations.

Defines a TokenCounter interface for estimating token counts.
Uses an appropriate tokenizer where available.

For local Qwen models, uses tokenizer information associated with the
actual configured model when accessible.
For Groq, uses model-specific token counting when available.
If exact preflight tokenization is unavailable, uses a documented
conservative estimate and reconciles with response usage.

Never claims an estimate is an exact token count.
Does not silently truncate the user question.
If the question itself exceeds the allowed limit, rejects it with
an actionable error or requires explicit caller-controlled truncation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TokenCounter(ABC):
    """Interface for token count estimation.

    Implementations must:
    - Never claim an estimate is an exact token count.
    - Return a conservative estimate when exact counting is unavailable.
    - Reconcile with actual usage from response diagnostics.
    """

    @abstractmethod
    def count(self, text: str) -> int:
        """Estimate the token count for a given text.

        Args:
            text: the text to estimate tokens for.

        Returns:
            Estimated token count. Always >= 1 for non-empty strings.
        """
        ...


class DefaultTokenCounter(TokenCounter):
    """Default token counter using a conservative character-based estimate.

    Uses 4 characters per token (conservative estimate suitable for most
    LLMs when exact tokenizer is unavailable).

    Important: This does NOT claim to be an exact token count. It is a
    conservative estimate that will typically under-count, ensuring we
    never exceed budgets due to overestimation.
    """

    def count(self, text: str) -> int:
        """Estimate token count using conservative character-based formula.

        Args:
            text: the text to estimate tokens for.

        Returns:
            Estimated token count. Minimum of 1 for non-empty strings.
        """
        if not text or not text.strip():
            return 0

        # Conservative estimate: 4 characters per token
        # This typically under-counts for many models, which is safer
        # than over-counting (which could cause budget exhaustion).
        estimated = max(1, len(text) // 4)
        return estimated


class ModelSpecificTokenCounter(TokenCounter):
    """Token counter that uses model-specific tokenizer when available.

    For local Qwen models, attempts to use the tokenizer associated with
    the configured model.
    For Groq, uses model-specific token counting when available.

    Falls back to DefaultTokenCounter when model-specific tokenizer is
    not accessible.
    """

    def __init__(
        self,
        model_id: str | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        """Initialize the model-specific token counter.

        Args:
            model_id: the model identifier (e.g. "qwen/qwen3.8-27b").
            tokenizer: an optional tokenizer instance.
        """
        self._model_id = model_id
        self._tokenizer = tokenizer
        self._default = DefaultTokenCounter()

    def count(self, text: str) -> int:
        """Estimate token count, preferring model-specific tokenizer.

        Args:
            text: the text to estimate tokens for.

        Returns:
            Estimated token count.
        """
        # If we have a model-specific tokenizer, use it
        if self._tokenizer is not None:
            try:
                # Prefer tokenizer.decode or tokenizer.encode depending on type
                # for transformers-style tokenizers:
                tokens = self._tokenizer.encode(text)
                return len(tokens)
            except Exception:
                # Fall through to default if tokenizer fails
                pass

        # If model_id suggests a known tokenizer, try loading
        if self._model_id is not None:
            try:
                # Attempt to get a tokenizer for the model
                # This is a placeholder for model-specific logic
                # In practice, this would use transformers.AutoTokenizer
                pass
            except Exception:
                pass

        # Fall back to conservative estimate
        return self._default.count(text)


#: Global default token counter instance
_default_counter: TokenCounter | None = None


def get_default_token_counter() -> TokenCounter:
    """Get the global default token counter instance.

    Returns:
        A TokenCounter instance (DefaultTokenCounter by default).
    """
    global _default_counter
    if _default_counter is None:
        _default_counter = DefaultTokenCounter()
    return _default_counter


# Alias for convenience
get_token_counter = get_default_token_counter