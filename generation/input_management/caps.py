"""Input and output caps for generation requests.

Configurable limits on request size, query tokens, conversation history,
retrieval context, model input, and completion tokens.

All limits are independently configurable and enforceable together.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Caps:
    """Configuration for input/output caps.

    Attributes:
        max_request_body_size: maximum bytes in the request body.
        max_user_query_tokens: maximum tokens for a user query.
        max_conversation_history_tokens: maximum tokens for conversation history.
        max_retrieval_context_tokens: maximum tokens for retrieved context.
        max_model_input_tokens: maximum tokens for model input (context window).
        max_completion_tokens: maximum tokens for model output/completion.
        max_context_window: maximum context window in tokens (across all sources).
        safety_classifier_input_limit: input tokens limit for safety classifiers.
    """

    max_request_body_size: int = 1_000_000
    """Maximum request body size in bytes (default: 1MB)."""

    max_user_query_tokens: int = 8192
    """Maximum tokens for a user query (default: 8192)."""

    max_conversation_history_tokens: int = 4096
    """Maximum tokens for conversation history (default: 4096)."""

    max_retrieval_context_tokens: int = 8192
    """Maximum tokens for retrieved untrusted context (default: 8192)."""

    max_model_input_tokens: int | None = None
    """Maximum tokens for model input. When None, uses model's context window.

    This is enforced alongside provider TPM admission budget.
    """

    max_completion_tokens: int | None = None
    """Maximum completion tokens. When None, uses model's output limits.

    If set, this caps the output tokens regardless of model capabilities.
    """

    max_context_window: int | None = None
    """Maximum context window in tokens across all sources.

    When None, uses the model's context window. Enforced together with
    max_model_input_tokens and provider budgets.
    """

    safety_classifier_input_limit: int = 512
    """Input tokens limit for safety classifiers (default: 512, matches
    Prompt Guard 2 models' context window).
    """


#: Default caps configuration following the specification defaults.
DEFAULT_CAPS = Caps()