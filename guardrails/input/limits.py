"""Request size and limits for the Guardrails ecosystem.

Request body size and message count limits. Uses the Generation subsystem's
existing input caps when available. Guardrails should not implement another
competing token-budget or request-rate-limiter system.

All limits are deterministic and configurable. Return actionable rejection
or review reasons.
"""

from __future__ import annotations

#: Default maximum request body size in bytes (approx 100KB)
DEFAULT_MAX_REQUEST_BYTES = 100_000

#: Default maximum number of context items
DEFAULT_MAX_CONTEXT_ITEMS = 100

#: Default maximum number of messages in conversation history
DEFAULT_MAX_MESSAGES = 50

#: Default maximum input tokens for safety classifiers (matches Generation's
#: safety_classifier_input_limit)
DEFAULT_SAFETY_CLASSIFIER_INPUT_LIMIT = 512


class RequestLimits:
    """Configurable request size and count limits for guardrail evaluation.

    Uses the Generation subsystem's existing input caps when available.
    Guardrails should not implement another competing token-budget or
    request-rate-limiter system.

    Attributes:
        max_request_bytes: maximum request body size in bytes.
        max_context_items: maximum number of retrieved context items.
        max_messages: maximum number of conversation messages.
        safety_classifier_input_limit: maximum input tokens for safety
            classifiers (Prompt Guard, Safeguard).
    """

    def __init__(
        self,
        *,
        max_request_bytes: int | None = None,
        max_context_items: int | None = None,
        max_messages: int | None = None,
        safety_classifier_input_limit: int | None = None,
    ) -> None:
        """Initialize request limits.

        Args:
            max_request_bytes: maximum request body size in bytes.
                When None, uses DEFAULT_MAX_REQUEST_BYTES.
            max_context_items: maximum number of retrieved context items.
                When None, uses DEFAULT_MAX_CONTEXT_ITEMS.
            max_messages: maximum number of conversation messages.
                When None, uses DEFAULT_MAX_MESSAGES.
            safety_classifier_input_limit: maximum input tokens for safety
                classifiers. When None, uses
                DEFAULT_SAFETY_CLASSIFIER_INPUT_LIMIT.
        """

        self.max_request_bytes = max_request_bytes or DEFAULT_MAX_REQUEST_BYTES
        self.max_context_items = max_context_items or DEFAULT_MAX_CONTEXT_ITEMS
        self.max_messages = max_messages or DEFAULT_MAX_MESSAGES
        self.safety_classifier_input_limit = safety_classifier_input_limit or DEFAULT_SAFETY_CLASSIFIER_INPUT_LIMIT

    def check_request_size(
        self,
        request_bytes: int,
    ) -> tuple[bool, list[str]]:
        """Check if the request body size is within limits.

        Args:
            request_bytes: the estimated size of the request in bytes.

        Returns:
            tuple of (within_limit, reason_codes).
        """

        reasons: list[str] = []

        if request_bytes > self.max_request_bytes:
            reasons.append("excessive_request_size")
            return False, reasons

        return True, reasons

    def check_context_item_count(
        self,
        context_item_count: int,
    ) -> tuple[bool, list[str]]:
        """Check if the number of context items is within limits.

        Args:
            context_item_count: the number of context items to check.

        Returns:
            tuple of (within_limit, reason_codes).
        """

        reasons: list[str] = []

        if context_item_count > self.max_context_items:
            reasons.append("excessive_context_items")
            return False, reasons

        return True, reasons

    def check_message_count(
        self,
        message_count: int,
    ) -> tuple[bool, list[str]]:
        """Check if the number of conversation messages is within limits.

        Args:
            message_count: the number of messages to check.

        Returns:
            tuple of (within_limit, reason_codes).
        """

        reasons: list[str] = []

        if message_count > self.max_messages:
            reasons.append("excessive_message_count")
            return False, reasons

        return True, reasons

    def check_safety_classifier_input(
        self,
        input_text: str,
    ) -> tuple[bool, int, list[str]]:
        """Check if input text is within the safety classifier token limit.

        Uses the rough 4-characters-per-token heuristic consistent with the
        Generation subsystem's safety_classifier_input_limit.

        Returns:
            tuple of (within_limit, estimated_tokens, reason_codes).
        """

        estimated_tokens = len(input_text) // 4
        reasons: list[str] = []

        if estimated_tokens > self.safety_classifier_input_limit:
            reasons.append("exceeds_safety_classifier_limit")
            return False, estimated_tokens, reasons

        return True, estimated_tokens, reasons
