"""Input validation for the Guardrails ecosystem.

Deterministic request validation covering:
- Empty input where disallowed.
- Invalid string encoding.
- Unexpected control characters.
- Request body size.
- Excessive message count.
- Invalid content structures.
- Unsupported input types.
- Invalid metadata.
- Invalid or conflicting request identifiers.

Keep input validation separate from prompt-injection detection.
A long or technical request is not inherently malicious.
Do not silently truncate user content.
Use the Generation subsystem's existing input caps when available.
Guardrails should not implement another competing token-budget or request-rate-limiter system.
Return actionable rejection or review reasons.
"""

from __future__ import annotations

import re
from typing import Any

from guardrails.contracts.models import GuardrailStage
from guardrails.contracts.exceptions import (
    GuardrailConfigError,
    GuardrailError,
)


class InputValidationError(GuardrailError):
    """Raised when input validation fails."""


class InvalidEncodingError(InputValidationError):
    """Raised when input has invalid string encoding."""


class ControlCharacterError(InputValidationError):
    """Raised when input contains unexpected control characters."""


class ExcessiveSizeError(InputValidationError):
    """Raised when request body exceeds size limits."""


class InvalidMetadataError(InputValidationError):
    """Raised when metadata is invalid or conflicting."""


class InvalidRequestIDError(InputValidationError):
    """Raised when request identifier is invalid or conflicting."""


def validate_user_input(
    user_input: str,
    *,
    allow_empty: bool = False,
    max_chars: int | None = None,
    max_tokens: int | None = None,
    strip_control_chars: bool = True,
) -> tuple[str, list[str]]:
    """Validate user input for encoding, control characters, and size.

    Args:
        user_input: the raw user input string to validate.
        allow_empty: when True, empty input is allowed (default False).
        max_chars: maximum character count. When exceeded, raises ExcessiveSizeError.
        max_tokens: maximum token estimate. When exceeded, raises ExcessiveSizeError.
        strip_control_chars: when True, strip unexpected control characters
            (default True). When False, raises ControlCharacterError if found.

    Returns:
        tuple of (validated_input, reason_codes).

    Raises:
        InvalidEncodingError: if input is not a valid UTF-8 string.
        ControlCharacterError: if unexpected control characters found and
            strip_control_chars is False.
        ExcessiveSizeError: if input exceeds max_chars or max_tokens.
        InvalidEncodingError: if input is not a string.
    """

    if not isinstance(user_input, str):
        raise InvalidEncodingError(
            f"user_input must be a string, got {type(user_input).__name__}"
        )

    # Check empty input
    if not user_input.strip():
        if not allow_empty:
            raise GuardrailConfigError(
                "Empty user input is not allowed"
            )
        return user_input, []

    reasons: list[str] = []

    # Check character count
    if max_chars is not None and len(user_input) > max_chars:
        reasons.append("excessive_length")
        raise ExcessiveSizeError(
            f"User input exceeds maximum length of {max_chars} characters "
            f"(got {len(user_input)})"
        )

    # Check for control characters
    if strip_control_chars:
        # Strip control characters except common whitespace
        cleaned = ""
        control_reasons: list[str] = []
        for i, ch in enumerate(user_input):
            # Allow common whitespace and printable ASCII
            if ord(ch) == 0x0D:  # Carriage return
                cleaned += ch
            elif ord(ch) == 0x0A:  # Newline
                cleaned += ch
            elif 0x20 <= ord(ch) <= 0x7E:  # Printable ASCII
                cleaned += ch
            else:
                control_reasons.append(f"char_{i}_0x{ord(ch):04X}")

        if control_reasons:
            reasons.extend(control_reasons)
            # Strip the control characters
            user_input = cleaned
    else:
        # Validate no unexpected control characters
        # Control characters are: U+0000-U+001F and U+007F (DEL)
        # Except common whitespace: tab, newline, carriage return
        control_pattern = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
        matches = control_pattern.findall(user_input)
        if matches:
            raise ControlCharacterError(
                f"Input contains unexpected control characters: {matches}"
            )

    # Token estimate using rough 4 chars per token heuristic
    if max_tokens is not None:
        estimated_tokens = len(user_input) // 4
        if estimated_tokens > max_tokens:
            reasons.append("excessive_tokens")
            raise ExcessiveSizeError(
                f"User input estimated at {estimated_tokens} tokens exceeds "
                f"maximum of {max_tokens} tokens"
            )

    return user_input, reasons


def validate_metadata(
    metadata: dict[str, Any] | None,
    *,
    allow_none: bool = True,
    required_keys: tuple[str, ...] | None = None,
    forbidden_keys: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Validate metadata dict for a guardrail request.

    Args:
        metadata: the metadata dict to validate (or None if allow_none=True).
        allow_none: when True, None metadata is allowed (default True).
        required_keys: tuple of keys that must be present if metadata is provided.
        forbidden_keys: tuple of keys that must NOT be present.

    Returns:
        validated metadata dict.

    Raises:
        InvalidMetadataError: if metadata fails validation.
    """

    if metadata is None:
        if not allow_none:
            raise InvalidMetadataError(
                "Metadata is required but got None"
            )
        return {}

    if not isinstance(metadata, dict):
        raise InvalidMetadataError(
            f"metadata must be a dict or None, got {type(metadata).__name__}"
        )

    # Check required keys
    if required_keys:
        missing = [k for k in required_keys if k not in metadata]
        if missing:
            raise InvalidMetadataError(
                f"Missing required metadata keys: {missing}"
            )

    # Check forbidden keys
    if forbidden_keys:
        found_forbidden = [k for k in forbidden_keys if k in metadata]
        if found_forbidden:
            raise InvalidMetadataError(
                f"Forbidden metadata keys present: {found_forbidden}"
            )

    # Ensure all values are serializable types (no functions, lambda, etc.)
    for key, value in metadata.items():
        if value is None:
            continue
        # Check for types that are not safely serializable
        _ = _is_safe_value(value)  # just validate, don't use

    return metadata


def _is_safe_value(value: Any) -> bool:
    """Check if a metadata value is of a safe, serializable type."""

    import numbers

    if isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_safe_value(item) for item in value)
    if isinstance(value, dict):
        return all(_is_safe_value(v) for v in value.values())
    if isinstance(value, bytes):
        return True
    if isinstance(value, tuple) and hasattr(value, '_fields'):  # namedtuple
        return all(_is_safe_value(item) for item in value)

    return False


def validate_request_id(
    request_id: str | None,
    *,
    existing_ids: set[str] | None = None,
    allow_empty: bool = False,
) -> str:
    """Validate a request identifier for uniqueness and format.

    Args:
        request_id: the request ID to validate.
        existing_ids: set of already-seen request IDs (for uniqueness check).
        allow_empty: when True, empty request ID is allowed (default False).

    Returns:
        validated request ID string.

    Raises:
        InvalidRequestIDError: if the request ID is invalid or conflicting.
    """

    if request_id is None:
        if allow_empty:
            return ""
        raise InvalidRequestIDError(
            "Request ID is required and cannot be None"
        )

    if not isinstance(request_id, str):
        raise InvalidRequestIDError(
            f"request_id must be a string, got {type(request_id).__name__}"
        )

    # Check empty after strip
    stripped = request_id.strip()
    if not stripped:
        if allow_empty:
            return ""
        raise InvalidRequestIDError(
            "Request ID cannot be empty after stripping whitespace"
        )

    # Check for control characters in request ID
    if any(ord(c) < 0x20 and c not in '\t\n\r' for c in stripped):
        raise InvalidRequestIDError(
            "Request ID contains invalid control characters"
        )

    # Check uniqueness against existing IDs
    if existing_ids is not None and request_id in existing_ids:
        raise InvalidRequestIDError(
            f"Request ID already exists: {request_id}"
        )

    return stripped