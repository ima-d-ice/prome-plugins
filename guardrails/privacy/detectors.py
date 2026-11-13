"""Privacy and sensitive information detection for the Guardrails ecosystem.

Implement a modular, local, deterministic detection layer for initial
detection categories. Does not claim comprehensive PII detection.

Detection categories:
- Email addresses.
- Phone numbers.
- Common API-key formats.
- Bearer tokens.
- Access tokens.
- Common cloud credential patterns.
- Private keys and obvious secret blocks.
- Password-like credential fields.
- Configurable application-specific patterns.

Actions:
- DETECT: identify sensitive patterns.
- WARN: flag but allow through with warning.
- MASK: produce derived copy with sensitive values masked.
- BLOCK: block the request/review.
- REVIEW: send for human review.

Redaction must be explicitly configured.

Never permanently modify:
- Source documents.
- Original chunks.
- Retrieval results.
- Provenance records.

For request/output redaction, produce a safe derived copy.

Preserve original source IDs and record which categories were redacted.

Never log full detected secrets.

Never expose Groq API keys, authorization headers or environment values in diagnostics.

Include false-positive and false-negative tests.

Do not send discovered secrets to external providers for classification.
"""

from __future__ import annotations

import re
from typing import Any

from guardrails.contracts.exceptions import (
    GuardrailConfigError,
    GuardrailError,
)

#: Regex patterns for initial detection categories
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_PATTERN = re.compile(r"(?:\+?1?[-\s.]?)?\(?[0-9]{3}\)?[-\s.]?[0-9]{3}[-\s.]?[0-9]{4}")
_API_KEY_PATTERN = re.compile(r"(?:sk-[a-zA-Z0-9]{20,}|api[key]_?[_-]?[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{36})")
_BEARER_TOKEN_PATTERN = re.compile(r"Bearer\s+[a-zA-Z0-9._-]{20,}")
_ACCESS_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9._-]{20,}(?::[a-zA-Z0-9._-]{20,})?")
_CLOUD_CREDENTIAL_PATTERN = re.compile(r"(?:AKIA[A-Z0-9]{16}|GCE-[a-zA-Z0-9]{10}|aws_[a-zA-Z0-9]{20})")
_PRIVATE_KEY_PATTERN = re.compile(r"-----BEGIN (?:RSA |EC |OPEN )?PRIVATE KEY-----")
_PASSWORD_FIELD_PATTERN = re.compile(r"(?:password|passwd|pwd|secretkey|api_key|apikey)\s*[:=]\s*['\"][^\n]{3,}", re.IGNORECASE)


class PrivacyDetectionError(GuardrailError):
    """Raised when privacy detection fails."""


class RedactionError(GuardrailError):
    """Raised when redaction operation fails."""


class DetectorResult:
    """Result from a privacy detector.

    Attributes:
        categories: list of detected sensitive categories.
        positions: list of (start, end) character positions for each detection.
        values: list of the actual detected values (redacted in diagnostics).
        redacted_text: the text with detected values replaced by masks.
        confidence: detection confidence per category (0.0 to 1.0).
    """

    def __init__(
        self,
        categories: list[str],
        positions: list[tuple[int, int]],
        values: list[str],
        redacted_text: str,
        confidence: dict[str, float] | None = None,
    ) -> None:
        self.categories = categories
        self.positions = positions
        self.values = values
        self.redacted_text = redacted_text
        self.confidence = confidence or {}

    def __dict__(self) -> dict[str, Any]:
        return {
            "categories": self.categories,
            "positions": self.positions,
            "values": self.values,  # NOTE: values should NOT be logged
            "redacted_text": self.redacted_text,
            "confidence": self.confidence,
        }


class PrivacyDetector:
    """Modular, local, deterministic privacy detection layer.

    Supports the following detection categories:
    - email: email addresses
    - phone: phone numbers
    - api_key: common API key formats
    - bearer_token: bearer token patterns
    - access_token: access token patterns
    - cloud_credential: common cloud credential patterns
    - private_key: private key patterns
    - password_field: password-like credential fields

    Each detector is independent and can be enabled/disabled via the
    configured rules. Detection is deterministic (no probabilistic models
    or external API calls).
    """

    # Supported detection categories
    CATEGORIES = {
        "email",
        "phone",
        "api_key",
        "bearer_token",
        "access_token",
        "cloud_credential",
        "private_key",
        "password_field",
    }

    def __init__(
        self,
        *,
        enabled_categories: list[str] | None = None,
    ) -> None:
        """Initialize the privacy detector.

        Args:
            enabled_categories: list of categories to enable.
                When None, all categories are enabled.
        """

        if enabled_categories is None:
            self._enabled = self.CATEGORIES.copy()
        else:
            invalid = set(enabled_categories) - self.CATEGORIES
            if invalid:
                raise GuardrailConfigError(
                    f"Unknown detection category(ies): {invalid}. "
                    f"Must be subset of {self.CATEGORIES}"
                )
            self._enabled = set(enabled_categories)

    def detect(self, text: str) -> DetectorResult:
        """Detect sensitive information in the given text.

        Performs deterministic pattern matching against all enabled
        categories. Returns a DetectorResult with detected values and
        redacted text.

        Args:
            text: the text to scan for sensitive information.

        Returns:
            DetectorResult containing:
                - categories: list of detected sensitive categories
                - positions: character positions of each detection
                - values: the actual detected values (DO NOT log these)
                - redacted_text: the text with sensitive values masked
                - confidence: detection confidence per category
        """

        if not isinstance(text, str):
            raise PrivacyDetectionError(
                f"Input must be a string, got {type(text).__name__}"
            )

        if not text:
            return DetectorResult(
                categories=[],
                positions=[],
                values=[],
                redacted_text="",
                confidence={},
            )

        # Run all enabled detectors and collect results
        all_categories: list[str] = []
        all_positions: list[tuple[int, int]] = []
        all_values: list[str] = []
        all_confidences: dict[str, float] = {}

        # Track redacted text - start with original and apply masks
        redacted = text

        # Detection order matters for overlap handling; run them all
        detectors: list[tuple[str, re.Pattern[str], callable[[re.Match[str]], str]]] = [
            ("email", _EMAIL_PATTERN, self._mask_email),
            ("phone", _PHONE_PATTERN, self._mask_phone),
            ("api_key", _API_KEY_PATTERN, self._mask_api_key),
            ("bearer_token", _BEARER_TOKEN_PATTERN, self._mask_bearer_token),
            ("access_token", _ACCESS_TOKEN_PATTERN, self._mask_access_token),
            ("cloud_credential", _CLOUD_CREDENTIAL_PATTERN, self._mask_cloud_credential),
            ("private_key", _PRIVATE_KEY_PATTERN, self._mask_private_key),
            ("password_field", _PASSWORD_FIELD_PATTERN, self._mask_password_field),
        ]

        for category, pattern, mask_func in detectors:
            if category not in self._enabled:
                continue

            # Find all matches in the current redacted text
            for match in pattern.finditer(redacted):
                start, end = match.span()
                value = match.group(0)

                # Check for overlap with already-detected positions
                if self._position_overlaps(all_positions, (start, end)):
                    # Skip overlap - already detected by an earlier category
                    continue

                # Record the detection
                all_categories.append(category)
                all_positions.append((start, end))
                all_values.append(value)

                # Apply mask to the redacted text
                masked = mask_func(match)
                redacted = redacted[:start] + masked + redacted[end:]

                # Set confidence (default 1.0 for deterministic pattern matching)
                all_confidences[category] = all_confidences.get(category, 0.0) + 0.1

        # Cap confidence at 1.0
        for cat in all_confidences:
            all_confidences[cat] = min(1.0, all_confidences[cat])

        return DetectorResult(
            categories=all_categories,
            positions=all_positions,
            values=all_values,
            redacted_text=redacted,
            confidence=all_confidences,
        )

    def _position_overlaps(
        self, existing: list[tuple[int, int]], new_pos: tuple[int, int]
    ) -> bool:
        """Check if a new position overlaps with any existing positions.

        Overlap means the new range shares any characters with an already-
        detected range. This prevents double-detection of the same sensitive
        value when multiple patterns match overlapping regions.
        """

        new_start, new_end = new_pos
        for exc_start, exc_end in existing:
            # Overlap if new starts before existing ends AND new ends after existing starts
            if new_start < exc_end and new_end > exc_start:
                return True
        return False

    # --- Masking functions ---

    @staticmethod
    def _mask_email(match: re.Match[str]) -> str:
        """Mask an email address: keep prefix, replace domain."""
        email = match.group(0)
        # Simple masking: a**@xxx.yyy
        parts = email.split("@")
        if len(parts) == 2:
            prefix = parts[0]
            domain = parts[1]
            # Mask the prefix: show first char, rest as asterisks
            if len(prefix) <= 1:
                masked_prefix = prefix
            else:
                masked_prefix = prefix[0] + "*" * (len(prefix) - 1)
            return f"{masked_prefix}@{domain}"
        return "***@***.***"

    @staticmethod
    def _mask_phone(match: re.Match[str]) -> str:
        """Mask a phone number: show first 3, rest as asterisks."""
        phone = match.group(0)
        # Keep first 6 chars, mask the rest
        if len(phone) <= 6:
            return "*" * len(phone)
        return phone[:6] + "*" * (len(phone) - 6)

    @staticmethod
    def _mask_api_key(match: re.Match[str]) -> str:
        """Mask an API key: show first 4, rest as asterisks."""
        key = match.group(0)
        if len(key) <= 4:
            return "*" * len(key)
        return key[:4] + "*" * (len(key) - 4)

    @staticmethod
    def _mask_bearer_token(match: re.Match[str]) -> str:
        """Mask a bearer token: show 'Bearer ', mask the token."""
        # The pattern includes 'Bearer ', mask everything after
        return "Bearer ***MASKED***"

    @staticmethod
    def _mask_access_token(match: re.Match[str]) -> str:
        """Mask an access token: show first 4, rest as asterisks."""
        token = match.group(0)
        # Remove any prefix-like stuff and mask the token portion
        if len(token) <= 4:
            return "*" * len(token)
        return token[:4] + "*" * (len(token) - 4)

    @staticmethod
    def _mask_cloud_credential(match: re.Match[str]) -> str:
        """Mask a cloud credential: show first/last chars, mask middle."""
        cred = match.group(0)
        if len(cred) <= 4:
            return "*" * len(cred)
        # Show first 3 and last 3, mask middle
        return cred[:3] + "*" * (len(cred) - 6) + cred[-3:]

    @staticmethod
    def _mask_private_key(match: re.Match[str]) -> str:
        """Mask a private key: replace with ***MASKED***."""
        return "***MASKED_PRIVATE_KEY***"

    @staticmethod
    def _mask_password_field(match: re.Match[str]) -> str:
        """Mask a password field: replace with ***MASKED***."""
        return "***MASKED***"


class PrivacyRules:
    """Privacy detection rules and action configuration.

    Supports the following actions:
    - DETECT: identify sensitive patterns (default, always performed).
    - WARN: flag but allow through with warning.
    - MASK: produce derived copy with sensitive values masked.
    - BLOCK: block the request/review.
    - REVIEW: send for human review.

    Rules are evaluated in order; the first matching action is applied.
    """

    def __init__(
        self,
        *,
        default_action: str = "detect",
        actions: dict[str, str] | None = None,
        mask_sensitive: bool = True,
    ) -> None:
        """Initialize the privacy rules.

        Args:
            default_action: default action when a sensitive pattern is detected.
                One of: "detect", "warn", "mask", "block", "review".
            actions: per-category action overrides. Keys are category names,
                values are one of: "detect", "warn", "mask", "block", "review".
            mask_sensitive: when True (default), masked redaction is applied
                in detector results.
        """

        self.default_action = default_action
        self._actions: dict[str, str] = actions or {}
        self.mask_sensitive = mask_sensitive

    def evaluate_action(
        self,
        categories: list[str],
        *,
        override_per_category: dict[str, str] | None = None,
    ) -> str:
        """Determine the action to take for detected categories.

        Args:
            categories: list of detected sensitive categories.
            override_per_category: optional per-category action overrides.

        Returns:
            The enforcement action: "detect", "warn", "mask", "block", or "review".
        """

        # Check per-category overrides first
        if override_per_category:
            for cat in categories:
                if cat in override_per_category:
                    return override_per_category[cat]

        # Check per-category rules
        for cat in categories:
            if cat in self._actions:
                return self._actions[cat]

        # Use default action
        return self.default_action

    def apply_to_detector_result(
        self,
        result: DetectorResult,
        action: str | None = None,
    ) -> dict[str, Any]:
        """Apply an action to a detector result and return a derived copy.

        Args:
            result: the DetectorResult from PrivacyDetector.detect().
            action: the action to apply. When None, uses the action from
                self.evaluate_action().

        Returns:
            A derived dict suitable for inclusion in request/output data
            with sensitive values handled according to the action.
        """

        if action is None:
            action = self.evaluate_action(result.categories)

        # Build the derived copy based on the action
        derived: dict[str, Any] = {
            "categories_detected": result.categories,
            "redacted_text": result.redacted_text,
            "action_taken": action,
        }

        # Include position and value info only when action is WARN or below
        # (for diagnostics, never log actual values)
        if action in ("warn", "detect", "review"):
            derived["detection_positions"] = result.positions
            # Do NOT include actual values in the derived copy
            derived["detection_count"] = len(result.positions)
        elif action in ("block", "review"):
            # For block/review, include count but not values
            derived["detection_count"] = len(result.positions)
            derived["detection_positions"] = result.positions

        # Always include the redacted text (safe derived copy)
        derived["redacted_text"] = result.redacted_text

        # Record which categories were redacted
        if result.categories:
            derived["categories_redacted"] = result.categories
        else:
            derived["categories_redacted"] = []

        return derived
