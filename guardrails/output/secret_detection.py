"""Secret detection in generated output for the Guardrails ecosystem.

Detects sensitive information patterns in generated text output using
the local, deterministic privacy detection layer.

Detection categories (same as PrivacyDetector):
- email addresses
- phone numbers
- common API key formats
- bearer tokens
- access tokens
- common cloud credential patterns
- private keys and obvious secret blocks
- password-like credential fields

Does not send discovered secrets to external providers for classification.

Returns structured validation findings without exposing actual secret values
in diagnostics.
"""

from __future__ import annotations

from typing import Any

from guardrails.contracts.exceptions import GuardrailError
from guardrails.privacy.detectors import PrivacyDetector, DetectorResult


class OutputSecretDetectionError(GuardrailError):
    """Raised when output secret detection fails."""


class OutputSecretScanner:
    """Detect sensitive secrets in generated output text.

    Uses the local PrivacyDetector for deterministic pattern matching.
    Does not call external providers or send discovered secrets anywhere.

    Returns structured findings that can be used for WARN/BLOCK/REVIEW
    decisions without exposing actual secret values in diagnostics.
    """

    def __init__(self, detector: PrivacyDetector | None = None) -> None:
        """Initialize the output secret scanner.

        Args:
            detector: PrivacyDetector instance. When None, a default detector
                with all categories enabled is created.
        """

        self._detector = detector or PrivacyDetector(enabled_categories=None)

    def scan(self, output_text: str) -> dict[str, Any]:
        """Scan output text for sensitive secrets.

        Args:
            output_text: the generated text to scan.

        Returns:
            dict with:
                - detected: True if any sensitive patterns were found.
                - categories: list of detected sensitive categories.
                - count: number of detections.
                - redaction_mask: the output text with sensitive values
                  replaced by masks (safe derived copy).
                - diagnostics: structured diagnostics (NO actual secret values).
        """

        if not isinstance(output_text, str):
            raise OutputSecretDetectionError(
                f"Output text must be a string, got {type(output_text).__name__}"
            )

        if not output_text.strip():
            return {
                "detected": False,
                "categories": [],
                "count": 0,
                "redaction_mask": output_text,
                "diagnostics": {
                    "note": "Empty output text; no secret detection performed",
                },
            }

        # Run detection
        detector_result: DetectorResult = self._detector.detect(output_text)

        # Build diagnostics - NEVER include actual secret values
        # Only include category info and position counts
        diagnostics: dict[str, Any] = {
            "categories_detected": detector_result.categories,
            "detection_count": len(detector_result.positions),
            "note": "Diagnostics exclude actual secret values per security policy",
        }

        # Build redaction mask (safe derived copy)
        redaction_mask = detector_result.redacted_text

        return {
            "detected": len(detector_result.categories) > 0,
            "categories": detector_result.categories,
            "count": len(detector_result.positions),
            "redaction_mask": redaction_mask,
            "diagnostics": diagnostics,
        }