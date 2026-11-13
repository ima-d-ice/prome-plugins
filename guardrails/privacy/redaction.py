"""Privacy redaction module for the Guardrails ecosystem.

Produces safe derived copies with sensitive information masked. Never
permanently modifies source documents, original chunks, retrieval results,
or provenance records.

All redaction is explicitly configured via PrivacyRules. The original text
is always preserved alongside any redacted derived copy.
"""

from __future__ import annotations

from typing import Any

from guardrails.privacy.detectors import (
    DetectorResult,
    PrivacyDetector,
    PrivacyRules,
)


class PrivacyRedactor:
    """Produce safe derived copies with sensitive information masked.

    Never permanently modifies:
    - Source documents.
    - Original chunks.
    - Retrieval results.
    - Provenance records.

    For request/output redaction, always produce a safe derived copy.
    The original text is preserved alongside the redacted copy.

    Redaction must be explicitly configured via PrivacyRules.
    """

    def __init__(self, rules: PrivacyRules | None = None, detector: PrivacyDetector | None = None) -> None:
        """Initialize the privacy redactor.

        Args:
            rules: PrivacyRules instance for action configuration.
                When None, uses default rules (default_action="detect").
            detector: PrivacyDetector instance. When None, a default detector
                with all categories enabled is created.
        """

        self._rules = rules or PrivacyRules()
        self._detector = detector or PrivacyDetector()

    def redact(self, text: str) -> dict[str, Any]:
        """Redact sensitive information from the given text.

        Produces a safe derived copy with sensitive values masked. The
        original text is preserved and returned alongside the redacted copy.

        Args:
            text: the text to redact.

        Returns:
            dict with the following keys:
                - original_text: the original (unmodified) text
                - redacted_text: the text with sensitive values masked
                - categories_detected: list of detected sensitive categories
                - action_taken: the enforcement action applied
                - detection_count: number of sensitive findings
        """

        # Run detection
        detector_result = self._detector.detect(text)

        # Evaluate action based on rules
        action = self._rules.evaluate_action(detector_result.categories)

        # Apply action to detector result
        derived = self._rules.apply_to_detector_result(detector_result, action)

        # Add original text preservation
        derived["original_text"] = text

        return derived

    def redact_request(
        self,
        request_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Redact sensitive information from a request dict.

        Scans all string values in the request for sensitive patterns and
        produces a derived copy with masked values.

        Args:
            request_data: the request data dict (typically user input,
                context items, or generated output).

        Returns:
            A derived request dict with sensitive values masked, alongside
            the original data for comparison/auditing.
        """

        derived: dict[str, Any] = {}
        original: dict[str, Any] = {}

        for key, value in request_data.items():
            # Preserve original
            original[key] = value

            # Redact string values
            if isinstance(value, str):
                redacted = self.redact(value)
                derived[key] = redacted["redacted_text"]
            elif isinstance(value, dict):
                # Recursively redact dict values
                derived[key] = self.redact_request(value)
                # Also preserve original reference (not a deep copy, just the structure)
                derived[f"_original_{key}"] = value
            elif isinstance(value, list):
                # Redact string items in list
                derived[key] = self._redact_list_items(value)
            else:
                # Non-string, non-dict values pass through
                derived[key] = value

        # Also preserve the full original for comparison
        derived["_original_request"] = original

        return derived

    def _redact_list_items(self, items: list[Any]) -> list[Any]:
        """Redact sensitive strings in a list, preserving non-string items."""

        result: list[Any] = []
        for item in items:
            if isinstance(item, str):
                redacted = self.redact(item)
                result.append(redacted["redacted_text"])
            elif isinstance(item, dict):
                result.append(self.redact_request(item))
            else:
                result.append(item)
        return result

    def redact_output(
        self,
        output_text: str,
    ) -> dict[str, Any]:
        """Redact sensitive information from generated output.

        Produces a safe derived copy of the generated text with sensitive
        values masked. The original output is preserved.

        Args:
            output_text: the generated text to redact.

        Returns:
            dict with original_text and redacted_text, plus detection info.
        """

        return self.redact(output_text)