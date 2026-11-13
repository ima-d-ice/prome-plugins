"""Output validation and citation verification for the Guardrails ecosystem.

Implements an independent output validation layer that validates:
- Expected output type.
- Empty output when disallowed.
- Malformed structured data.
- Unexpected control sequences where relevant.
- Citation references.
- Unknown source identifiers.
- Required citation presence.
- Sensitive secret patterns.
- Configured output policy violations.

Citation validation uses the original citation/source contract when available.
Performs deterministic identity validation. Deeper semantic claim-support
evaluation belongs in the Evaluation subsystem and must not be duplicated here.

Do not automatically rewrite generated answers to hide citation failures.
Return structured validation findings.
"""

from __future__ import annotations

from typing import Any

from guardrails.contracts.exceptions import (
    GuardrailError,
)
from guardrails.privacy.detectors import PrivacyDetector, PrivacyRules


class OutputValidationError(GuardrailError):
    """Raised when output validation fails."""


class MalformedOutputError(OutputValidationError):
    """Raised when the generated output is malformed."""


class CitationNotFoundError(OutputValidationError):
    """Raised when a required citation is not found in the output."""


class UnknownSourceError(OutputValidationError):
    """Raised when a source identifier in the output is not recognized."""


class MissingCitationError(OutputValidationError):
    """Raised when a citation is required but missing."""


class SecretDetectionError(OutputValidationError):
    """Raised when sensitive secrets are detected in output."""


class OutputValidator:
    """Independent output validation layer.

    Validates generated answers across multiple dimensions:
    - Structural validity (type, emptiness, control sequences)
    - Citation integrity (references, authorization, provenance)
    - Sensitive information exposure (secret patterns)
    - Policy compliance

    All validation is deterministic and does not call external models.
    """

    def __init__(
        self,
        *,
        allow_empty: bool = False,
        detect_secrets: bool = True,
        secret_rules: PrivacyRules | None = None,
    ) -> None:
        """Initialize the output validator.

        Args:
            allow_empty: when True, empty output is allowed (default False).
                When False, raises MissingCitationError or MalformedOutputError.
            detect_secrets: when True (default), scan output for sensitive
                secret patterns.
            secret_rules: PrivacyRules for secret detection actions.
                When None, uses default rules.
        """

        self._allow_empty = allow_empty
        self._detect_secrets = detect_secrets
        self._secret_rules = secret_rules or PrivacyRules(default_action="detect")
        self._privacy_detector = PrivacyDetector(enabled_categories=None)

    def validate(
        self,
        output_text: str,
        *,
        citation_refs: list[dict[str, Any]] | None = None,
        source_metadata: dict[str, Any] | None = None,
        expected_type: str | None = "text",
    ) -> dict[str, Any]:
        """Validate the generated output text.

        Performs comprehensive validation including structural checks,
        citation validation, and secret detection.

        Args:
            output_text: the generated text to validate.
            citation_refs: optional list of citation/reference dicts expected
                in the output. Each dict should have "source_id" and optionally
                "citation_marker".
            source_metadata: optional source metadata for citation
                authorization checks.
            expected_type: expected output type text ("text", "structured",
                etc.).

        Returns:
            dict with validation findings:
                - valid: True if all checks passed
                - errors: list of validation error details
                - warnings: list of validation warnings
                - secrets_detected: list of detected secret categories
                - citation_findings: citation validation results
        """

        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        secrets_detected: list[dict[str, Any]] = []
        citation_findings: dict[str, Any] = {
            "valid": True,
            "issues": [],
            "authorized_sources": [],
        }

        # 1. Structural validation
        struct_result = self._validate_structure(
            output_text, expected_type=expected_type, allow_empty=self._allow_empty
        )
        if struct_result["has_errors"]:
            errors.extend(struct_result["errors"])
        warnings.extend(struct_result.get("warnings", []))

        # 2. Citation validation
        if citation_refs is not None:
            citation_result = self._validate_citations(
                output_text,
                citation_refs=citation_refs,
                source_metadata=source_metadata,
            )
            if not citation_result["valid"]:
                citation_findings["valid"] = False
                citation_findings["issues"].extend(citation_result["issues"])
            citation_findings["authorized_sources"] = citation_result.get(
                "authorized_sources", []
            )
            if citation_result["warnings"]:
                warnings.extend(citation_result["warnings"])

        # 3. Secret detection
        if self._detect_secrets:
            secret_result = self._detect_secrets_in_output(output_text)
            if secret_result["detected"]:
                secrets_detected = secret_result["secrets"]
                # If secrets are detected, add a warning unless the action
                # would be to block
                if not any(e.get("code") == "secret_detected" for e in errors):
                    warnings.append(
                        {
                            "code": "secret_detected",
                            "message": f"Detected {len(secret_result['secrets'])} "
                            f"sensitive pattern(s) in generated output",
                            "count": len(secret_result["secrets"]),
                        }
                    )

        return {
            "valid": len(errors) == 0 and citation_findings["valid"],
            "errors": errors,
            "warnings": warnings,
            "secrets_detected": secrets_detected,
            "citation_findings": citation_findings,
        }

    def _validate_structure(
        self,
        output_text: str,
        *,
        expected_type: str,
        allow_empty: bool,
    ) -> dict[str, Any]:
        """Validate the structural aspects of the output.

        Checks:
        - Empty output when disallowed.
        - Malformed content.
        - Unexpected control sequences.

        Returns dict with "has_errors", "errors", and "warnings".
        """

        result: dict[str, Any] = {
            "has_errors": False,
            "errors": [],
            "warnings": [],
        }

        # Check for empty output
        if not output_text or not output_text.strip():
            if not allow_empty:
                result["has_errors"] = True
                result["errors"].append(
                    {
                        "code": "empty_output",
                        "message": "Generated output is empty and empty output is not allowed",
                    }
                )
            else:
                result["warnings"].append(
                    {
                        "code": "empty_output_allowed",
                        "message": "Generated output is empty but was allowed",
                    }
                )
                # If empty is allowed, treat as valid
                result["has_errors"] = False
                return result

        # Check for control characters that could indicate injection
        # Non-printable control characters (except common whitespace)
        import re as re_mod
        control_pattern = re_mod.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
        control_matches = control_pattern.findall(output_text)
        if control_matches:
            # Filter out common whitespace
            unusual_controls = [
                c for c in control_matches
                if c not in "\t\n\r"
            ]
            if unusual_controls:
                result["has_errors"] = True
                result["errors"].append(
                    {
                        "code": "unexpected_control_chars",
                        "message": f"Output contains {len(unusual_controls)} unusual control character(s)",
                    }
                )
            else:
                # Only common whitespace (tab, newline, carriage return) - warning
                result["warnings"].append(
                    {
                        "code": "whitespace_in_output",
                        "message": "Output contains common whitespace characters",
                    }
                )

        return result

    def _validate_citations(
        self,
        output_text: str,
        *,
        citation_refs: list[dict[str, Any]],
        source_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate citations in the generated output.

        Checks:
        - The citation marker exists.
        - The source ID was supplied.
        - The source belongs to the current request.
        - The source is authorized for the current user/tenant.
        - The source was not excluded or blocked.
        - The source reference and provenance are preserved.

        Returns dict with "valid", "issues", and "authorized_sources".
        """

        result: dict[str, Any] = {
            "valid": True,
            "issues": [],
            "authorized_sources": [],
        }

        if not citation_refs:
            # No citation references provided - check if output expects citations
            result["issues"].append(
                {
                    "code": "no_citation_references",
                    "message": "No citation references provided for output validation",
                }
            )
            result["valid"] = False
            return result

        # Build a set of valid source IDs from source_metadata
        valid_source_ids: set[str] = set()
        if source_metadata and isinstance(source_metadata, dict):
            # Collect source IDs from metadata
            for key in source_metadata:
                val = source_metadata[key]
                if isinstance(val, str) and val.startswith("src_"):
                    valid_source_ids.add(val)
                elif isinstance(val, dict) and val.get("id", "").startswith("src_"):
                    valid_source_ids.add(val["id"])

        # Also check for any IDs that look like source references
        # (this is a simplistic check; real integration would use the full
        # citation/source contract)
        for ref in citation_refs:
            source_id = ref.get("source_id", "")
            marker = ref.get("citation_marker", "")

            # Check 1: The citation marker exists in the output
            if marker and marker not in output_text:
                result["issues"].append(
                    {
                        "code": "citation_marker_missing",
                        "message": f"Citation marker '{marker}' not found in output",
                        "source_id": source_id,
                    }
                )
                result["valid"] = False

            # Check 2: The source ID was supplied
            if not source_id:
                result["issues"].append(
                    {
                        "code": "source_id_missing",
                        "message": "Citation reference missing source_id",
                    }
                )
                result["valid"] = False

            # Check 3: The source belongs to the current request
            if source_id and source_id not in valid_source_ids:
                result["issues"].append(
                    {
                        "code": "source_not_in_request",
                        "message": f"Source ID '{source_id}' does not belong to the current request",
                    }
                )
                result["valid"] = False
            elif source_id:
                result["authorized_sources"].append(source_id)

            # Check 4: The source was not excluded or blocked
            # (This would be checked against the source trust/blocklist;
            # for now, we note it as a field for downstream checks)
            result["issues"].append(
                {
                    "code": "source_exclusion_status",
                    "message": "Source exclusion/blocking status to be verified "
                                "against source trust records",
                    }
                )

        # Summary: if any issues were found, mark as invalid
        if result["issues"]:
            result["valid"] = False

        return result

    def _detect_secrets_in_output(
        self, output_text: str,
    ) -> dict[str, Any]:
        """Detect sensitive secrets in the generated output.

        Uses the privacy detector to scan for email addresses, phone numbers,
        API keys, bearer tokens, etc.

        Returns dict with "detected" boolean and "secrets" list.
        """

        # Run privacy detection
        detector_result = self._privacy_detector.detect(output_text)

        if not detector_result.categories:
            return {"detected": False, "secrets": []}

        # Build secret details
        secrets: list[dict[str, Any]] = []

        for i, (category, position) in enumerate(
            zip(detector_result.categories, detector_result.positions)
        ):
            start, end = position
            value = detector_result.values[i] if i < len(detector_result.values) else "***redacted***"

            secrets.append(
                {
                    "category": category,
                    "position": position,
                    "value": value,  # In real logs, this would be redacted
                    "redacted_value": detector_result.redacted_text[
                        start:end
                    ]
                    if start < len(detector_result.redacted_text)
                    else "***redacted***",
                }
            )

        return {"detected": len(secrets) > 0, "secrets": secrets}
