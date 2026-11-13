"""Exceptions for the Guardrails ecosystem.

All exceptions are dependency-free and contain no imports of model SDKs,
HTTP clients, or optional heavy dependencies. They follow the same pattern
as the generation exceptions: a base class plus narrow subclasses.
"""

from __future__ import annotations


class GuardrailError(Exception):
    """Base class for all guardrail errors."""


class GuardrailConfigError(GuardrailError, ValueError):
    """Invalid guardrail configuration (bad limits, bad stage, ...)."""


class GuardrailDependencyError(GuardrailError, ImportError):
    """An optional dependency needed by the guardrail is not installed.

    The message always names the extra to install.
    """


class GuardrailEvaluationError(GuardrailError):
    """A guardrail evaluation failed (classification error, policy error, ...)."""


class GuardrailPromptGuardError(GuardrailError):
    """Prompt Guard classification failed or returned an error."""


class GuardrailSafeguardError(GuardrailError):
    """Safety evaluator (GPT-OSS Safeguard) failed or returned an error."""


class GuardrailPolicyError(GuardrailError):
    """Policy loading, validation, or execution error."""


class DuplicatePolicyIDError(GuardrailPolicyError):
    """Attempted to register a policy with a duplicate ID."""


class InvalidStageError(GuardrailConfigError):
    """Policy references an unknown guardrail stage."""


class InvalidActionError(GuardrailConfigError):
    """Policy references an invalid enforcement action."""


class MissingMandatoryFieldError(GuardrailConfigError):
    """Policy is missing a required field."""


class InvalidPriorityError(GuardrailConfigError):
    """Policy has an invalid priority value."""


class UnknownConditionTypeError(GuardrailConfigError):
    """Policy contains an unknown condition type."""


class AccessControlError(GuardrailError):
    """Access-control denial or policy conflict."""


class SourceTrustError(GuardrailError):
    """Source trust state error or conflict."""


class PrivacyRedactionError(GuardrailError):
    """Privacy redaction operation failed."""


class CitationValidationError(GuardrailError):
    """Citation validation failed."""


class EnforcementError(GuardrailError):
    """Enforcement engine error or policy conflict."""
