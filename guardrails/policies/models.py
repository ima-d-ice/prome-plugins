"""Policy data model for the Guardrails ecosystem.

A policy is explicit, versioned, inspectable, and deterministic. It contains
all metadata needed for the policy engine to make enforcement decisions.

Follow the existing RAG contract patterns: dependency-free, dataclasses with
frozen=True where immutability is desired, __post_init__ validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from guardrails.contracts.exceptions import (
    GuardrailConfigError,
    InvalidActionError,
    InvalidPriorityError,
    InvalidStageError,
    MissingMandatoryFieldError,
    UnknownConditionTypeError,
)

SUPPORTED_ACTIONS = frozenset({"allow", "warn", "block", "review"})
SUPPORTED_STAGES = frozenset({
    "input",
    "query",
    "ingestion",
    "retrieved_context",
    "context_assembly",
    "generation_input",
    "generation_output",
    "citation_validation",
})

DEFAULT_PRIORITY = 0


def _validate_policy(
    policy_id: str,
    version: str,
    description: str,
    enforcement_action: str,
    priority: int,
    applicable_stages: Any | None,
    conditions: Any | None,
) -> None:
    """Validate policy constructor arguments."""

    # Validate policy_id is non-empty
    if not policy_id or not isinstance(policy_id, str):
        raise MissingMandatoryFieldError("policy_id must be a non-empty string")

    # Validate version is non-empty
    if not version or not isinstance(version, str):
        raise MissingMandatoryFieldError("policy version must be a non-empty string")

    # Validate description is non-empty
    if not description or not isinstance(description, str):
        raise MissingMandatoryFieldError("policy description must be a non-empty string")

    # Validate enforcement_action
    if enforcement_action not in SUPPORTED_ACTIONS:
        raise InvalidActionError(
            f"Invalid enforcement_action: {enforcement_action}. "
            f"Must be one of {SUPPORTED_ACTIONS}"
        )

    # Validate priority is an integer
    if not isinstance(priority, int):
        raise InvalidPriorityError(
            f"Policy priority must be an integer, got {type(priority).__name__}"
        )

    # Validate applicable_stages if provided
    if applicable_stages is not None:
        if not isinstance(applicable_stages, frozenset):
            raise GuardrailConfigError(
                "applicable_stages must be a frozenset or None"
            )
        invalid_stages = applicable_stages - SUPPORTED_STAGES
        if invalid_stages:
            raise InvalidStageError(
                f"Invalid stage(s) in applicable_stages: {invalid_stages}. "
                f"Must be subset of {SUPPORTED_STAGES}"
            )

    # Validate conditions type structure
    if conditions is not None:
        if not isinstance(conditions, dict):
            raise UnknownConditionTypeError(
                "Policy conditions must be a dict"
            )


@dataclass(frozen=True)
class Policy:
    """A deterministic, versioned security policy for the RAG pipeline.

    Attributes:
        policy_id: unique identifier for this policy (must be unique across
            all policies in a configuration).
        version: policy version string for reproducible behavior.
        description: human-readable description of the policy's purpose.
        enabled: when False, the policy is ignored (deprecated rather than
            deleted).
        applicable_stages: pipeline stages this policy applies to. When empty
            or None, the policy applies to all stages.
        conditions: policy conditions that must be met for the enforcement
            action to trigger. Each condition has a "type" and "parameters".
        enforcement_action: one of "allow", "warn", "block", "review".
        priority: integer priority for deterministic precedence. Higher priority
            policies execute first. Default: 0.
        expiry_date: optional expiry date; policies past this date are
            automatically disabled.
        deprecation_meta: optional metadata for deprecation tracking.
    """

    policy_id: str
    version: str
    description: str
    enabled: bool = True
    applicable_stages: frozenset[str] | None = None
    conditions: dict[str, Any] | None = None
    enforcement_action: str = "allow"
    priority: int = DEFAULT_PRIORITY
    expiry_date: datetime | None = None
    deprecation_meta: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _validate_policy(
            self.policy_id,
            self.version,
            self.description,
            self.enforcement_action,
            self.priority,
            self.applicable_stages,
            self.conditions,
        )

    def applies_to_stage(self, stage: str) -> bool:
        """Check if this policy applies to the given pipeline stage."""

        if not self.enabled:
            return False

        if self.applicable_stages is None:
            return True

        return stage in self.applicable_stages

    def is_expired(self) -> bool:
        """Check if this policy has passed its expiry date."""

        if self.expiry_date is None:
            return False

        now = datetime.now(timezone.utc)
        return now > self.expiry_date


def default_policy() -> Policy:
    """Provide a neutral default policy for testing.

    Returns a policy that effectively does nothing (allow all stages),
    useful as a baseline when no custom policies are configured.
    """

    return Policy(
        policy_id="default-neutral",
        version="1.0.0",
        description="Neutral default policy - no restrictions",
        enabled=True,
        enforcement_action="allow",
        priority=0,
    )
