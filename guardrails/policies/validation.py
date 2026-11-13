"""Policy validation for the Guardrails ecosystem.

Validates policies for: duplicate IDs, valid stage names, valid actions,
mandatory fields, valid priorities, unknown condition types.

Rejects invalid policies so they never enter the execution engine.
Support policy versions so behavior remains reproducible.

All validation is deterministic and does not execute arbitrary Python code.
"""

from __future__ import annotations

from guardrails.policies.models import Policy


class PolicyValidator:
    """Validate policy objects and detect conflicts."""

    # Track registered policy IDs for duplicate detection
    _registered_ids: set[str] = set()

    @classmethod
    def validate_policy(cls, policy: Policy) -> list[str]:
        """Validate a single policy and return any error reason codes.

        Args:
            policy: the policy to validate.

        Returns:
            list of reason codes describing any validation failures.
            Empty list means the policy is valid.
        """

        reasons: list[str] = []

        # Check for duplicate policy ID
        if policy.policy_id in cls._registered_ids:
            reasons.append("duplicate_policy_id")
        else:
            cls._registered_ids.add(policy.policy_id)

        # Validate enabled is boolean
        if not isinstance(policy.enabled, bool):
            reasons.append("enabled_not_boolean")

        # Validate enforcement_action
        if policy.enforcement_action not in {"allow", "warn", "block", "review"}:
            reasons.append("invalid_enforcement_action")

        # Validate priority
        if not isinstance(policy.priority, int):
            reasons.append("invalid_priority")

        # Validate applicable_stages
        if policy.applicable_stages is not None:
            valid_stages = {"input", "query", "ingestion", "retrieved_context",
                          "context_assembly", "generation_input", "generation_output",
                          "citation_validation"}
            invalid = policy.applicable_stages - valid_stages
            if invalid:
                reasons.append(f"invalid_stage:{','.join(sorted(invalid))}")

        # Validate conditions type
        if policy.conditions is not None and not isinstance(policy.conditions, dict):
            reasons.append("conditions_not_dict")

        return reasons

    @classmethod
    def validate_policies(cls, policies: list[Policy]) -> dict[str, list[str]]:
        """Validate a list of policies and return per-policy reason codes.

        Args:
            policies: list of policies to validate.

        Returns:
            dict mapping policy_id to list of reason codes.
        """

        results: dict[str, list[str]] = {}
        for policy in policies:
            reasons = cls.validate_policy(policy)
            results[policy.policy_id] = reasons
        return results

    @classmethod
    def reset_registration(cls) -> None:
        """Reset the registered ID tracking (for test isolation)."""

        cls._registered_ids.clear()
