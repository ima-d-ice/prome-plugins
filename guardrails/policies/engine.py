"""Deterministic policy engine for the Guardrails ecosystem.

Implements deterministic precedence for policy execution:

1. Mandatory access-control denial must never be overridden by a model
   classifier returning ALLOW.
2. BLOCK decision must not be silently downgraded to WARN.
3. When policies conflict, apply documented precedence and produce a clear
   decision trace.

The engine executes policies in priority order (higher priority first).
It preserves every individual result and produces a structured decision trace.

Do not execute arbitrary policy-supplied Python. Never use eval, exec or
dynamic code execution. Use JSON, YAML or TOML for configuration.
"""

from __future__ import annotations

from typing import Any

from guardrails.contracts.models import GuardrailDecision, GuardrailStage
from guardrails.policies.models import Policy
from guardrails.policies.validation import PolicyValidator


class PolicyEngine:
    """Deterministic policy engine that executes policies in a predictable order.

    Policy execution precedence rules:

    1. **Mandatory access-control denial**: An access-control denial must never
       be overridden by a model classifier returning ALLOW. If any policy
       returns BLOCK or REVIEW for an access-control stage, that decision is
       final regardless of subsequent policies.

    2. **BLOCK not silently downgraded**: A BLOCK decision must not be
       downgraded to WARN. If any policy returns BLOCK, the final decision
       is BLOCK.

    3. **Priority ordering**: Policies are executed in priority order
       (higher priority first). The first conclusive decision (BLOCK > REVIEW >
       WARN > ALLOW) becomes the result, unless a higher-priority policy later
       overrides it.

    4. **Documented conflict resolution**: When policies conflict, the engine
       produces a clear decision trace showing which policies influenced the
       result and why.

    The engine does NOT:
    - Execute arbitrary Python code supplied by policies
    - Use eval, exec or dynamic code execution
    - Override deterministic access-control denials with model classifiers
    - Silently downgrade BLOCK to WARN
    """

    def __init__(self, policies: list[Policy] | None = None) -> None:
        """Initialize the policy engine.

        Args:
            policies: optional list of Policy instances to register.
                When None, starts with an empty policy set.
        """

        self._policies: list[Policy] = policies or []
        self._validator = PolicyValidator()
        # Reset and re-register all policies
        PolicyValidator._registered_ids.clear()
        for policy in self._policies:
            PolicyValidator._registered_ids.add(policy.policy_id)

    def add_policy(self, policy: Policy) -> None:
        """Add a policy to the engine.

        Args:
            policy: the policy to add.

        Raises:
            DuplicatePolicyIDError: if a policy with the same ID already
                exists.
        """

        if policy.policy_id in PolicyValidator._registered_ids:
            from guardrails.contracts.exceptions import DuplicatePolicyIDError
            raise DuplicatePolicyIDError(
                f"Duplicate policy ID: {policy.policy_id}"
            )

        PolicyValidator._registered_ids.add(policy.policy_id)
        self._policies.append(policy)

    def remove_policy(self, policy_id: str) -> None:
        """Remove a policy by ID.

        Args:
            policy_id: the policy ID to remove.
        """

        self._policies = [p for p in self._policies if p.policy_id != policy_id]
        PolicyValidator._registered_ids.discard(policy_id)

    def execute(
        self,
        stage: GuardrailStage,
        *,
        user_input: str | None = None,
        context_items: list[dict[str, Any]] | None = None,
        source_references: list[str] | None = None,
    ) -> GuardrailDecision:
        """Execute applicable policies for the given pipeline stage.

        Policies are executed in priority order (highest first). The decision
        trace is preserved and the final result follows documented precedence.

        Args:
            stage: the pipeline stage being evaluated.
            user_input: optional user input for context-sensitive policies.
            context_items: optional retrieved context items.
            source_references: optional source document/chunk references.

        Returns:
            GuardrailDecision with the final enforcement decision and full
            metadata.
        """

        # Sort policies by priority (highest first), stable sort preserves
        # insertion order for equal priorities
        sorted_policies = sorted(
            self._policies,
            key=lambda p: p.priority,
            reverse=True,
        )

        # Track decision trace for auditing
        decision_trace: list[dict[str, Any]] = []

        # Track the strongest decisions
        # Order of strength: BLOCK > REVIEW > WARN > ALLOW
        final_decision = GuardrailDecision.ALLOW
        final_reason_codes: list[str] = []
        final_policy_ids: list[str] = []
        final_source_refs: list[str] = []
        final_detector: str | None = None

        for policy in sorted_policies:
            if not policy.applies_to_stage(stage.value):
                continue

            # Policy applies to this stage - execute its logic
            # Since policies are data-driven (conditions + enforcement_action),
            # we evaluate conditions against the provided context and determine
            # the enforcement action.

            policy_result = self._evaluate_policy_conditions(
                policy,
                stage=stage,
                user_input=user_input,
                context_items=context_items or [],
                source_references=source_references or [],
            )

            decision_trace.append({
                "policy_id": policy.policy_id,
                "policy_version": policy.version,
                "action": policy_result["action"],
                "reason_codes": policy_result.get("reason_codes", []),
                "stage": stage.value,
                "applies_to_stage": policy.applies_to_stage(stage.value),
                "is_expired": policy.is_expired(),
            })

            # Apply precedence rules
            action = policy_result["action"]

            # Rule 1: BLOCK is absolute - if any policy returns BLOCK, final is BLOCK
            if action == GuardrailDecision.BLOCK:
                final_decision = GuardrailDecision.BLOCK
                final_reason_codes = policy_result.get("reason_codes", [])
                final_policy_ids = [policy.policy_id]
                final_source_refs = policy_result.get("source_references", [])
                final_detector = policy_result.get("detector_model")
                # Still continue tracing but BLOCK is final
                continue

            # Rule 2: REVIEW takes precedence over WARN and ALLOW
            if action == GuardrailDecision.REVIEW:
                if final_decision in (GuardrailDecision.ALLOW, GuardrailDecision.WARN):
                    final_decision = GuardrailDecision.REVIEW
                    final_reason_codes = policy_result.get("reason_codes", [])
                    final_policy_ids = [policy.policy_id]
                    final_source_refs = policy_result.get("source_references", [])
                    final_detector = policy_result.get("detector_model")
                # If BLOCK already set, keep BLOCK (Rule 1 takes priority)

            # Rule 3: WARN takes precedence over ALLOW
            if action == GuardrailDecision.WARN:
                if final_decision == GuardrailDecision.ALLOW:
                    final_decision = GuardrailDecision.WARN
                    final_reason_codes = policy_result.get("reason_codes", [])
                    final_policy_ids = [policy.policy_id]
                    final_source_refs = policy_result.get("source_references", [])
                    final_detector = policy_result.get("detector_model")

            # Rule 4: ALLOW is the default - only overridden by above

        # Build diagnostics from the decision trace
        diagnostics: dict[str, Any] = {
            "decision_trace": decision_trace,
            "total_policies_evaluated": len(sorted_policies),
            "policies_applicable": len([
                p for p in sorted_policies
                if p.applies_to_stage(stage.value)
            ]),
        }

        # Construct the final decision
        decision = GuardrailDecision(
            decision=final_decision,
            stage=stage,
            reason_codes=final_reason_codes,
            policy_ids=final_policy_ids,
            policy_versions=[p.version for p in self._policies if p.policy_id in final_policy_ids]
            if final_policy_ids
            else None,
            source_references=final_source_refs,
            detector_model=final_detector,
            diagnostics=diagnostics,
        )

        return decision

    def _evaluate_policy_conditions(
        self,
        policy: Policy,
        *,
        stage: GuardrailStage,
        user_input: str | None = None,
        context_items: list[dict[str, Any]] | None = None,
        source_references: list[str] | None = None,
    ) -> dict[str, Any]:
        """Evaluate a policy's conditions against the provided context.

        This method is data-driven: it reads the policy's conditions and
        enforcement_action, then evaluates conditions against the actual
        runtime context. No arbitrary Python code is executed.

        The default implementation evaluates simple pattern-matching conditions.
        Subclasses or integrations can override this for custom condition types.

        Returns:
            dict with "action" (one of allow/warn/block/review) and
            optional "reason_codes", "source_references", "detector_model".
        """

        # Default: if no conditions are defined, the enforcement_action
        # is the default result
        if policy.conditions is None:
            return {
                "action": policy.enforcement_action,
                "reason_codes": [],
                "source_references": source_references or [],
                "detector_model": None,
            }

        # Evaluate conditions against the provided context
        # This is a scaffold: condition types are resolved by name, and
        # each type performs deterministic pattern matching or structural
        # checks against the provided context.

        reason_codes: list[str] = []
        matched_conditions: list[str] = []

        for condition_type, condition_params in policy.conditions.items():
            # Validate condition type is supported
            supported_types = {"keyword", "pattern", "metadata", "trust"}
            if condition_type not in supported_types:
                # Unknown condition type - skip but record
                reason_codes.append(f"unknown_condition_type:{condition_type}")
                continue

            # Evaluate based on condition type
            condition_met = self._evaluate_condition_type(
                condition_type,
                condition_params,
                user_input=user_input,
                context_items=context_items or [],
                source_references=source_references or [],
            )

            if condition_met:
                matched_conditions.append(condition_type)

        # If any conditions matched, use the enforcement_action;
        # otherwise, the default is "allow" (no violation detected)
        if matched_conditions:
            return {
                "action": policy.enforcement_action,
                "reason_codes": matched_conditions,
                "source_references": source_references or [],
                "detector_model": None,
            }
        else:
            # No conditions matched - default to allow
            return {
                "action": "allow",
                "reason_codes": [],
                "source_references": source_references or [],
                "detector_model": None,
            }

    def _evaluate_condition_type(
        self,
        condition_type: str,
        condition_params: dict[str, Any],
        *,
        user_input: str | None = None,
        context_items: list[dict[str, Any]] | None = None,
        source_references: list[str] | None = None,
    ) -> bool:
        """Evaluate a single condition type against the provided context.

        Supported condition types:
            - "keyword": checks if specified keywords/phrases appear in text
            - "pattern": checks if regex patterns match against text
            - "metadata": checks if chunk/item metadata contains specified keys/values
            - "trust": checks source trust state

        Returns:
            True if the condition is met (violation detected), False otherwise.
        """

        if condition_type == "keyword":
            return self._check_keywords(condition_params, user_input, context_items)

        if condition_type == "pattern":
            return self._check_patterns(condition_params, user_input, context_items)

        if condition_type == "metadata":
            return self._check_metadata(condition_params, context_items or [])

        if condition_type == "trust":
            return self._check_trust_condition(condition_params, source_references or [])

        # Unknown condition type - default to not met
        return False

    @staticmethod
    def _check_keywords(
        params: dict[str, Any],
        user_input: str | None,
        context_items: list[dict[str, Any]],
    ) -> bool:
        """Check if specified keywords appear in user input or context."""

        keywords = params.get("keywords", [])
        if not isinstance(keywords, list):
            return False

        # Check user input
        if user_input:
            lower = user_input.lower()
            if any(keyword.lower() in lower for keyword in keywords):
                return True

        # Check context items
        for item in context_items:
            text = item.get("text", "")
            if isinstance(text, str):
                lower = text.lower()
                if any(keyword.lower() in lower for keyword in keywords):
                    return True

        return False

    @staticmethod
    def _check_patterns(
        params: dict[str, Any],
        user_input: str | None,
        context_items: list[dict[str, Any]],
    ) -> bool:
        """Check if specified regex patterns match against text."""

        import re

        patterns = params.get("patterns", [])
        if not isinstance(patterns, list):
            return False

        # Check user input
        if user_input:
            for pattern in patterns:
                try:
                    if re.search(pattern, user_input):
                        return True
                except re.error:
                    # Malformed pattern - skip, don't crash
                    pass

        # Check context items
        for item in context_items:
            text = item.get("text", "")
            if isinstance(text, str):
                for pattern in patterns:
                    try:
                        if re.search(pattern, text):
                            return True
                    except re.error:
                        pass

        return False

    @staticmethod
    def _check_metadata(
        params: dict[str, Any],
        context_items: list[dict[str, Any]],
    ) -> bool:
        """Check if chunk/item metadata contains specified keys/values."""

        required = params.get("required", {})
        if not isinstance(required, dict):
            return False

        for item in context_items:
            metadata = item.get("metadata", {})
            if not isinstance(metadata, dict):
                continue

            # Check all required key-value pairs are present
            all_present = all(
                metadata.get(key) == value
                for key, value in required.items()
            )
            if all_present:
                return True

        return False

    @staticmethod
    def _check_trust_condition(
        params: dict[str, Any],
        source_references: list[str],
    ) -> bool:
        """Check trust state conditions.

        Returns True if the trust condition is violated (i.e., source is
        untrusted/suspicious/blocked).
        """

        # Check for blocked or untrusted sources
        blocked_ids = params.get("blocked_ids", [])
        suspicious_ids = params.get("suspicious_ids", [])

        for ref in source_references:
            if blocked_ids and ref in blocked_ids:
                return True
            if suspicious_ids and ref in suspicious_ids:
                return True

        # Check for specific trust states
        required_state = params.get("required_state")
        if required_state:
            # This is a simplified check - in a full implementation,
            # would check the actual trust state of each source
            pass

        return False
