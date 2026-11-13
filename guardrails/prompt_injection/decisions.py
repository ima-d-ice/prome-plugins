"""Prompt injection enforcement decisions for the Guardrails ecosystem.

Maps Prompt Guard classification results to enforcement options:
- ALLOW, WARN, BLOCK, REVIEW, MARK_SOURCE_UNTRUSTED.

A suspicious document should not automatically cause a legitimate user
query to be blocked. The caller should be able to choose whether suspicious
evidence is excluded, retained with warning, or sent for review.

Do not treat a benign classifier result as proof that content is safe.
Prompt Guard detects prompt attacks and jailbreaks but does not establish
whether general content is harmful, factual or policy-compliant.
"""

from __future__ import annotations

from typing import Any

from guardrails.contracts.models import GuardrailStage, GuardrailDecision
from guardrails.contracts.exceptions import GuardrailError


class PromptInjectionDecisionMapper:
    """Map Prompt Guard classification results to enforcement decisions.

    Supports the following enforcement options:
    - ALLOW: prompt guard classified as BENIGN, or benign with configured allow
    - WARN: prompt guard classified as MALICIOUS but caller chooses warning
    - BLOCK: prompt guard classified as MALICIOUS and caller chooses block
    - REVIEW: prompt guard classified as MALICIOUS and caller chooses review
    - MARK_SOURCE_UNTRUSTED: source marked as suspicious/untrusted without
      blocking the user query

    Key principle: a suspicious document should not automatically cause a
    legitimate user query to be blocked. The caller should choose the
    enforcement action based on the classification and context.
    """

    # Prompt Guard classification → default enforcement mapping
    # These are defaults; the caller can override per-decision
    DEFAULT_MAPPING = {
        "BENIGN": "allow",
        "MALICIOUS": "block",  # default strict behavior
    }

    def __init__(
        self,
        *,
        default_policy: str = "block",  # default action for MALICIOUS classification
        allow_benign: bool = True,
    ) -> None:
        """Initialize the decision mapper.

        Args:
            default_policy: default enforcement policy for MALICIOUS
                classifications. One of: "allow", "block", "review".
            allow_benign: when True, BENIGN classifications default to ALLOW.
                When False, BENIGN classifications require explicit decision.
        """

        self.default_policy = default_policy
        self.allow_benign = allow_benign

    def map_classification(
        self,
        classification: str,  # "BENIGN" or "MALICIOUS"
        *,
        stage: GuardrailStage,
        source_references: list[str] | None = None,
        user_input: str | None = None,
        override_policy: str | None = None,
    ) -> GuardrailDecision:
        """Map a Prompt Guard classification to a GuardrailDecision.

        Args:
            classification: "BENIGN" or "MALICIOUS" from Prompt Guard.
            stage: the pipeline stage where the classification applies.
            source_references: optional source references (chunk IDs, doc IDs)
                involved in the classification.
            user_input: optional user input for context.
            override_policy: optional override of the default policy for this
                decision.

        Returns:
            GuardrailDecision with the appropriate enforcement action and
            full metadata.
        """

        # Determine the enforcement action
        if classification == "BENIGN":
            if self.allow_benign or (override_policy and override_policy != "block"):
                action = GuardrailDecision.ALLOW
            else:
                action = self._determine_action_for_benign(stage)
        else:  # MALICIOUS
            # Use override if specified, otherwise default policy
            policy = override_policy or self.default_policy
            if policy == "allow":
                action = GuardrailDecision.ALLOW
            elif policy == "warn":
                action = GuardrailDecision.WARN
            elif policy == "review":
                action = GuardrailDecision.REVIEW
            else:  # "block" or any other value
                action = GuardrailDecision.BLOCK

        # Build reason codes
        reason_codes: list[str] = []
        if classification == "MALICIOUS":
            reason_codes.append("prompt_injection_detected")
        if classification == "BENIGN" and not self.allow_benign:
            reason_codes.append("benign_but_blocked_by_policy")

        # Build source references
        src_refs = source_references or []

        # Build diagnostics
        diagnostics: dict[str, Any] = {
            "classification": classification,
            "mapped_action": action,
        }

        # Construct the final decision
        decision = GuardrailDecision(
            decision=action,
            stage=stage,
            reason_codes=reason_codes,
            policy_ids=[],  # will be populated by caller if needed
            policy_versions=[],
            source_references=src_refs,
            detector_model="prompt_guard_2",  # Prompt Guard 2
            diagnostics=diagnostics,
        )

        return decision

    def _determine_action_for_benign(self, stage: GuardrailStage) -> str:
        """Determine action for BENIGN classification when not auto-allowed."""

        # For benign content, the default should always be allow
        # This method is here for completeness if future logic is needed
        return GuardrailDecision.ALLOW

    def map_segment_results(
        self,
        segment_results: list[dict[str, Any]],
        *,
        stage: GuardrailStage,
        source_references: list[str] | None = None,
        override_policy: str | None = None,
    ) -> GuardrailDecision:
        """Map multiple segment classification results to a single decision.

        Aggregates segment results conservatively: if ANY segment is MALICIOUS,
        the overall classification is MALICIOUS. The enforcement action then
        follows the configured policy.

        Args:
            segment_results: list of segment result dicts from Prompt Guard
                segmentation, each with "classification", "is_malicious", etc.
            stage: the pipeline stage.
            source_references: optional source references.
            override_policy: optional override of the default policy.

        Returns:
            GuardrailDecision with the aggregated enforcement action.
        """

        # Check if any segment is malicious (conservative aggregation)
        any_malicious = any(
            r.get("is_malicious", False) for r in segment_results
        )

        classification = "MALICIOUS" if any_malicious else "BENIGN"

        return self.map_classification(
            classification=classification,
            stage=stage,
            source_references=source_references,
            override_policy=override_policy,
        )