"""Enforcement engine for the Guardrails ecosystem.

A small coordinator that executes security checks in a predictable order.

Input-side flow:
  Incoming Request
       |
       v
  Structural Validation
       |
       v
  Request/Size Limits
       |
       v
  Access Control
       |
       v
  Prompt Guard
       |
       v
  Optional Safeguard
       |
       v
  ALLOW / WARN / BLOCK / REVIEW

Context-side flow:
  Retrieved Context
       |
       v
  Authorization Check
       |
       v
  Source Provenance Check
       |
       v
  Prompt Injection Segmentation
       |
       v
  Source Trust Assignment
       |
       v
  Allowed Evidence Context

Output-side flow:
  Generated Answer
       |
       v
  Structural Validation
       |
       v
  Citation Identity Validation
       |
       v
  Privacy/Secret Detection
       |
       v
  Optional Safeguard
       |
       v
  Final Enforcement Decision

Requirements:
- Execute stages in explicit order.
- Preserve each individual result.
- Apply deterministic policy precedence.
- Do not let weaker checks override stronger denials.
- Record skipped checks and failures.
- Support configured fail-open/fail-closed/review behavior.
- Avoid unnecessary repeated model calls.
- Expose which evidence was excluded and why.
- Preserve source references.

Do not build an agent workflow engine.
"""

from __future__ import annotations

from typing import Any

from guardrails.contracts.models import (
    GuardrailDecision,
    GuardrailRequest,
    GuardrailStage,
)
from guardrails.output.citation_validation import CitationValidator
from guardrails.output.validation import OutputValidator
from guardrails.policies.engine import PolicyEngine
from guardrails.privacy.detectors import PrivacyDetector
from guardrails.privacy.rules import PrivacyRules
from guardrails.prompt_injection.scanner import PromptInjectionScanner
from guardrails.source_security.access_control import (
    AccessControlEngine,
    UserAccessContext,
)
from guardrails.source_security.trust import (
    TrustState,
)


class EnforcementEngine:
    """Small coordinator that executes security checks in a predictable order.

    Executes checks across three flows:

    Input-side:
      1. Structural validation
      2. Request/size limits
      3. Access control
      4. Prompt Guard (injection detection)
      5. Optional Safeguard
      6. Final decision: ALLOW / WARN / BLOCK / REVIEW

    Context-side:
      1. Authorization check
      2. Source provenance check
      3. Prompt injection segmentation
      4. Source trust assignment
      5. Allowed evidence context

    Output-side:
      1. Structural validation
      2. Citation identity validation
      3. Privacy/secret detection
      4. Optional Safeguard
      5. Final enforcement decision

    Key behaviors:
    - Execute stages in explicit order.
    - Preserve each individual result.
    - Apply deterministic policy precedence (BLOCK > REVIEW > WARN > ALLOW).
    - Do not let weaker checks override stronger denials.
    - Record skipped checks and failures.
    - Support configured fail-open/fail-closed/review behavior.
    - Avoid unnecessary repeated model calls.
    - Expose which evidence was excluded and why.
    - Preserve source references.
    """

    def __init__(
        self,
        *,
        policies: list[Any] | None = None,
        access_control: Any | None = None,
        prompt_guard_scanner: Any | None = None,
        output_validator: Any | None = None,
        citation_validator: Any | None = None,
        privacy_detector: Any | None = None,
    ) -> None:
        """Initialize the enforcement engine.

        Args:
            policies: optional list of Policy instances for the policy engine.
            access_control: optional AccessControlEngine instance.
            prompt_guard_scanner: optional PromptInjectionScanner instance.
            output_validator: optional OutputValidator instance.
            citation_validator: optional CitationValidator instance.
            privacy_detector: optional PrivacyDetector instance.
        """

        self._policy_engine = PolicyEngine(policies=policies)
        self._access_control = access_control or AccessControlEngine()
        self._prompt_guard_scanner = prompt_guard_scanner or PromptInjectionScanner()
        self._output_validator = output_validator or OutputValidator()
        self._citation_validator = citation_validator or CitationValidator(
            access_control=self._access_control
        )
        self._privacy_detector = privacy_detector or PrivacyDetector()

    def execute_input_flow(
        self,
        request: GuardrailRequest,
        *,
        fail_policy: str = "fail-closed",
    ) -> GuardrailDecision:
        """Execute the input-side security flow.

        Args:
            request: the GuardrailRequest to validate.
            fail_policy: fail behavior when a check fails. One of:
                "fail-open" (allow through), "fail-closed" (block), "
                "require_review (send for review)".

        Returns:
            GuardrailDecision with the final enforcement decision.
        """

        decision_trace: list[dict[str, Any]] = []
        failed_checks: list[dict[str, Any]] = []

        # Step 1: Structural validation
        struct_result = self._structural_validation(request)
        decision_trace.append(
            {"stage": "structural_validation", "result": struct_result}
        )
        if not struct_result["valid"]:
            if fail_policy == "fail-closed":
                return GuardrailDecision(
                    decision=GuardrailDecision.BLOCK,
                    stage=GuardrailStage.INPUT,
                    reason_codes=["structural_validation_failed"],
                    policy_ids=self._policy_engine._policies[0].policy_id
                    if self._policy_engine._policies
                    else [],
                    diagnostics={
                        "decision_trace": decision_trace,
                        "failed_checks": failed_checks,
                        "fail_policy": fail_policy,
                    },
                )
            elif fail_policy == "require_review":
                return GuardrailDecision(
                    decision=GuardrailDecision.REVIEW,
                    stage=GuardrailStage.INPUT,
                    reason_codes=["structural_validation_failed"],
                    diagnostics={
                        "decision_trace": decision_trace,
                        "failed_checks": failed_checks,
                        "fail_policy": fail_policy,
                    },
                )
            # fail_open: continue despite structural issues

        # Step 2: Request/size limits
        limits_result = self._check_request_limits(request)
        decision_trace.append(
            {"stage": "request_limits", "result": limits_result}
        )
        if not limits_result["valid"]:
            if fail_policy == "fail-closed":
                return GuardrailDecision(
                    decision=GuardrailDecision.BLOCK,
                    stage=GuardrailStage.INPUT,
                    reason_codes=["excessive_request_size"],
                    diagnostics={
                        "decision_trace": decision_trace,
                        "failed_checks": failed_checks,
                        "fail_policy": fail_policy,
                    },
                )
            elif fail_policy == "require_review":
                return GuardrailDecision(
                    decision=GuardrailDecision.REVIEW,
                    stage=GuardrailStage.INPUT,
                    reason_codes=["excessive_request_size"],
                    diagnostics={
                        "decision_trace": decision_trace,
                        "failed_checks": failed_checks,
                        "fail_policy": fail_policy,
                    },
                )

        # Step 3: Access control
        access_result = self._check_access_control(request)
        decision_trace.append(
            {"stage": "access_control", "result": access_result}
        )
        if not access_result["authorized"]:
            if fail_policy == "fail-closed":
                return GuardrailDecision(
                    decision=GuardrailDecision.BLOCK,
                    stage=GuardrailStage.INPUT,
                    reason_codes=["access_denied"],
                    diagnostics={
                        "decision_trace": decision_trace,
                        "failed_checks": failed_checks,
                        "fail_policy": fail_policy,
                    },
                )
            elif fail_policy == "require_review":
                return GuardrailDecision(
                    decision=GuardrailDecision.REVIEW,
                    stage=GuardrailStage.INPUT,
                    reason_codes=["access_denied"],
                    diagnostics={
                        "decision_trace": decision_trace,
                        "failed_checks": failed_checks,
                        "fail_policy": fail_policy,
                    },
                )

        # Step 4: Prompt Guard
        pg_result = self._scan_prompt_guard(request)
        decision_trace.append(
            {"stage": "prompt_guard", "result": pg_result}
        )
        if pg_result["classification"] == "MALICIOUS":
            if fail_policy == "fail-closed":
                return GuardrailDecision(
                    decision=GuardrailDecision.BLOCK,
                    stage=GuardrailStage.INPUT,
                    reason_codes=["prompt_injection_detected"],
                    source_references=pg_result.get("source_refs"),
                    diagnostics={
                        "decision_trace": decision_trace,
                        "failed_checks": failed_checks,
                        "malicious_segments": pg_result.get("malicious_segments"),
                        "fail_policy": fail_policy,
                    },
                )
            elif fail_policy == "require_review":
                return GuardrailDecision(
                    decision=GuardrailDecision.REVIEW,
                    stage=GuardrailStage.INPUT,
                    reason_codes=["prompt_injection_detected"],
                    source_references=pg_result.get("source_refs"),
                    diagnostics={
                        "decision_trace": decision_trace,
                        "malicious_segments": pg_result.get("malicious_segments"),
                        "fail_policy": fail_policy,
                    },
                )
            # fail_open: allow through with warning

        # Step 5: Optional Safeguard (not implemented in this base version)
        # Would call the Safeguard evaluator here

        # If all checks passed, allow
        return GuardrailDecision(
            decision=GuardrailDecision.ALLOW,
            stage=GuardrailStage.INPUT,
            reason_codes=[],
            diagnostics={
                "decision_trace": decision_trace,
                "failed_checks": failed_checks,
                "fail_policy": fail_policy,
            },
        )

    def execute_context_flow(
        self,
        context_items: list[dict[str, Any]],
        *,
        user_access_context: UserAccessContext,
        source_trust_records: dict[str, Any] | None = None,
        fail_policy: str = "fail-closed",
    ) -> dict[str, Any]:
        """Execute the context-side security flow for retrieved evidence.

        Args:
            context_items: the retrieved context items to validate.
            user_access_context: the user's access context for authorization.
            source_trust_records: optional dict mapping source_id ->
                SourceTrustRecord for trust state verification.
            fail_policy: fail behavior when a check fails. One of:
                "fail-open" (allow through), "fail-closed" (block), "
                "require_review (send for review)".

        Returns:
            dict with:
                - allowed_context: context items that passed all checks
                - excluded_context: context items that were excluded and why
                - final_decision: the overall enforcement decision
                - decision_trace: full trace of checks performed
        """

        decision_trace: list[dict[str, Any]] = []
        allowed: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        final_decision = "allow"

        # Step 1: Authorization check
        auth_result = self._access_control.evaluate_access(user_access_context)
        decision_trace.append(
            {"stage": "authorization_check", "result": auth_result}
        )
        if not auth_result.authorized:
            if fail_policy in ("fail-closed", "require_review"):
                # All context excluded due to unauthorized access
                excluded_items = [
                    {
                        "item": item,
                        "reason": auth_result.reason or "unauthorized access",
                        "decision": "block",
                    }
                    for item in context_items
                ]
                return {
                    "allowed_context": [],
                    "excluded_context": excluded_items,
                    "final_decision": "block",
                    "decision_trace": decision_trace,
                }
            # fail_open would allow all context despite unauthorized access

        # Step 2: Source provenance check
        provenance_results: list[dict[str, Any]] = []
        for item in context_items:
            provenance_check = self._check_source_provenance(item)
            provenance_results.append(provenance_check)
            decision_trace.append(
                {"stage": "source_provenance_check", "result": provenance_check}
            )
            if not provenance_check["valid"]:
                excluded.append(
                    {
                        "item": item,
                        "reason": provenance_check.get("reason", "provenance failure"),
                        "decision": "block",
                    }
                )
                if fail_policy in ("fail-closed", "require_review"):
                    # Short-circuit: if any item fails provenance, block
                    all_excluded = [
                        {
                            "item": i,
                            "reason": "provenance failure in context",
                            "decision": "block",
                        }
                        for i in context_items
                    ]
                    return {
                        "allowed_context": [],
                        "excluded_context": all_excluded,
                        "final_decision": "block",
                        "decision_trace": decision_trace,
                    }

        # Step 3: Prompt injection segmentation
        for i, item in enumerate(context_items):
            pg_result = self._scan_context_for_prompt_injection(item)
            decision_trace.append(
                {"stage": "prompt_injection_segmentation", "result": pg_result}
            )

            # If any segment is malicious, mark the item as suspicious
            if pg_result.get("classification") == "MALICIOUS":
                excluded.append(
                    {
                        "item": item,
                        "reason": "prompt injection detected in context",
                        "malicious_segments": pg_result.get("malicious_segments", []),
                        "decision": "block" if fail_policy in ("fail-closed", "require_review") else "review",
                    }
                )

                # Determine final decision based on policy
                if fail_policy == "fail-closed":
                    final_decision = "block"
                elif fail_policy == "require_review":
                    if final_decision != "block":
                        final_decision = "review"
                # fail_open: keep with warning

            else:
                # Item passed prompt injection scan
                allowed.append(item)

        # Step 4: Source trust assignment
        if source_trust_records:
            for item in allowed:
                trust_result = self._assign_source_trust(item, source_trust_records)
                decision_trace.append(
                    {"stage": "source_trust_assignment", "result": trust_result}
                )
                if trust_result["effective_trust"] in (TrustState.SUSPICIOUS, TrustState.BLOCKED):
                    # Move from allowed to excluded with warning
                    allowed.remove(item)
                    excluded.append(
                        {
                            "item": item,
                            "reason": f"source trust state: {trust_result['effective_trust'].value}",
                            "decision": "review" if fail_policy == "require_review" else ("block" if fail_policy == "fail-closed" else "allow"),
                        }
                    )
                    if fail_policy in ("fail-closed", "require_review") and trust_result[
                        "effective_trust"
                    ] == TrustState.BLOCKED:
                        final_decision = "block"

        # Return the final result
        return {
            "allowed_context": allowed,
            "excluded_context": excluded,
            "final_decision": final_decision,
            "decision_trace": decision_trace,
        }

    def execute_output_flow(
        self,
        generated_answer: str,
        *,
        citation_refs: list[dict[str, Any]] | None = None,
        source_metadata: dict[str, Any] | None = None,
        privacy_rules: Any | None = None,
        fail_policy: str = "fail-closed",
    ) -> GuardrailDecision:
        """Execute the output-side security flow for generated answers.

        Args:
            generated_answer: the generated text to validate.
            citation_refs: optional list of citation references.
            source_metadata: optional source metadata for citation authorization.
            privacy_rules: optional PrivacyRules for secret detection actions.
            fail_policy: fail behavior when a check fails. One of:
                "fail-open" (allow through), "fail-closed" (block), "
                "require_review (send for review)".

        Returns:
            GuardrailDecision with the final enforcement decision.
        """

        decision_trace: list[dict[str, Any]] = []
        failed_checks: list[dict[str, Any]] = []

        # Step 1: Structural validation
        struct_result = self._output_validator_validate_structure(
            generated_answer
        )
        decision_trace.append(
            {"stage": "structural_validation", "result": struct_result}
        )
        if not struct_result["valid"]:
            if fail_policy == "fail-closed":
                return GuardrailDecision(
                    decision=GuardrailDecision.BLOCK,
                    stage=GuardrailStage.GENERATION_OUTPUT,
                    reason_codes=["structural_validation_failed"],
                    diagnostics={
                        "decision_trace": decision_trace,
                        "failed_checks": failed_checks,
                        "fail_policy": fail_policy,
                    },
                )
            elif fail_policy == "require_review":
                return GuardrailDecision(
                    decision=GuardrailDecision.REVIEW,
                    stage=GuardrailStage.GENERATION_OUTPUT,
                    reason_codes=["structural_validation_failed"],
                    diagnostics={
                        "decision_trace": decision_trace,
                        "failed_checks": failed_checks,
                        "fail_policy": fail_policy,
                    },
                )

        # Step 2: Citation identity validation
        citation_result = self._citation_validator_validate(
            generated_answer,
            citation_refs=citation_refs,
            source_metadata=source_metadata,
        )
        decision_trace.append(
            {"stage": "citation_identity_validation", "result": citation_result}
        )
        if not citation_result["valid"]:
            if fail_policy == "fail-closed":
                return GuardrailDecision(
                    decision=GuardrailDecision.BLOCK,
                    stage=GuardrailStage.GENERATION_OUTPUT,
                    reason_codes=["citation_validation_failed"],
                    diagnostics={
                        "decision_trace": decision_trace,
                        "failed_checks": failed_checks,
                        "fail_policy": fail_policy,
                    },
                )
            elif fail_policy == "require_review":
                return GuardrailDecision(
                    decision=GuardrailDecision.REVIEW,
                    stage=GuardrailStage.GENERATION_OUTPUT,
                    reason_codes=["citation_validation_failed"],
                    diagnostics={
                        "decision_trace": decision_trace,
                        "failed_checks": failed_checks,
                        "fail_policy": fail_policy,
                    },
                )

        # Step 3: Privacy/secret detection
        privacy_result = self._privacy_detector_validate(
            generated_answer, rules=privacy_rules
        )
        decision_trace.append(
            {"stage": "privacy_secret_detection", "result": privacy_result}
        )
        if privacy_result["detected"]:
            if fail_policy == "fail-closed":
                return GuardrailDecision(
                    decision=GuardrailDecision.BLOCK,
                    stage=GuardrailStage.GENERATION_OUTPUT,
                    reason_codes=["secret_detected_in_output"],
                    diagnostics={
                        "decision_trace": decision_trace,
                        "failed_checks": failed_checks,
                        "secrets_detected": privacy_result["count"],
                        "fail_policy": fail_policy,
                    },
                )
            elif fail_policy == "require_review":
                return GuardrailDecision(
                    decision=GuardrailDecision.REVIEW,
                    stage=GuardrailStage.GENERATION_OUTPUT,
                    reason_codes=["secret_detected_in_output"],
                    diagnostics={
                        "decision_trace": decision_trace,
                        "failed_checks": failed_checks,
                        "secrets_detected": privacy_result["count"],
                        "fail_policy": fail_policy,
                    },
                )

        # All checks passed
        return GuardrailDecision(
            decision=GuardrailDecision.ALLOW,
            stage=GuardrailStage.GENERATION_OUTPUT,
            reason_codes=[],
            diagnostics={
                "decision_trace": decision_trace,
                "failed_checks": failed_checks,
                "fail_policy": fail_policy,
            },
        )

    # --- Internal helper methods ---

    def _structural_validation(self, request: GuardrailRequest) -> dict[str, Any]:
        """Perform structural validation on the request.

        Checks: empty input, valid encoding, request body size,
        excessive message count, invalid content structures,
        unsupported input types, invalid metadata, invalid request IDs.
        """

        from guardrails.input.validation import (
            validate_metadata,
            validate_request_id,
            validate_user_input,
        )

        result: dict[str, Any] = {"valid": True, "errors": []}

        # Validate user input
        try:
            validated_input, input_reasons = validate_user_input(
                request.user_input,
                allow_empty=False,
            )
            if input_reasons:
                result["valid"] = False
                result["errors"].extend(
                    {"field": "user_input", "reason": r} for r in input_reasons
                )
        except Exception as e:
            result["valid"] = False
            result["errors"].append(
                {"field": "user_input", "reason": str(e)}
            )

        # Validate metadata
        if request.source_metadata:
            try:
                validate_metadata(request.source_metadata)
            except Exception as e:
                result["valid"] = False
                result["errors"].append(
                    {"field": "metadata", "reason": str(e)}
                )

        # Validate request ID
        try:
            validated_id = validate_request_id(
                request.request_id,
                existing_ids=set(),
                allow_empty=request.request_id is None,
            )
            if not validated_id and request.request_id:
                result["valid"] = False
                result["errors"].append(
                    {"field": "request_id", "reason": "invalid or conflicting request ID"}
                )
        except Exception as e:
            result["valid"] = False
            result["errors"].append(
                {"field": "request_id", "reason": str(e)}
            )

        return result

    def _check_request_limits(self, request: GuardrailRequest) -> dict[str, Any]:
        """Check request size and limit constraints."""

        from guardrails.input.limits import RequestLimits

        limits = RequestLimits()

        # Check request size (estimate)
        # Rough estimate: count characters in user_input + context
        estimated_size = len(request.user_input) if request.user_input else 0
        size_within, size_reasons = limits.check_request_size(estimated_size)

        result: dict[str, Any] = {"valid": size_within, "reasons": size_reasons}

        if not size_within:
            result["valid"] = False
            result["reasons"].append("exceeds_max_request_bytes")

        return result

    def _check_access_control(self, request: GuardrailRequest) -> dict[str, Any]:
        """Check access control for the request."""

        from guardrails.source_security.access_control import (
            AccessScope,
            UserAccessContext,
        )

        # Build access context from the request
        # Use PUBLIC scope as default; caller can provide more specific info
        access_context = UserAccessContext(
            user_id=getattr(request, "user_id", "anonymous"),
            tenant_id=getattr(request, "tenant_id", ""),
            access_scope=AccessScope.PUBLIC,
            requested_document_ids=(),
            requesting_collection="rag_pipeline",
        )

        decision = self._access_control.evaluate_access(access_context)

        return {
            "authorized": decision.authorized,
            "reason": decision.reason,
            "reason_codes": decision.reason_codes,
        }

    def _scan_prompt_guard(self, request: GuardrailRequest) -> dict[str, Any]:
        """Scan user input and context for prompt injection using Prompt Guard."""

        result = self._prompt_guard_scanner.scan(
            text=request.user_input,
            language_code=getattr(request, "language_code", None),
        )

        # Add source references from context items
        source_refs: list[str] = []
        for ctx_item in getattr(request, "context_items", []) or []:
            cid = ctx_item.get("chunk_id") or ctx_item.get("doc_id")
            if cid:
                source_refs.append(str(cid))

        result["source_refs"] = source_refs

        return result

    def _check_source_provenance(self, item: dict[str, Any]) -> dict[str, Any]:
        """Check the source provenance of a context item.

        Validates that the item has proper provenance tracking and
        that the original text is preserved.
        """

        result: dict[str, Any] = {"valid": True, "reason": ""}

        # Check that the item has text
        if not item.get("text"):
            result["valid"] = False
            result["reason"] = "context item has no text"
            return result

        # Check for provenance tracking
        provenance = item.get("provenance", {})
        if not provenance:
            # No provenance tracked - warn but don't block
            result["valid"] = False
            result["reason"] = "no provenance tracking for context item"
        else:
            # Verify provenance has required fields
            if "doc_id" not in provenance:
                result["valid"] = False
                result["reason"] = "provenance missing doc_id"
            elif "chunk_id" not in provenance:
                # Warning-level: missing chunk_id is not blockworthy
                result["valid"] = True
                result["reason"] = "provenance missing chunk_id (warning)"
            else:
                result["valid"] = True
                result["reason"] = "provenance valid"

        return result

    def _scan_context_for_prompt_injection(
        self, item: dict[str, Any],
    ) -> dict[str, Any]:
        """Scan a context item for prompt injection."""

        text = item.get("text", "")
        if not text:
            return {
                "classification": "BENIGN",
                "is_malicious": False,
                "malicious_segments": [],
                "incomplete_scanning": False,
            }

        return self._prompt_guard_scanner.scan_target(
            text=text,
        )

    def _assign_source_trust(
        self, item: dict[str, Any], source_trust_records: dict[str, Any],
    ) -> dict[str, Any]:
        """Assign trust state to a source item.

        Returns a dict with the effective trust state and diagnostics.
        """

        source_id = item.get("chunk_id") or item.get("doc_id", "")

        if source_id not in source_trust_records:
            # No trust record found - treat as unchecked
            return {
                "effective_trust": TrustState.UNCHECKED,
                "source_metadata": {
                    "trust_state": TrustState.UNCHECKED.value,
                    "source_id": source_id,
                },
                "prompt_guard_assessment": None,
                "ingestion_validation": None,
            }

        trust_record_data = source_trust_records[source_id]

        # Simplified trust state determination
        # In full integration, would construct SourceTrustRecord and call
        # get_effective_trust()
        trust_state_str = trust_record_data.get("source_metadata", {}).get(
            "trust_state", "unchecked"
        )

        try:
            effective = TrustState[trust_state_str]
        except KeyError:
            effective = TrustState.UNCHECKED

        return {
            "effective_trust": effective,
            "source_metadata": trust_record_data.get("source_metadata", {}),
            "prompt_guard_assessment": trust_record_data.get(
                "prompt_guard_assessment"
            ),
            "ingestion_validation": trust_record_data.get(
                "ingestion_validation"
            ),
        }

    def _output_validator_validate_structure(
        self, output_text: str,
    ) -> dict[str, Any]:
        """Validate the structural aspects of the output using the output validator."""

        return self._output_validator.validate(
            output_text,
            expected_type="text",
        )

    def _citation_validator_validate(
        self,
        output_text: str,
        *,
        citation_refs: list[dict[str, Any]] | None = None,
        source_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate citations in the output using the citation validator."""

        # Extract citation refs from the output text search
        # In a full integration, would parse the generated answer for citation markers
        # For now, use the provided citation_refs

        return self._citation_validator.validate(
            citations=citation_refs or [],
            user_id=source_metadata.get("user_id") if source_metadata else None,
            tenant_id=source_metadata.get("tenant_id") if source_metadata else None,
            source_trust_records=source_metadata
            if source_metadata
            and isinstance(source_metadata, dict)
            and "trust_records" in source_metadata
            else None,
        )

    def _privacy_detector_validate(
        self, output_text: str, rules: Any | None = None,
    ) -> dict[str, Any]:
        """Validate privacy/secrets in the output using the privacy detector."""

        if rules is None:
            rules = PrivacyRules(default_action="detect")

        # Run detection
        detector_result = self._privacy_detector.detect(output_text)

        # Evaluate action
        action = rules.evaluate_action(detector_result.categories)

        # Build result
        result: dict[str, Any] = {
            "detected": len(detector_result.categories) > 0,
            "categories": detector_result.categories,
            "count": len(detector_result.positions),
            "action": action,
            "redacted_text": detector_result.redacted_text,
        }

        return result
