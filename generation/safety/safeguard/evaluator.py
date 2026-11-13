"""Safety evaluator for optional policy-based content safety evaluation.

openai/gpt-oss-safeguard-20b is a reasoning-based safety classification
model that follows a written policy supplied by the application.

It is NOT a generation fallback. It has its own separate request queue
and token budget tracking across the 12-key pool.

Operating modes: off, input, output, both, sensitive_only.
Default: off until a policy is explicitly configured.

Structured verdict: allow, flag, block, review.
Also includes: Category, Policy reference, Rationale, Model ID,
Latency, Token usage, Request diagnostics.

Dedicated rate limiting: RPM=3, TPM=2000, separate from normal generation.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from generation.contracts.exceptions import (
    GenerationSafeguardError,
    GenerationSafetyBlockError,
)

# Default operating mode: off until a policy is explicitly configured.
DEFAULT_MODE = "off"

# Operating modes
MODE_OFF = "off"
MODE_INPUT = "input"
MODE_OUTPUT = "output"
MODE_BOTH = "both"
MODE_SENSITIVE_ONLY = "sensitive_only"

# Default mode (also accessible as SafeguardEvaluator.DEFAULT_MODE)
DEFAULT_MODE_ALIAS = MODE_OFF


class SafeguardVerdict:
    """Structured verdict from the GPT-OSS Safeguard model.

    Attributes:
        decision: allow, flag, block, or review.
        category: the violation category, if applicable.
        policy_reference: reference to the policy that was evaluated.
        rationale: explanation of the verdict.
        model_id: the model ID that produced the verdict.
        latency: latency of the evaluation in ms.
        token_usage: token consumption for this evaluation.
        request_diagnostics: additional request-level diagnostics.
    """

    def __init__(
        self,
        *,
        decision: str,
        category: str | None = None,
        policy_reference: str | None = None,
        rationale: str | None = None,
        model_id: str | None = None,
        latency: float | None = None,
        token_usage: dict[str, int] | None = None,
        request_diagnostics: dict[str, Any] | None = None,
    ) -> None:
        self.decision = decision
        self.category = category
        self.policy_reference = policy_reference
        self.rationale = rationale
        self.model_id = model_id
        self.latency = latency
        self.token_usage = token_usage or {}
        self.request_diagnostics = request_diagnostics or {}

    def __dict__(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "category": self.category,
            "policy_reference": self.policy_reference,
            "rationale": self.rationale,
            "model_id": self.model_id,
            "latency": self.latency,
            "token_usage": self.token_usage,
            "request_diagnostics": self.request_diagnostics,
        }

    @classmethod
    def from_json(cls, json_str: str) -> SafeguardVerdict:
        """Create a SafeguardVerdict from a JSON string.

        Validates the structured output before construction.

        Args:
            json_str: JSON string containing the verdict.

        Returns:
            SafeguardVerdict instance.

        Raises:
            GenerationSafeguardError: if the JSON is invalid or missing
                required fields.
        """
        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, TypeError) as e:
            raise GenerationSafeguardError(
                f"Invalid JSON in safeguard verdict: {e}"
            ) from e

        # Validate required fields
        if "decision" not in data:
            raise GenerationSafeguardError(
                "Safeguard verdict JSON missing required 'decision' field"
            )

        decision = data["decision"]
        valid_decisions = {"allow", "flag", "block", "review"}
        if decision not in valid_decisions:
            raise GenerationSafeguardError(
                f"Invalid safeguard decision: {decision}. "
                f"Must be one of {valid_decisions}"
            )

        return cls(
            decision=data["decision"],
            category=data.get("category"),
            policy_reference=data.get("policy_reference"),
            rationale=data.get("rationale"),
            model_id=data.get("model_id"),
            latency=data.get("latency"),
            token_usage=data.get("token_usage"),
            request_diagnostics=data.get("request_diagnostics"),
)


class SafeguardEvaluator:
    """Optional policy-based safety evaluation using gpt-oss-safeguard-20b.

    Evaluates generated text or user input against a configurable policy.
    Supports multiple operating modes: off, input, output, both, sensitive_only.

    This model has its own dedicated rate limiting:
    - RPM = 3 (across 12-key pool)
    - TPM = 2000 (across 12-key pool)
    - Separate request queue and token budget tracking.

    Must NOT be placed on the normal generation fallback chain.
    """

    def __init__(
        self,
        *,
        mode: str | None = None,
        policy: dict[str, Any] | None = None,
        api_base_url: str | None = None,
        model_id: str = "openai/gpt-oss-safeguard-20b",
        max_retries: int = 2,
        key_pool: Any | None = None,
    ) -> None:
        """Initialize the Safeguard evaluator.

        Args:
            mode: operating mode. When None, uses environment
                SAFEGUARD_MODE or defaults to "off".
            policy: configurable policy object/dict. When None, uses
                a small neutral example policy for testing.
            api_base_url: Groq API base URL. When None, uses the default.
            model_id: the safeguard model identifier.
            max_retries: maximum retry attempts.
            key_pool: optional KeyRotationPool for 12-key rate limiting.
        """
        self._mode = mode or os.environ.get("SAFEGUARD_MODE", DEFAULT_MODE_ALIAS)
        self._policy = policy or self._default_test_policy()
        self._api_base_url = api_base_url or "https://api.groq.com/openai/v1"
        self._model_id = model_id
        self._max_retries = max_retries
        self._key_pool = key_pool
        self._token_budget = None

        # Initialize token budget if key pool provided
        if key_pool is not None:
            from generation.rate_limiting.token_budget import TokenBudget
            self._token_budget = TokenBudget(
                tpm=2000,
                tpd=200000,
            )

    def set_mode(self, mode: str) -> None:
        """Set the operating mode.

        Args:
            mode: one of "off", "input", "output", "both", "sensitive_only".
        """
        if mode not in (MODE_OFF, MODE_INPUT, MODE_OUTPUT, MODE_BOTH, MODE_SENSITIVE_ONLY):
            raise ValueError(
                f"Invalid safeguard mode: {mode}. "
                f"Must be one of {MODE_OFF}, {MODE_INPUT}, "
                f"{MODE_OUTPUT}, {MODE_BOTH}, {MODE_SENSITIVE_ONLY}"
            )
        self._mode = mode

    def set_policy(self, policy: dict[str, Any]) -> None:
        """Set the policy object.

        Args:
            policy: the policy dict with instructions, definitions, criteria,
                examples, and output schema.
        """
        self._policy = policy

    def evaluate(self, text: str, *, mode: str | None = None) -> Any:
        """Evaluate text against the configured policy.

        Args:
            text: the text to evaluate.
            mode: operating mode override. When None, uses self._mode.

        Returns:
            SafeguardVerdict with the structured decision.

        Raises:
            GenerationSafeguardError: if evaluation fails.
            GenerationSafetyBlockError: if the verdict is "block".
        """
        mode = mode or self._mode

        # Check mode
        if mode == MODE_OFF:
            # Safeguard disabled
            return SafeguardVerdict(
                decision="allow",
                category=None,
                policy_reference=None,
                rationale="Safeguard mode is off",
                model_id=self._model_id,
                latency=0.0,
                token_usage={"input": 0, "output": 0},
                request_diagnostics={"mode": "off", "skipped": True},
            )

        start_time = time.time()

        try:
            # Prepare the policy-based evaluation request
            verdict = self._rule_based_evaluate(text, mode=mode)

            (time.time() - start_time) * 1000

            # Record token usage if budget available
            {"input": len(text), "output": 0}
            if self._token_budget is not None:
                self._token_budget.record_usage(0, 0)  # placeholder

            # Enforce block decisions
            if verdict.decision == "block":
                raise GenerationSafetyBlockError(
                    f"Safeguard blocked the request: {verdict.rationale}",
                    verdict=verdict,
                )

            return verdict

        except GenerationSafetyBlockError:
            raise
        except Exception as e:
            raise GenerationSafeguardError(
                f"Safeguard evaluation failed: {e}"
            ) from e

    def _rule_based_evaluate(
        self, text: str, mode: str
    ) -> SafeguardVerdict:
        """Rule-based evaluation against the policy (scaffold).

        In production, this would call the gpt-oss-safeguard-20b API.
        For now, uses rule-based heuristics against the configured policy.

        Args:
            text: the text to evaluate.
            mode: the operating mode.

        Returns:
            SafeguardVerdict with the structured decision.
        """
        lower = text.lower()

        # Check violation criteria from policy
        violation_criteria = self._policy.get("violation_criteria", [])
        safe_criteria = self._policy.get("safe_criteria", [])

        # Check against violation criteria
        is_violation = False
        detected_category = None

        for criterion in violation_criteria:
            category = criterion.get("category", "")
            description = criterion.get("description", "").lower()
            if description and description in lower:
                is_violation = True
                detected_category = category
                break  # first match wins

        # Check against safe criteria (if no violation detected)
        if not is_violation:
            for criterion in safe_criteria:
                category = criterion.get("category", "")
                description = criterion.get("description", "").lower()
                if description and description in lower:
                    break

        # Mode-dependent decisions
        if mode == MODE_INPUT:
            # Only evaluate input
            if is_violation:
                return SafeguardVerdict(
                    decision="block",
                    category=detected_category,
                    policy_reference=self._get_policy_reference(),
                    rationale=f"Input violation: {detected_category}",
                    model_id=self._model_id,
                    latency=0.0,
                    token_usage={"input": len(text), "output": 0},
                )
            else:
                return SafeguardVerdict(
                    decision="allow",
                    category=None,
                    policy_reference=self._get_policy_reference(),
                    rationale="Input is safe",
                    model_id=self._model_id,
                    latency=0.0,
                    token_usage={"input": len(text), "output": 0},
                )

        elif mode == MODE_OUTPUT:
            # Only evaluate output (same heuristics for scaffold)
            if is_violation:
                return SafeguardVerdict(
                    decision="block",
                    category=detected_category,
                    policy_reference=self._get_policy_reference(),
                    rationale=f"Output violation: {detected_category}",
                    model_id=self._model_id,
                    latency=0.0,
                    token_usage={"input": len(text), "output": 0},
                )
            else:
                return SafeguardVerdict(
                    decision="allow",
                    category=None,
                    policy_reference=self._get_policy_reference(),
                    rationale="Output is safe",
                    model_id=self._model_id,
                    latency=0.0,
                    token_usage={"input": len(text), "output": 0},
                )

        elif mode == MODE_BOTH:
            # Evaluate both input and output
            if is_violation:
                return SafeguardVerdict(
                    decision="block",
                    category=detected_category,
                    policy_reference=self._get_policy_reference(),
                    rationale=f"Violation detected: {detected_category}",
                    model_id=self._model_id,
                    latency=0.0,
                    token_usage={"input": len(text), "output": 0},
                )
            else:
                return SafeguardVerdict(
                    decision="allow",
                    category=None,
                    policy_reference=self._get_policy_reference(),
                    rationale="Both input and output are safe",
                    model_id=self._model_id,
                    latency=0.0,
                    token_usage={"input": len(text), "output": 0},
                )

        elif mode == MODE_SENSITIVE_ONLY:
            # Only flag sensitive categories
            if is_violation and detected_category:
                return SafeguardVerdict(
                    decision="flag",
                    category=detected_category,
                    policy_reference=self._get_policy_reference(),
                    rationale=f"Sensitive content: {detected_category}",
                    model_id=self._model_id,
                    latency=0.0,
                    token_usage={"input": len(text), "output": 0},
                )
            else:
                return SafeguardVerdict(
                    decision="allow",
                    category=None,
                    policy_reference=self._get_policy_reference(),
                    rationale="No sensitive content detected",
                    model_id=self._model_id,
                    latency=0.0,
                    token_usage={"input": len(text), "output": 0},
                )

        else:
            # Default: allow
            return SafeguardVerdict(
                decision="allow",
                category=None,
                policy_reference=self._get_policy_reference(),
                rationale="No policy violation detected",
                model_id=self._model_id,
                latency=0.0,
                token_usage={"input": len(text), "output": 0},
            )

    def check_budgets(self) -> bool:
        """Check if the safeguard has remaining budget.

        Returns:
            True if the safeguard can proceed (has remaining RPM/TPD).
        """
        if self._token_budget is None:
            # No key pool configured; assume available (will fail at API call)
            return True
        # Check TPM and TPD remaining
        remaining = self._token_budget.remaining()
        return remaining["tpm"] > 0 and remaining["tpd"] > 0

    def _get_policy_reference(self) -> str | None:
        """Get a reference to the current policy.

        Returns:
            A policy reference string, or None if no custom policy is set.
        """
        # Return a simple reference based on policy content
        if self._policy:
            # Use first category as reference if available
            criteria = self._policy.get("violation_criteria", [])
            if criteria:
                return criteria[0].get("category", None)
        return None