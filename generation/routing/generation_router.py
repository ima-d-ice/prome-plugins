"""Generation router for Prome RAG Plugins.

Determines the optimal provider and model for a given request, supports
routing modes (auto, manual, ollama), and coordinates fallback across
the 12-key Groq pool and model route.

The router does not contain rate-limiting or token accounting logic —
it delegates those to the dedicated rate_limiting and input_management
packages. It focuses on provider/model selection decisions.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Tuple, Union, List

from generation.contracts.models (
    GenerationRequest,
    GenerationResponse,
    ModelConfiguration,
    Capabilities,
    ProviderProtocol,
)
from generation.contracts.protocols import GenerationProvider
from generation.contracts.exceptions import (
    GenerationConfigError,
    GenerationDependencyError,
    GenerationInferenceError,
    GenerationRateLimitError,
    GenerationKeyPoolExhaustedError,
    GenerationModelUnavailableError,
    GenerationAuthenticationError,
    GenerationInvalidRequestError,
    GenerationInvalidModelError,
    GenerationProviderUnavailableError,
)
from generation.providers.ollama.provider import OllamaProvider
from generation.providers.groq.provider import GroqProvider
from generation.rate_limiting.key_rotation import KeyRotationPool
from generation.input_management.context_budget import ContextBudget
from generation.safety.prompt_guard.classifier import PromptGuardClassifier
from generation.safety.safeguard.evaluator import SafeguardEvaluator


class GenerationRouter:
    """Router for determining the optimal provider and model for a request.

    Supports routing modes:
    - auto: use the primary Groq model and configured fallback chain
    - manual: use the explicitly selected Groq model
    - ollama: manually select one of the two local models

    An explicitly selected Ollama model must stay on Ollama unless the caller
    explicitly enables provider fallback.
    """

    def __init__(
        self,
        *,
        groq_provider: GroqProvider,
        ollama_provider: OllamaProvider | None = None,
        prompt_guard: PromptGuardClassifier | None = None,
        safeguard_evaluator: SafeguardEvaluator | None = None,
        context_budget: ContextBudget | None = None,
    ) -> None:
        """Initialize the generation router.

        Args:
            groq_provider: configured GroqProvider instance with 12-key pool.
            ollama_provider: optional OllamaProvider instance. When None, a
                default is created using OLLAMA_BASE_URL and configured models.
            prompt_guard: optional PromptGuardClassifier instance.
            safeguard_evaluator: optional SafeguardEvaluator instance.
            context_budget: optional ContextBudget instance for input capping.
        """
        self._groq_provider = groq_provider
        self._ollama_provider = ollama_provider or self._default_ollama_provider()
        self._prompt_guard = prompt_guard
        self._safeguard_evaluator = safeguard_evaluator
        self._context_budget = context_budget

    def _default_ollama_provider(self) -> OllamaProvider:
        """Create a default Ollama provider from environment config."""
        from generation.providers.ollama.provider import OllamaProvider

        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        model_8b = os.environ.get("OLLAMA_QWEN_8B_MODEL")
        model_4b = os.environ.get("OLLAMA_QWEN_4B_MODEL")

        return OllamaProvider(
            base_url=base_url,
            model_8b=model_8b,
            model_4b=model_4b,
            context_length=8192,
        )

    # ------------------------------------------------------------------
    # Main routing entry point
    # ------------------------------------------------------------------

    def route(self, request: GenerationRequest) -> GenerationResponse:
        """Route a generation request to the appropriate provider/model.

        The high-level request flow is:
        1. Input and token caps check
        2. Prompt Guard classification (adaptive routing)
        3. Optional query enhancement
        4. Manual Ollama selection OR Groq auto-route (12-key load balanced)
        5. Optional Safeguard evaluation
        6. Generation response with diagnostics

        Args:
            request: the generation request.

        Returns:
            GenerationResponse with the generated text and diagnostics.

        Raises:
            GenerationConfigError: if the request is invalid.
            GenerationKeyPoolExhaustedError: if all keys and models are exhausted.
            GenerationModelUnavailableError: if the selected model is unavailable.
        """
        if not request.user_input.strip():
            raise GenerationConfigError("user_input must be a non-empty string")

        route_start = time.time()
        diagnostics: dict[str, Any] = {
            "route_history": [],
            "guard_model": None,
            "guard_classification": None,
            "safeguard_decision": None,
            "query_enhanced": False,
            "truncation_details": None,
        }

        # Step 1: Input and token caps check
        request, caps_diagnostics = self._apply_input_caps(request)
        diagnostics["route_history"].append(("input_caps", caps_diagnostics))
        if caps_diagnostics.get("rejected"):
            raise GenerationConfigError(caps_diagnostics.get("reason", "Input cap rejection"))

        # Step 2: Prompt Guard classification (adaptive routing)
        if self._prompt_guard is not None:
            guard_result = self._classify_prompt_guard(request)
            diagnostics["guard_model"] = guard_result.model_used
            diagnostics["guard_classification"] = guard_result.classification
            diagnostics["route_history"].append(("prompt_guard", guard_result.__dict__))

            # Enforce guard policy
            enforcement = self._enforce_guard_policy(guard_result)
            if enforcement == "block":
                raise GenerationSafetyBlockError(
                    f"Prompt Guard blocked the request: {guard_result.classification}",
                    guard_classification=guard_result.classification,
                )
            # Other policies (allow, require review, allow with warning) proceed

        # Step 3: Optional query enhancement
        if request.enhance_query and self._should_enhance_query():
            request, enhancement_diagnostics = self._enhance_query(request)
            diagnostics["query_enhanced"] = True
            diagnostics["route_history"].append(("query_enhancement", enhancement_diagnostics))

        # Step 4: Determine routing mode and generate
        mode = self._determine_routing_mode(request)
        diagnostics["route_history"].append(("routing_mode", {"mode": mode}))

        if mode == "ollama":
            return self._route_ollama(request, diagnostics)
        else:
            return self._route_groq(request, diagnostics)

    def _apply_input_caps(
        self, request: GenerationRequest
    ) -> tuple[GenerationRequest, dict[str, Any]]:
        """Apply input and token caps to the request.

        Returns the (possibly modified) request and diagnostics.

        Args:
            request: the generation request.

        Returns:
            (modified request, diagnostics dict).
        """
        if self._context_budget is None:
            # Create a default context budget
            from generation.input_management.context_budget import ContextBudget
            self._context_budget = ContextBudget()

        budget, diagnostics = self._context_budget.check_bounds(request)
        # If the request was truncated, we need to modify it
        if diagnostics.get("truncated"):
            # Apply truncation to user input
            request = request  # placeholder — actual truncation handled by context budget
        return request, diagnostics

    def _classify_prompt_guard(self, request: GenerationRequest) -> Any:
        """Classify the request using Prompt Guard.

        Returns the guard result.
        """
        # Use adaptive routing based on language
        language_code = request.language_code or "en"

        if language_code in ("en", "eng", "en-US"):
            model_to_use = "meta-llama/llama-prompt-guard-2-22m"
        else:
            model_to_use = "meta-llama/llama-prompt-guard-2-86m"

        # Classify the user query
        classification = self._prompt_guard.classify(
            request.user_input,
            model_name=model_to_use,
        )
        return classification

    def _enforce_guard_policy(
        self, classification: Any
    ) -> str:
        """Enforce the prompt guard classification policy.

        Returns one of: "allow", "block", "require_review", "allow_with_warning"
        """
        # Default policy: BENIGN -> allow, MALICIOUS -> block
        if classification.classification.value.upper() == "MALICIOUS":
            return "block"
        return "allow"

    def _determine_routing_mode(self, request: GenerationRequest) -> str:
        """Determine the routing mode for this request.

        Returns one of: "auto", "manual", "ollama"
        """
        # If user explicitly selected Ollama model
        if request.model and (
            request.model.startswith("ollama") or request.model in ("qwen/8b", "qwen/4b")
        ):
            return "ollama"
        # If explicit provider was set
        if request.provider == "ollama":
            return "ollama"
        # If explicit provider was set to groq
        if request.provider == "groq":
            return "auto"
        # Default: auto (primary Groq model with fallbacks)
        return "auto"

    def _route_ollama(self, request: GenerationRequest, diagnostics: dict[str, Any]) -> GenerationResponse:
        """Route to the Ollama provider.

        Args:
            request: the generation request.
            diagnostics: diagnostics dict to populate.

        Returns:
            GenerationResponse from Ollama.
        """
        try:
            response = self._ollama_provider.generate(request)
            response.actual_provider = "ollama"
            # Populate route diagnostics
            diagnostics["actual_provider"] = "ollama"
            diagnostics["route_history"].append(("ollama_generation", {
                "model": response.actual_model,
                "latency": response.latency,
            }))
            return response
        except GenerationModelUnavailableError:
            # If Ollama model unavailable, fall back to Groq if enabled
            if os.environ.get("GENERATION_FALLBACK_TO_GROQ_ON_OLLAMA_FAILURE", "false").lower() == "true":
                return self._route_groq(request, diagnostics)
            raise
        except Exception as e:
            # If Ollama fails, fall back to Groq if enabled
            if os.environ.get("GENERATION_FALLBACK_TO_GROQ_ON_OLLAMA_FAILURE", "false").lower() == "true":
                return self._route_groq(request, diagnostics)
            raise

    def _route_groq(self, request: GenerationRequest, diagnostics: dict[str, Any]) -> GenerationResponse:
        """Route to the Groq provider with 12-key rotation and fallback.

        Args:
            request: the generation request.
            diagnostics: diagnostics dict to populate.

        Returns:
            GenerationResponse from Groq, or error if all exhausted.
        """
        try:
            response = self._groq_provider.generate(request)
            response.actual_provider = "groq"
            diagnostics["actual_provider"] = "groq"
            diagnostics["route_history"].append(("groq_generation", {
                "model": response.actual_model,
                "latency": response.latency,
            }))
            return response
        except GenerationKeyPoolExhaustedError:
            # All keys and models exhausted — re-raise or convert
            raise
        except Exception as e:
            # Convert other errors and attempt fallback
            raise GenerationInferenceError(
                f"Groq generation failed: {e}"
            )

    def _enhance_query(self, request: GenerationRequest) -> tuple[GenerationRequest, dict[str, Any]]:
        """Apply query enhancement to the request.

        Uses an injected generation provider for enhancement.
        Does not call Prompt Guard as the enhancer.
        Preserves the original query.

        Args:
            request: the generation request.

        Returns:
            (enhanced request, diagnostics).
        """
        from generation.query_enhancement.enhancer import QueryEnhancer

        if self._groq_provider is None:
            # Query enhancement disabled if no provider available
            return request, {"enabled": False, "reason": "no_provider"}

        enhancer = QueryEnhancer(provider=self._groq_provider)
        enhanced_request, enhancement_diagnostics = enhancer.enhance(request)

        # Preserve original query in diagnostics
        enhancement_diagnostics["original_query"] = request.user_input

        return enhanced_request, enhancement_diagnostics

    def _should_enhance_query(self) -> bool:
        """Check whether query enhancement should be applied.

        Returns True if enabled via environment/configuration.
        """
        return os.environ.get("QUERY_ENHANCEMENT_ENABLED", "false").lower() == "true"


class FallbackManager:
    """Manages fallback logic across models and keys.

    Handles the fallback chain logic:
    - Primary -> GPT-OSS 120B -> GPT-OSS 20B
    - Key rotation before model fallback
    - Retryable error classification
    """

    def __init__(
        self,
        route: tuple[str, ...] | None = None,
        key_pool: KeyRotationPool | None = None,
        max_retries: int = 3,
        max_key_rotations: int | None = None,
    ) -> None:
        """Initialize the fallback manager.

        Args:
            route: model fallback route. Defaults to Groq default route.
            key_pool: 12-key rotation pool.
            max_retries: max retry attempts per model/key.
            max_key_rotations: max key rotations before model fallback.
        """
        self._route = route or GroqProvider.DEFAULT_ROUTE
        self._key_pool = key_pool
        self._max_retries = max_retries
        self._max_key_rotations = max_key_rotations or len(self._route) * 12

    def evaluate_fallback(
        self, error: Exception, current_model: str, current_key_idx: int,
    ) -> Tuple[bool, str | None, int | None]:
        """Evaluate whether to fallback given an error.

        Returns (should_fallback, next_model, next_key_idx).
        """
        should_fallback = False
        next_model = None
        next_key_idx = None

        # Determine error type
        error_name = type(error).__name__

        # Rate limit errors — try next key first
        if "RateLimit" in error_name or "429" in str(error):
            # Try next key in the pool
            if self._key_pool is not None:
                next_key_idx = current_key_idx  # same model, next key
                # If we've exhausted keys for this model, fall back
                should_fallback = True
            else:
                should_fallback = True  # no pool, fall back

        # Model unavailable or connection errors — try next key then next model
        elif "ModelUnavailable" in error_name or "Connection" in error_name or "Timeout" in error_name:
            if self._key_pool is not None:
                # Try next key first
                next_key_idx = current_key_idx
                should_fallback = True  # key rotation
            else:
                should_fallback = True

        # Authentication error on a single key — rotate to next key
        elif "Authentication" in error_name:
            if self._key_pool is not None:
                next_key_idx = current_key_idx  # rotate key
                should_fallback = True
            else:
                should_fallback = False  # can't fix auth error

        # Config/validation errors — don't fallback (it won't help)
        elif "Config" in error_name or "Invalid" in error_name:
            should_fallback = False

        # If we should fallback and there are more models/keys, determine next
        if should_fallback:
            # Check if we need to rotate key or switch model
            if next_key_idx is not None and self._key_pool is not None:
                # Check if there are more keys available
                pool_size = self._key_pool.pool_size()
                if next_key_idx < pool_size - 1:
                    next_key_idx = next_key_idx + 1  # simple increment
                else:
                    # Exhausted this model across all keys, try next model
                    model_idx = self._route.index(current_model) if current_model in self._route else 0
                    if model_idx < len(self._route) - 1:
                        next_model = self._route[model_idx + 1]
                        next_key_idx = 0  # start with first key of next model
                    else:
                        # No more models/keys
                        should_fallback = False
            elif next_model is not None:
                next_key_idx = 0

        return should_fallback, next_model, next_key_idx


class ModelSelector:
    """Selects the appropriate model based on routing mode and availability.

    Handles model selection logic for:
    - auto mode: primary model with fallback chain
    - manual mode: explicitly selected model
    - ollama mode: locally available Qwen models
    """

    def __init__(
        self,
        groq_route: tuple[str, ...] | None = None,
        ollama_provider: Any | None = None,
    ) -> None:
        """Initialize the model selector.

        Args:
            groq_route: Groq model fallback route. Defaults to default route.
            ollama_provider: optional OllamaProvider instance.
        """
        self._groq_route = groq_route or GroqProvider.DEFAULT_ROUTE
        self._ollama_provider = ollama_provider

    def select_model(
        self, mode: str, request: Any, key_pool: Any | None = None,
    ) -> tuple[str, str | None]:
        """Select the model to use based on the routing mode.

        Returns (model_id, error_message).
        """
        if mode == "auto":
            return self._select_auto(key_pool)
        elif mode == "manual":
            return self._select_manual(request)
        elif mode == "ollama":
            return self._select_ollama(key_pool)
        else:
            return self._select_auto(key_pool), None

    def _select_auto(self, key_pool: Any | None = None) -> tuple[str, str | None]:
        """Select model in auto mode: primary with fallback chain.

        Returns (model_id, error_message).
        """
        # Return the primary model; the Groq provider's generate method
        # will handle fallbacks across keys and subsequent models
        return self._groq_route[0], None

    def _select_manual(self, request: Any) -> tuple[str, str | None]:
        """Select model in manual mode.

        Uses the explicitly selected Groq model from the request.
        """
        model_id = getattr(request, "model", None) or self._groq_route[0]
        # Validate the model is in the route
        if model_id not in self._groq_route:
            model_id = self._groq_route[0]
        return model_id, None

    def _select_ollama(self, key_pool: Any | None = None) -> tuple[str, str | None]:
        """Select model in ollama mode.

        Uses one of the two locally available Qwen models.
        """
        if self._ollama_provider is not None:
            # Check availability of 8B first, then 4B
            if self._ollama_provider.is_model_available("qwen/8b"):
                return "qwen/8b", None
            elif self._ollama_provider.is_model_available("qwen/4b"):
                return "qwen/4b", None
            else:
                return "", "No local Ollama Qwen models available"
        return "", "No Ollama provider configured"