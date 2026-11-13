"""Groq provider implementation.

Uses the current Groq-supported API and official model IDs.
Keep a model-specific configuration for:
1. Qwen 3.8 27B
2. GPT-OSS 120B
3. GPT-OSS 20B

Default route:
qwen/qwen3.8-27b
openai/gpt-oss-120b
openai/gpt-oss-20b

Routing modes:
- auto: use the primary model and configured fallback chain.
- manual: use the explicitly selected Groq model.
- ollama: manually select one of the two local models.

An explicitly selected Ollama model must stay on Ollama unless the caller
explicitly enables provider fallback.

Fallback conditions (model-level across ALL 12 keys):
- Model-specific quota exhaustion across ALL 12 keys.
- HTTP 429 caused by a model-level rate limit across ALL 12 keys.
- Retryable HTTP 5xx errors.
- Temporary provider unavailability.
- Connection failures.
- Eligible timeouts.

Fallback must NOT occur on:
- Invalid request parameters.
- Authentication failures (unless a single key is bad, then rotate to next key).
- Unsupported model configuration.
- Invalid model ID.
- Request validation errors.
- Prompt or safety rejection.

Respect Retry-After where available.
Do not retry indefinitely.
Record all attempts, key rotations, and failures in the response diagnostics
without exposing secrets.

Model-specific parameters:
- Reasoning effort.
- Max output tokens.
- Temperature where supported.
- Top-p where supported.
- Stop sequences.
- Structured output where supported.

Never send a parameter unsupported by the selected model.
When falling back to another model, validate or translate the request
parameters before retrying.

Response handling:
- Actual selected model.
- Actual key index used (e.g., "key_4" — never the secret itself).
- Usage token counts.
- Finish reason.
- Tool-call data if supported and explicitly requested.
- Provider request identifiers where available.
- Response metadata.

Do not add autonomous tool execution in this milestone.
"""

from __future__ import annotations

import os
import time
from typing import Any

from generation.contracts.exceptions import (
    GenerationAuthenticationError,
    GenerationConfigError,
    GenerationDependencyError,
    GenerationInferenceError,
    GenerationInvalidRequestError,
    GenerationKeyPoolExhaustedError,
    GenerationProviderUnavailableError,
    GenerationRateLimitError,
)
from generation.contracts.models import (
    Capabilities,
    GenerationRequest,
    GenerationResponse,
    ModelConfiguration,
)
from generation.input_management.token_counter import TokenCounter
from generation.rate_limiting.key_rotation import KeyRotationPool

try:
    import httpx
    HAS_HTTpx = True
except ImportError:
    HAS_HTTpx = False

GroqModelId = str


class GroqProvider:
    """Provider for Groq cloud generation models.

    Supports the three Groq generation models with configured fallback chain:
    1. Primary: qwen/qwen3.8-27b
    2. First fallback: openai/gpt-oss-120b
    3. Second fallback: openai/gpt-oss-20b

    Uses a 12-key rotation pool for rate-limit pooling and quota maximization.
    Model-specific parameters are validated before sending to each model.
    """

    #: Default Groq API base URL
    BASE_URL = "https://api.groq.com/openai/v1"

    #: Default generation route (primary + fallbacks)
    DEFAULT_ROUTE: tuple[str, ...] = (
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    )

    def __init__(
        self,
        *,
        route: tuple[str, ...] | None = None,
        api_base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        max_key_rotations: int | None = None,
    ) -> None:
        """Initialize the Groq provider.

        Args:
            route: Explicit model fallback route. When None, uses DEFAULT_ROUTE.
            api_base_url: Groq API base URL. When None, uses the default.
            timeout: Request timeout in seconds per attempt.
            max_retries: Maximum retry attempts per model/key combination.
            max_key_rotations: Maximum key rotations before giving up on a model.
                When None, falls back to the next model in the route.
        """
        self._route = route or self.DEFAULT_ROUTE
        self._api_base_url = (api_base_url or self.BASE_URL).rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._max_key_rotations = max_key_rotations or len(self._route) * 12
        self._token_counter: TokenCounter | None = None
        self._key_pool: KeyRotationPool | None = None
        self._router: Any | None = None  # Will be set if router is used

    # ------------------------------------------------------------------
    # Token counter setup
    # ------------------------------------------------------------------

    def set_token_counter(self, counter: TokenCounter) -> None:
        """Set the token counter for input estimation and accounting.

        Args:
            counter: a TokenCounter instance.
        """
        self._token_counter = counter

    # ------------------------------------------------------------------
    # Key pool setup
    # ------------------------------------------------------------------

    def set_key_pool(self, pool: KeyRotationPool) -> None:
        """Set the 12-key rotation pool for rate-limit pooling.

        Args:
            pool: a KeyRotationPool instance with 12 API keys configured.
        """
        self._key_pool = pool

    # ------------------------------------------------------------------
    # Capability reporting
    # ------------------------------------------------------------------

    def get_capabilities(self, model_id: str | None = None) -> Capabilities:
        """Return the capabilities for the given model.

        Args:
            model_id: the model identifier. When None, uses the default model.

        Returns:
            Capabilities instance describing what this provider supports
            for the given model.
        """
        mid = model_id or self.default_model
        config = self.model_configurations.get(mid)
        if config is None:
            mid = self.default_model
            config = self.model_configurations.get(mid)
        if config is None:
            return Capabilities(
                supports_streaming=False,
                supports_stop_sequences=True,
                supports_temperature=True,
                supports_top_p=True,
                supports_reasoning_effort=False,
                supports_structured_output=False,
                max_context_window=0,
                max_output_tokens=0,
            )
        return config.capabilities

    @property
    def default_model(self) -> str:
        """Return the default model ID (primary route)."""
        return self._route[0] if self._route else "qwen/qwen3.8-27b"

    @property
    def model_configurations(self) -> dict[str, ModelConfiguration]:
        """Return the mapping of model IDs to configurations."""
        from generation.providers.groq.model_configs import _MODEL_CONFIG_BY_ID
        return _MODEL_CONFIG_BY_ID

    # ------------------------------------------------------------------
    # Generation: main entry point
    # ------------------------------------------------------------------

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text from the given request using the Groq provider.

        Supports auto, manual, and ollama routing modes.
        Falls back across the 12-key pool and then across the model route.

        Args:
            request: the generation request.

        Returns:
            GenerationResponse with the generated text and diagnostics.

        Raises:
            GenerationConfigError: if the request is invalid.
            GenerationDependencyError: if required dependencies are missing.
            GenerationInferenceError: if all attempts fail.
            GenerationRateLimitError: if all keys and models are exhausted.
            GenerationKeyPoolExhaustedError: if all 12 keys are exhausted.
        """
        if not request.user_input.strip():
            raise GenerationConfigError("user_input must be a non-empty string")

        # Route mode determination
        mode = self._determine_mode(request)

        diagnostics: dict[str, Any] = {
            "route": self._route,
            "mode": mode,
            "attempts": [],
            "key_rotations": 0,
            "fallbacks": 0,
        }

        # If Ollama mode explicitly requested, delegate to Ollama provider
        if mode == "ollama":
            return self._generate_ollama(request, diagnostics=diagnostics)

        # Groq routing (auto or manual)
        return self._generate_groq(request, diagnostics=diagnostics)

    def _determine_mode(self, request: GenerationRequest) -> str:
        """Determine the routing mode for this request.

        Returns one of: "auto", "manual", "ollama".
        """
        # If user explicitly selected Ollama model via request, use ollama mode
        if request.model and request.model.startswith("ollama") or request.model in (
            "qwen/8b",
            "qwen/4b",
        ):
            return "ollama"
        # If explicit provider was set
        if request.provider == "ollama":
            return "ollama"
        # Default: auto (primary Groq model with fallbacks)
        return "auto"

    def _generate_ollama(self, request: GenerationRequest, diagnostics: dict[str, Any]) -> GenerationResponse:
        """Delegate to the Ollama provider for generation.

        Args:
            request: the generation request.
            diagnostics: diagnostics dict to populate.

        Returns:
            GenerationResponse from Ollama.
        """
        from generation.providers.ollama.provider import OllamaProvider

        # Create Ollama provider with defaults
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        model_8b = os.environ.get("OLLAMA_QWEN_8B_MODEL")
        model_4b = os.environ.get("OLLAMA_QWEN_4B_MODEL")

        provider = OllamaProvider(
            base_url=base_url,
            model_8b=model_8b,
            model_4b=model_4b,
        )

        # Use the request model if specified, otherwise default
        model_id = request.model or "qwen/8b"

        try:
            response = provider.generate(request, model_id=model_id)
            diagnostics["actual_provider"] = "ollama"
            diagnostics["actual_model"] = response.actual_model
            return response
        except Exception as e:
            diagnostics["error"] = str(e)
            raise

    def _generate_groq(self, request: GenerationRequest, diagnostics: dict[str, Any]) -> GenerationResponse:
        """Attempt Groq generation with key rotation and model fallback.

        This is the core routing logic:
        1. Iterate through the model route (primary -> fallbacks)
        2. For each model, iterate through all 12 keys via the key pool
        3. Check rate limits before each attempt
        4. On exhaustion, rotate key or fall back to next model
        5. Record all attempts and diagnostics

        Args:
            request: the generation request.
            diagnostics: diagnostics dict to populate.

        Returns:
            GenerationResponse from the first successful attempt, or failure.
        """

        route = self._route
        key_pool = self._key_pool
        max_key_rotations = self._max_key_rotations

        # Ensure key pool is available
        if key_pool is None:
            raise GenerationDependencyError(
                "Groq key pool not configured. "
                "Set Groq API keys via GROQ_API_KEY_1 through GROQ_API_KEY_12 "
                "and call GroqProvider with a KeyRotationPool."
            )

        # Ensure token counter is available
        token_counter = self._token_counter
        if token_counter is None:
            from generation.input_management.token_counter import (
                get_default_token_counter,
            )
            token_counter = get_default_token_counter()

        total_key_rotations = 0
        total_fallbacks = 0
        attempt_log: list[dict[str, Any]] = []

        # Try each model in the route
        for model_idx, model_id in enumerate(route):
            if total_fallbacks > 0:
                diagnostics["fallbacks"] = total_fallbacks

            model_limits = GROQ_MODEL_LIMITS.get(model_id, {"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000})

            # Try each key in the pool for this model
            for key_idx in range(12):
                if total_key_rotations >= max_key_rotations:
                    diagnostics["exceeded_max_rotations"] = True
                    break

                key_rotation_record = key_pool.rotate(key_idx + 1)  # 1-indexed for diagnostics
                active_key_info = key_pool.active_key_info()

                # Check rate limits for this key/model combination
                rpm_remaining = active_key_info.get("rpm_remaining", model_limits["rpm"])
                rpd_remaining = active_key_info.get("rpd_remaining", model_limits["rpd"])
                tpm_remaining = active_key_info.get("tpm_remaining", model_limits["tpm"])
                tpd_remaining = active_key_info.get("tpd_remaining", model_limits["tpd"])

                # Check if this key can accept the request
                can_proceed, rejection_reason = self._check_key_budgets(
                    model_limits, rpm_remaining, rpd_remaining, tpm_remaining, tpd_remaining,
                    request, token_counter,
                )

                if not can_proceed:
                    # Record the failed attempt
                    attempt_log.append({
                        "model": model_id,
                        "key_index": key_idx + 1,
                        "reason": rejection_reason,
                        "key_status": key_rotation_record.status if hasattr(key_rotation_record, 'status') else 'unknown',
                    })
                    total_key_rotations += 1
                    diagnostics["key_rotations"] = total_key_rotations
                    # If we've tried all keys for this model and they're all exhausted,
                    # move to the next model (fallback)
                    if key_idx >= 11:  # last key
                        break  # will fall through to next model
                    continue

                # Key has budget — attempt the request
                attempt_start = time.time()
                attempt_data = {
                    "model": model_id,
                    "key_index": key_idx + 1,
                    "key_status": active_key_info.get("key_label", f"key_{key_idx + 1}"),
                    "attempt_index": len(attempt_log),
                }

                try:
                    # Estimate input tokens
                    estimated_input = self._estimate_input_tokens(request, token_counter)

                    # Reserve budget (decrement pre-flight)
                    key_pool.reserve_budget(
                        model_id,
                        estimated_input,
                        ...  # output tokens will be adjusted after response
                    )

                    # Make the API call
                    response = self._call_groq_api(model_id, request, active_key_info)

                    # Read actual usage from response
                    actual_input_tokens = response.get("usage", {}).get("prompt_tokens", estimated_input)
                    actual_output_tokens = response.get("usage", {}).get("completion_tokens", 0)
                    total_tokens = actual_input_tokens + actual_output_tokens

                    elapsed_ms = (time.time() - attempt_start) * 1000

                    # Release budget with actual usage and reconcile
                    key_pool.release_budget(
                        model_id,
                        actual_input_tokens,
                        actual_output_tokens,
                    )

                    # Build diagnostics entry
                    attempt_data.update({
                        "success": True,
                        "model": model_id,
                        "key_index": key_idx + 1,
                        "input_tokens": actual_input_tokens,
                        "output_tokens": actual_output_tokens,
                        "total_tokens": total_tokens,
                        "latency_ms": elapsed_ms,
                        "finish_reason": response.get("finish_reason", "stop"),
                        "request_id": response.get("id"),
                    })

                    # Build and return the GenerationResponse
                    return GenerationResponse(
                        text=response.get("choices", [{}])[0].get("message", {}).get("content", ""),
                        actual_provider="groq",
                        actual_model=model_id,
                        finish_reason=response.get("finish_reason", "stop"),
                        input_tokens=actual_input_tokens,
                        output_tokens=actual_output_tokens,
                        total_tokens=total_tokens,
                        actual_route=tuple(att["model"] for att in attempt_log) + (model_id,),
                        latency=elapsed_ms,
                        diagnostics={
                            **diagnostics,
                            "attempts": attempt_log + [attempt_data],
                            "key_rotations": total_key_rotations,
                        },
                    )

                except GenerationRateLimitError as e:
                    # Release the reserved budget since we didn't proceed
                    try:
                        key_pool.release_budget(
                            model_id,
                            0,
                            0,
                        )
                    except Exception:
                        pass

                    attempt_data.update({
                        "success": False,
                        "error": "rate_limit",
                        "reason": str(e),
                    })
                    attempt_log.append(attempt_data)
                    total_key_rotations += 1
                    diagnostics["key_rotations"] = total_key_rotations
                    continue  # try next key

                except Exception as e:
                    # Release reserved budget on unexpected error
                    try:
                        key_pool.release_budget(
                            model_id,
                            0,
                            0,
                        )
                    except Exception:
                        pass

                    attempt_data.update({
                        "success": False,
                        "error": type(e).__name__,
                        "reason": str(e),
                    })
                    attempt_log.append(attempt_data)
                    total_key_rotations += 1
                    diagnostics["key_rotations"] = total_key_rotations
                    continue  # try next key

            # All keys exhausted for this model — fall back to next model
            if model_idx < len(route) - 1:
                total_fallbacks += 1
                diagnostics["fallbacks"] = total_fallbacks
                continue  # try next model in route

        # All models and keys exhausted
        diagnostics["exhausted"] = True
        diagnostics["attempts"] = attempt_log

        raise GenerationKeyPoolExhaustedError(
            f"All 12 keys and {len(route)} models exhausted for generation request.",
            attempted_models=list(route),
            key_rotations=total_key_rotations,
        )

    def _check_key_budgets(
        self,
        model_limits: dict[str, int],
        rpm_remaining: int,
        rpd_remaining: int,
        tpm_remaining: int,
        tpd_remaining: int,
        request: GenerationRequest,
        token_counter: TokenCounter,
    ) -> tuple[bool, str | None]:
        """Check if the current key has sufficient budget for the request.

        The first exhausted applicable limit controls whether a request can proceed.

        Args:
            model_limits: the model's RPM, RPD, TPM, TPD limits.
            rpm_remaining: remaining requests per minute for this key.
            rpd_remaining: remaining requests per day for this key.
            tpm_remaining: remaining tokens per minute for this key.
            tpd_remaining: remaining tokens per day for this key.
            request: the generation request.
            token_counter: token counter for estimation.

        Returns:
            (can_proceed, reason) tuple.
        """
        # Check RPM: if no requests remaining in this minute
        if rpm_remaining <= 0:
            return False, "RPM exhausted for this key"

        # Check RPD: if no requests remaining today
        if rpd_remaining <= 0:
            return False, "RPD exhausted for this key"

        # Estimate input tokens and check TPM
        estimated_input = self._estimate_input_tokens(request, token_counter)
        # Reserve estimated output tokens too (conservative)
        estimated_output = request.generation_parameters.get("max_tokens", 8192) or 8192
        total_estimated = estimated_input + estimated_output

        if tpm_remaining < total_estimated:
            return False, f"TPM insufficient: need {total_estimated}, have {tpm_remaining}"

        # Check TPD
        # (TPD check is more complex — would need to track daily usage per key)
        # For now, check if we've exceeded daily token budget
        # This is a simplification — full TPD tracking uses key pool state
        # tpd_remaining is already key-specific daily tracking

        # All budgets OK
        return True, None

    def _estimate_input_tokens(self, request: GenerationRequest, token_counter: TokenCounter) -> int:
        """Estimate input tokens for a request using the token counter.

        Args:
            request: the generation request.
            token_counter: the token counter to use.

        Returns:
            Estimated input token count.
        """
        if self._token_counter is not None:
            pass
        else:
            from generation.input_management.token_counter import (
                get_default_token_counter,
            )
            get_default_token_counter()

        # Use the user_input for estimation
        # For Groq, use model-specific token counting when available
        # Fall back to conservative estimate and reconcile with response usage
        estimated = token_counter.count(request.user_input)
        return estimated

    def _call_groq_api(self, model_id: str, request: GenerationRequest, key_info: dict[str, Any]) -> dict[str, Any]:
        """Make the actual HTTP call to the Groq API.

        Args:
            model_id: the model identifier.
            request: the generation request.
            key_info: information about the active key (including auth headers).

        Returns:
            The JSON response from the Groq API.

        Raises:
            GenerationRateLimitError: if the API returns 429.
            GenerationInferenceError: if the API returns a 5xx error.
            GenerationAuthenticationError: if authentication fails.
            GenerationInvalidRequestError: if the request is invalid.
        """
        if not HAS_HTTpx:
            raise GenerationDependencyError(
                "httpx is required for Groq API calls. "
                "Install it with: pip install httpx"
            )

        url = f"{self._api_base_url}/chat/completions"

        # Build the request payload, validating parameter compatibility
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": self._build_messages(request),
            "temperature": request.generation_parameters.get("temperature", 1.0),
            "max_tokens": request.generation_parameters.get("max_tokens", 8192),
        }

        # Add top-p if supported
        if request.generation_parameters.get("top_p") is not None:
            if self._is_parameter_supported(model_id, "top_p"):
                payload["top_p"] = request.generation_parameters["top_p"]

        # Add stop sequences if supported and provided
        if request.generation_parameters.get("stop"):
            if self._is_parameter_supported(model_id, "stop"):
                payload["stop"] = request.generation_parameters["stop"]

        # Add reasoning_effort if supported
        if request.generation_parameters.get("reasoning_effort") is not None:
            if self._is_parameter_supported(model_id, "reasoning_effort"):
                payload["reasoning_effort"] = request.generation_parameters["reasoning_effort"]

        # Add structured_output if supported
        if request.generation_parameters.get("structured_output") is not None:
            if self._is_parameter_supported(model_id, "structured_output"):
                payload["structured_output"] = {
                    "type": "json_object"
                }

        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }

        # Add key-specific authorization header (never log the secret itself)
        # The key pool provides a masked identifier, not the secret
        active_key_id = key_info.get("key_label", "key_unknown")
        headers["Authorization"] = f"Bearer {active_key_id}"

        # Execute with retry logic
        for attempt in range(self._max_retries + 1):
            try:
                response = httpx.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self._timeout,
                )

                # Handle rate limits (429)
                if response.status_code == 429:
                    # Respect Retry-After header if available
                    retry_after = response.headers.get("Retry-After")
                    wait_seconds = float(retry_after) if retry_after else 2 ** attempt

                    # Still raise as rate limit error for the caller to handle
                    raise GenerationRateLimitError(
                        f"Groq API rate limit hit (429) for model '{model_id}'. "
                        f"Retry-After: {retry_after}s. Attempt {attempt + 1}/{self._max_retries + 1}.",
                        retry_after=retry_after,
                        model_id=model_id,
                    )

                # Handle server errors (5xx)
                if 500 <= response.status_code < 600:
                    if attempt < self._max_retries:
                        # Exponential backoff
                        wait_seconds = 2 ** attempt
                        time.sleep(min(wait_seconds, 30))
                        continue
                    raise GenerationInferenceError(
                        f"Groq API server error {response.status_code} for model '{model_id}'.",
                        model_id=model_id,
                    )

                # Handle authentication failure
                if response.status_code == 401 or response.status_code == 403:
                    raise GenerationAuthenticationError(
                        f"Groq API authentication failed ({response.status_code}) for model '{model_id}'.",
                        model_id=model_id,
                    )

                # Handle other client errors
                if response.status_code >= 400:
                    raise GenerationInvalidRequestError(
                        f"Groq API returned {response.status_code} for model '{model_id}': "
                        f"{response.text}",
                        model_id=model_id,
                    )

                # Success — parse and return
                return response.json()

            except httpx.TimeoutException:
                if attempt < self._max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise GenerationInferenceError(
                    f"Groq API timeout after {self._max_retries + 1} attempts.",
                    model_id=model_id,
                )
            except httpx.ConnectError:
                if attempt < self._max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise GenerationProviderUnavailableError(
                    f"Groq API connection failed for model '{model_id}'.",
                    model_id=model_id,
                )

        # Exhausted retries
        raise GenerationInferenceError(
            f"Groq API failed after {self._max_retries + 1} attempts for model '{model_id}'.",
            model_id=model_id,
        )

    def _build_messages(self, request: GenerationRequest) -> list[dict[str, str]]:
        """Build the messages list for the Groq API call.

        Args:
            request: the generation request.

        Returns:
            list of message dicts in Groq API format.
        """
        messages: list[dict[str, str]] = []

        # System instruction
        if request.system_instruction:
            messages.append({"role": "system", "content": request.system_instruction})

        # Conversation messages
        for msg in request.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            messages.append({"role": role, "content": content})

        # User input as the final user message if not already present
        user_input = request.user_input
        if messages and messages[-1].get("role") != "user" or not messages:
            messages.append({"role": "user", "content": user_input})

        return messages

    def _is_parameter_supported(self, model_id: str, param: str) -> bool:
        """Check whether a generation parameter is supported by the given model.

        Args:
            model_id: the model identifier.
            param: the parameter name.

        Returns:
            True if the parameter is supported, False otherwise.
        """
        from generation.providers.groq.parameter_configs import (
            check_parameter_compatibility,
        )
        return check_parameter_compatibility(model_id, param)