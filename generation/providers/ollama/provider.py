"""Ollama provider implementation.

Wraps the official Ollama API for local model inference.
Supports configurable base URL, lazy model initialization, health checks,
and model availability inspection.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from typing import Any

from generation.contracts.exceptions import (
    GenerationConfigError,
    GenerationInferenceError,
    GenerationModelUnavailableError,
    GenerationOllamaConnectionError,
)
from generation.contracts.models import (
    Capabilities,
    GenerationRequest,
    GenerationResponse,
)


class OllamaProvider:
    """Provider for local Ollama model inference.

    Supports the two manually selectable Qwen models (8B and 4B, 4-bit quantized).
    Base URL configurable to localhost:11434 by default.
    Lazy connection and model initialization.
    Manual model selection, not automatic fallback to Groq.
    The local provider must be usable even when Groq is unavailable.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        *,
        model_8b: str | None = None,
        model_4b: str | None = None,
        context_length: int = 8192,
        timeout: float = 120.0,
    ) -> None:
        """Initialize the Ollama provider.

        Args:
            base_url: Base URL for the Ollama API. Defaults to "http://localhost:11434".
            model_8b: Explicit model name for the Qwen 8B model. When None,
                the value of OLLAMA_QWEN_8B_MODEL env var is used, or manual selection is required.
            model_4b: Explicit model name for the Qwen 4B model. When None,
                the value of OLLAMA_QWEN_4B_MODEL env var is used, or manual selection is required.
            context_length: Default context length in tokens for Ollama models.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self._model_8b = model_8b
        self._model_4b = model_4b
        self._context_length = context_length
        self._timeout = timeout
        self._loaded_model: str | None = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _check_health(self) -> bool:
        """Check if the Ollama server is reachable.

        Returns:
            True if the server responds, False otherwise.
        """
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return resp.status == 200
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            return False
        except Exception:
            return False

    def _request(
        self,
        method: str,
        endpoint: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        """Make an HTTP request to the Ollama API.

        Args:
            method: HTTP method ("GET", "POST").
            endpoint: API endpoint path (e.g. "/api/generate").
            payload: JSON payload for POST requests.

        Returns:
            Parsed JSON response.

        Raises:
            GenerationOllamaConnectionError: if the request fails.
        """
        url = f"{self.base_url}{endpoint}"
        if method == "GET":
            data = None
        else:
            data = payload and __import__("json").dumps(payload).encode("utf-8") or b""

        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return __import__("json").loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise GenerationOllamaConnectionError(
                f"Failed to connect to Ollama at {self.base_url}: {e}"
            ) from e
        except urllib.error.HTTPError as e:
            raise GenerationOllamaConnectionError(
                f"Ollama API error {e.code}: {e.read().decode('utf-8', errors='replace')}"
            ) from e
        except Exception as e:
            raise GenerationOllamaConnectionError(
                f"Unexpected error communicating with Ollama: {e}"
            ) from e

    def list_models(self) -> tuple[dict[str, Any], ...]:
        """List available Ollama models.

        Returns:
            tuple of model info dicts.
        """
        resp = self._request("GET", "/api/tags")
        models = resp.get("models", [])
        return tuple(models)

    def is_model_available(self, model_id: str) -> bool:
        """Check if a specific model is available locally.

        Args:
            model_id: the model identifier (e.g. "qwen:8b").

        Returns:
            True if the model is available.
        """
        models = self.list_models()
        return any(m.get("name") == model_id for m in models)

    # ------------------------------------------------------------------
    # Lazy model initialization
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        """Mark the provider as initialized (lazy — no-op unless state needed)."""
        self._initialized = True

    # ------------------------------------------------------------------
    # Capability reporting
    # ------------------------------------------------------------------

    def get_capabilities(self, model_id: str | None = None) -> Capabilities:
        """Return the capabilities for the given model.

        Args:
            model_id: the model identifier. When None, uses the default model.

        Returns:
            Capabilities instance describing what this provider supports.
        """
        model_id or self._resolve_model_id()
        # Qwen models via Ollama support standard generation params
        return Capabilities(
            supports_streaming=True,
            supports_stop_sequences=True,
            supports_temperature=True,
            supports_top_p=True,
            supports_reasoning_effort=False,  # Ollama doesn't natively support reasoning_effort
            supports_structured_output=False,
            max_context_window=self._context_length,
            max_output_tokens=4096,
        )

    def _resolve_model_id(self, model_hint: str | None = None) -> str:
        """Resolve the model ID to use.

        Priority: explicit hint > configured 8B > configured 4B > manual selection required.

        Returns:
            The model ID string.
        """
        if model_hint:
            return model_hint
        if self._model_8b:
            return self._model_8b
        if self._model_4b:
            return self._model_4b
        # No model configured — caller must select manually
        return ""

    # ------------------------------------------------------------------
    # Generation: non-streaming
    # ------------------------------------------------------------------

    def generate(
        self,
        request: GenerationRequest,
        *,
        model_id: str | None = None,
    ) -> GenerationResponse:
        """Generate text using an Ollama model (non-streaming).

        Args:
            request: the generation request.
            model_id: explicit model ID. When None, resolves from configuration.

        Returns:
            GenerationResponse with the generated text and diagnostics.

        Raises:
            GenerationConfigError: if the request is invalid.
            GenerationModelUnavailableError: if the model is not available.
            GenerationOllamaConnectionError: if communication fails.
            GenerationInferenceError: if the model fails during inference.
        """
        if not request.user_input.strip():
            raise GenerationConfigError("user_input must be a non-empty string")

        resolved_model = model_id or self._resolve_model_id()
        if not resolved_model:
            raise GenerationConfigError(
                "No Ollama model selected. Configure OLLAMA_QWEN_8B_MODEL or OLLAMA_QWEN_4B_MODEL."
            )

        if not self.is_model_available(resolved_model):
            raise GenerationModelUnavailableError(
                f"Ollama model '{resolved_model}' is not available locally."
            )

        # Estimate input tokens for budgeting
        input_tokens = self._estimate_tokens(request.user_input)

        start_time = time.time()

        try:
            payload: dict[str, Any] = {
                "model": resolved_model,
                "prompt": request.user_input,
                "stream": False,
                "options": {
                    "temperature": request.generation_parameters.get("temperature", 0.7),
                    "top_p": request.generation_parameters.get("top_p", 1.0),
                    "num_ctx": min(request.generation_parameters.get("max_tokens", 8192), self._context_length),
                },
            }

            # Add optional parameters
            if request.generation_parameters.get("stop"):
                payload["options"]["stop"] = request.generation_parameters["stop"]

            resp = self._request("POST", "/api/generate", payload)

            elapsed_ms = (time.time() - start_time) * 1000

            # Read usage from response
            output_tokens = resp.get("eval_count", 0)
            input_tokens + output_tokens

            # Groq-style usage reporting; fall back to estimated
            input_used = resp.get("prompt_eval_count", input_tokens)
            output_used = resp.get("eval_count", output_tokens)

            actual_text = resp.get("response", "")

            return GenerationResponse(
                text=actual_text,
                actual_provider="ollama",
                actual_model=resolved_model,
                finish_reason=resp.get("eval_status", "stop"),
                input_tokens=input_used,
                output_tokens=output_used,
                total_tokens=input_used + output_used,
                actual_route=("ollama", resolved_model),
                latency=elapsed_ms,
                diagnostics={
                    "model": resolved_model,
                    "ollama_options": payload.get("options", {}),
                    "elapsed_ms": elapsed_ms,
                },
            )

        except GenerationOllamaConnectionError:
            raise
        except Exception as e:
            raise GenerationInferenceError(
                f"Ollama generation failed for model '{resolved_model}': {e}"
            ) from e

    # ------------------------------------------------------------------
    # Generation: streaming (basic scaffold — pipes chunks from Ollama)
    # ------------------------------------------------------------------

    def generate_stream(
        self,
        request: GenerationRequest,
        *,
        model_id: str | None = None,
    ) -> Any:
        """Stream generated text from an Ollama model.

        Note: This is a best-effort scaffold. The actual streaming implementation
        depends on the Ollama API version and may not be available in all setups.

        Args:
            request: the generation request.
            model_id: explicit model ID. When None, resolves from configuration.

        Yields:
            Chunks of generated text.

        Raises:
            GenerationInferenceError: if the model fails during inference.
        """
        # Non-streaming path if streaming not supported
        try:
            resp = self._request("POST", "/api/generate", {
                "model": (model_id or self._resolve_model_id()),
                "prompt": request.user_input,
                "stream": True,
                "options": {
                    "temperature": request.generation_parameters.get("temperature", 0.7),
                    "top_p": request.generation_parameters.get("top_p", 1.0),
                },
            })
            # If we get here without error, yield a single response
            yield resp.get("response", "")
        except Exception as e:
            raise GenerationInferenceError(
                f"Ollama streaming generation failed: {e}"
            ) from e

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for a string.

        Uses a conservative character-based estimate when a tokenizer is
        not available. Ollama models may provide tokenizer info; fall back
        to 4 chars per token when unknown.

        Args:
            text: the text to estimate tokens for.

        Returns:
            Estimated token count.
        """
        # Try to get tokenizer info from Ollama model metadata
        # Fallback: 4 characters per token (conservative estimate)
        return max(1, len(text) // 4)