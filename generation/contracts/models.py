"""Shared data models for the generation ecosystem.

These models are dependency-free: they contain no imports of Ollama, Groq,
PyTorch, Transformers, or HTTP SDKs. They describe the requests, responses,
model configurations, and provider protocols that flow through the generation
pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GenerationTaskType(str, Enum):
    """Type of generation task."""

    TEXT = "text"
    CODE = "code"
    FUNCTION_CALL = "function_call"
    STRUCTURED_OUTPUT = "structured_output"


@dataclass(frozen=True)
class ModelConfiguration:
    """Configuration describing a generation model.

    Attributes:
        provider: the provider name (e.g. "groq", "ollama").
        model_id: the model identifier (e.g. "qwen/qwen3.8-27b").
        task_type: the task type this model supports.
        context_window: maximum context window in tokens.
        output_limits: maximum output tokens supported.
        supported_parameters: generation parameters this model accepts.
        quantization_metadata: quantization info for local models.
        rate_limit_config: RPM, RPD, TPM, TPD limits.
    """

    provider: str
    model_id: str
    task_type: GenerationTaskType
    context_window: int
    output_limits: dict[str, Any] = field(default_factory=dict)
    supported_parameters: dict[str, Any] = field(default_factory=dict)
    quantization_metadata: dict[str, Any] = field(default_factory=dict)
    rate_limit_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationRequest:
    """Request object for a generation call.

    Attributes:
        user_input: the raw user query or prompt.
        messages: optional conversation messages (role, content pairs).
        context_items: optional retrieved context items to include.
        system_instruction: optional system-level instruction.
        provider: optional provider selection ("groq", "ollama").
        model: optional model selection (e.g. "qwen/qwen3.8-27b").
        language_code: optional ISO-639-1 language code.
        generation_parameters: temperature, top-p, max_tokens, etc.
        input_token_limit: optional hard limit on input tokens.
        output_token_limit: optional hard limit on output tokens.
        enhance_query: optional flag to request query enhancement.
        safety_config: optional safety configuration dict.
        request_id: unique identifier for this request.
    """

    user_input: str
    messages: tuple[dict[str, str], ...] = field(default_factory=tuple)
    context_items: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    system_instruction: str | None = None
    provider: str | None = None
    model: str | None = None
    language_code: str | None = None
    generation_parameters: dict[str, Any] = field(default_factory=dict)
    input_token_limit: int | None = None
    output_token_limit: int | None = None
    enhance_query: bool = False
    safety_config: dict[str, Any] | None = None
    request_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.user_input, str):
            raise ValueError("user_input must be a string")
        if self.input_token_limit is not None and (not isinstance(self.input_token_limit, int) or self.input_token_limit < 0):
            raise ValueError("input_token_limit must be a non-negative integer or None")
        if self.output_token_limit is not None and (not isinstance(self.output_token_limit, int) or self.output_token_limit < 0):
            raise ValueError("output_token_limit must be a non-negative integer or None")


@dataclass(frozen=True)
class GenerationResponse:
    """Response object from a generation call.

    Attributes:
        text: the generated text.
        actual_provider: the provider that served the request.
        actual_model: the model ID that actually responded.
        finish_reason: why the generation stopped (e.g. "stop", "length", "tool_calls").
        input_tokens: number of input tokens used.
        output_tokens: number of output tokens generated.
        total_tokens: input + output tokens.
        actual_route: history of providers/models tried (fallback chain).
        latency: wall-clock time for the generation in milliseconds.
        request_id: correlates with the request ID.
        citation_source_refs: optional citation/reference references.
        diagnostics: additional diagnostics about the generation.
    """

    text: str
    actual_provider: str
    actual_model: str
    finish_reason: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    actual_route: tuple[dict[str, str], ...] = field(default_factory=tuple)
    latency: float = 0.0
    request_id: str | None = None
    citation_source_refs: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Capabilities:
    """Capability reporting for a generation provider.

    Attributes:
        supports_streaming: whether the provider supports streaming responses.
        supports_stop_sequences: whether stop sequences are supported.
        supports_temperature: whether temperature parameter is accepted.
        supports_top_p: whether top-p parameter is accepted.
        supports_reasoning_effort: whether reasoning_effort is supported.
        supports_structured_output: whether structured output is supported.
        max_context_window: maximum context window in tokens.
        max_output_tokens: maximum output tokens supported.
    """

    supports_streaming: bool
    supports_stop_sequences: bool
    supports_temperature: bool
    supports_top_p: bool
    supports_reasoning_effort: bool
    supports_structured_output: bool
    max_context_window: int
    max_output_tokens: int


@dataclass(frozen=True)
class ProviderProtocol:
    """Common interface for generation providers.

    This is a data-carrying structure, not a Python Protocol class,
    so implementations can opt-in to runtime_checkable without
    forcing library-specific imports into the contract layer.
    """

    name: str
    supplier: str | None = None
    capabilities: Capabilities | None = None
    model_configs: dict[str, ModelConfiguration] | None = None


class ModelParameter(str, Enum):
    """Supported generation parameter names."""

    TEMPERATURE = "temperature"
    TOP_P = "top_p"
    MAX_TOKENS = "max_tokens"
    REASONING_EFFORT = "reasoning_effort"
    STOP = "stop"
    STRUCTURED_OUTPUT = "structured_output"


@dataclass(frozen=True)
class GenerationDiagnostics:
    """Non-response diagnostics a generation call reports."""

    retry_count: int = 0
    """Number of retry attempts made."""

    fallback_count: int = 0
    """Number of model fallbacks taken."""

    key_rotations: int = 0
    """Number of key rotations performed."""

    rate_limit_wait_ms: int = 0
    """Total time spent waiting for rate limits."""

    guard_model: str | None = None
    """Prompt Guard model used, if any."""

    guard_classification: str | None = None
    """Prompt Guard classification result."""

    safeguard_decision: str | None = None
    """Safety evaluator decision, if any."""

    query_enhanced: bool = False
    """Whether query enhancement was applied."""

    truncation_details: dict[str, Any] | None = None
    """Details about any input truncation applied."""