"""Groq model-specific configurations and rate-limit settings.

Contains the explicit model configurations for the three Groq generation models
and the 12-key rate-limit definitions.
"""

from __future__ import annotations

from generation.contracts.models import GenerationTaskType, ModelConfiguration

#: Rate-limit configuration per Groq model (per key).
#: These are the initial reference configurations defined in the spec.
GROQ_MODEL_LIMITS: dict[str, dict[str, int]] = {
    "qwen/qwen3.8-27b": {
        "rpm": 30,
        "rpd": 1000,
        "tpm": 8000,
        "tpd": 200000,
    },
    "openai/gpt-oss-120b": {
        "rpm": 30,
        "rpd": 1000,
        "tpm": 8000,
        "tpd": 200000,
    },
    "openai/gpt-oss-20b": {
        "rpm": 30,
        "rpd": 1000,
        "tpm": 8000,
        "tpd": 200000,
    },
}

#: Explicit model configurations for the three Groq generation models.
#: The default fallback chain is:
#: qwen/qwen3.8-27b -> openai/gpt-oss-120b -> openai/gpt-oss-20b
GROQ_MODEL_CONFIGS: tuple[str, ...] = (
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
)

#: Map model ID to its configuration.
_MODEL_CONFIG_BY_ID: dict[str, ModelConfiguration] = {
    "qwen/qwen3.8-27b": ModelConfiguration(
        provider="groq",
        model_id="qwen/qwen3.8-27b",
        task_type=GenerationTaskType.TEXT,
        context_window=32768,
        output_limits={"max_output_tokens": 8192},
        supported_parameters={
            "temperature": True,
            "top_p": True,
            "max_tokens": True,
            "reasoning_effort": False,
            "stop": True,
            "structured_output": False,
        },
        quantization_metadata={},
        rate_limit_config={"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000},
    ),
    "openai/gpt-oss-120b": ModelConfiguration(
        provider="groq",
        model_id="openai/gpt-oss-120b",
        task_type=GenerationTaskType.TEXT,
        context_window=128000,
        output_limits={"max_output_tokens": 8192},
        supported_parameters={
            "temperature": True,
            "top_p": True,
            "max_tokens": True,
            "reasoning_effort": True,
            "stop": True,
            "structured_output": True,
        },
        quantization_metadata={},
        rate_limit_config={"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000},
    ),
    "openai/gpt-oss-20b": ModelConfiguration(
        provider="groq",
        model_id="openai/gpt-oss-20b",
        task_type=GenerationTaskType.TEXT,
        context_window=32768,
        output_limits={"max_output_tokens": 8192},
        supported_parameters={
            "temperature": True,
            "top_p": True,
            "max_tokens": True,
            "reasoning_effort": True,
            "stop": True,
            "structured_output": True,
        },
        quantization_metadata={},
        rate_limit_config={"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000},
    ),
}