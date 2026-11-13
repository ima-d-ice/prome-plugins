"""Ollama Qwen 8B model configuration.

Represents the Qwen 8B model at 4-bit quantization, configured for local
Ollama inference. Model tag is user-configurable via
OLLAMA_QWEN_8B_MODEL environment variable.

Do not invent model tags or assume a specific Qwen release.
Inspect the actual local Ollama installation to determine the correct tag.
"""

from __future__ import annotations

from generation.contracts.models import GenerationTaskType, ModelConfiguration

#: The default model tag for Qwen 8B at 4-bit quantization.
#: Users must configure this via OLLAMA_QWEN_8B_MODEL to match their local Ollama installation.
OLLAMA_QWEN_8B_MODEL: str | None = None

#: Default context window for Qwen 8B 4-bit (varies by release, 8192 is common).
CONTEXT_WINDOW_8B = 8192

#: Default output limits for Qwen 8B 4-bit.
OUTPUT_LIMITS_8B = {"max_output_tokens": 2048}

#: Supported generation parameters for Qwen 8B via Ollama.
SUPPORTED_PARAMETERS_8B = {
    "temperature": True,
    "top_p": True,
    "max_tokens": True,
    "stop": True,
}

qwen_8b_configuration = ModelConfiguration(
    provider="ollama",
    model_id="",  # Placeholder — user must set OLLAMA_QWEN_8B_MODEL to match local installation
    task_type=GenerationTaskType.TEXT,
    context_window=CONTEXT_WINDOW_8B,
    output_limits=OUTPUT_LIMITS_8B,
    supported_parameters=SUPPORTED_PARAMETERS_8B,
    quantization_metadata={"bits": 4, "quantization": "Q4_K_M", "description": "Qwen 8B 4-bit model, tag configured via OLLAMA_QWEN_8B_MODEL"},
    rate_limit_config={"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000},
)