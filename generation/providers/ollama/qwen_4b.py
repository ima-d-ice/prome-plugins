"""Ollama Qwen 4B model configuration.

Represents the Qwen 4B model at 4-bit quantization, configured for local
Ollama inference. Model tag is user-configurable via
OLLAMA_QWEN_4B_MODEL environment variable.

Do not invent model tags or assume a specific Qwen release.
Inspect the actual local Ollama installation to determine the correct tag.
"""

from __future__ import annotations

from generation.contracts.models import GenerationTaskType, ModelConfiguration

#: The default model tag for Qwen 4B at 4-bit quantization.
#: Users must configure this via OLLAMA_QWEN_4B_MODEL to match their local Ollama installation.
OLLAMA_QWEN_4B_MODEL: str | None = None

#: Default context window for Qwen 4B 4-bit (varies by release, 8192 is common).
CONTEXT_WINDOW_4B = 8192

#: Default output limits for Qwen 4B 4-bit.
OUTPUT_LIMITS_4B = {"max_output_tokens": 2048}

#: Supported generation parameters for Qwen 4B via Ollama.
SUPPORTED_PARAMETERS_4B = {
    "temperature": True,
    "top_p": True,
    "max_tokens": True,
    "stop": True,
}

qwen_4b_configuration = ModelConfiguration(
    provider="ollama",
    model_id="",  # Placeholder — user must set OLLAMA_QWEN_4B_MODEL to match local installation
    task_type=GenerationTaskType.TEXT,
    context_window=CONTEXT_WINDOW_4B,
    output_limits=OUTPUT_LIMITS_4B,
    supported_parameters=SUPPORTED_PARAMETERS_4B,
    quantization_metadata={"bits": 4, "quantization": "Q4_K_M", "description": "Qwen 4B 4-bit model, tag configured via OLLAMA_QWEN_4B_MODEL"},
    rate_limit_config={"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000},
)