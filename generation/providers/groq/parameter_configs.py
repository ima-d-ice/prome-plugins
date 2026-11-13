"""Groq model-specific parameter compatibility settings.

Defines which generation parameters each supported model accepts.
Never send a parameter unsupported by the selected model.
When falling back to another model, validate or translate the request
parameters before retrying.
"""

from __future__ import annotations

from generation.contracts.models import (
    ModelConfiguration,
    ModelParameter,
    Supports,
)

#: Parameter compatibility for qwen/qwen3.8-27b
QWEN38_27B_PARAMS = {
    ModelParameter.TEMPERATURE: Supports.YES,
    ModelParameter.TOP_P: Supports.YES,
    ModelParameter.MAX_TOKENS: Supports.YES,
    ModelParameter.REASONING_EFFORT: Supports.NO,
    ModelParameter.STOP: Supports.YES,
    ModelParameter.STRUCTURED_OUTPUT: Supports.NO,
}

#: Parameter compatibility for openai/gpt-oss-120b
GPT_OSS_120B_PARAMS = {
    ModelParameter.TEMPERATURE: Supports.YES,
    ModelParameter.TOP_P: Supports.YES,
    ModelParameter.MAX_TOKENS: Supports.YES,
    ModelParameter.REASONING_EFFORT: Supports.YES,
    ModelParameter.STOP: Supports.YES,
    ModelParameter.STRUCTURED_OUTPUT: Supports.YES,
}

#: Parameter compatibility for openai/gpt-oss-20b
GPT_OSS_20B_PARAMS = {
    ModelParameter.TEMPERATURE: Supports.YES,
    ModelParameter.TOP_P: Supports.YES,
    ModelParameter.MAX_TOKENS: Supports.YES,
    ModelParameter.REASONING_EFFORT: Supports.YES,
    ModelParameter.STOP: Supports.YES,
    ModelParameter.STRUCTURED_OUTPUT: Supports.YES,
}


def check_parameter_compatibility(model_id: str, param: ModelParameter) -> bool:
    """Check whether a generation parameter is supported by a given model.

    Args:
        model_id: the model identifier.
        param: the generation parameter to check.

    Returns:
        True if the parameter is supported, False otherwise.
    """
    configs = {
        "qwen/qwen3.8-27b": QWEN38_27B_PARAMS,
        "openai/gpt-oss-120b": GPT_OSS_120B_PARAMS,
        "openai/gpt-oss-20b": GPT_OSS_20B_PARAMS,
    }
    config = configs.get(model_id)
    if config is None:
        return False
    return config.get(param) == Supports.YES


def model_configuration(model_id: str) -> ModelConfiguration:
    """Return the ModelConfiguration for the given Groq model ID.

    Args:
        model_id: the model identifier.

    Returns:
        ModelConfiguration instance.
    """
    from generation.providers.groq.model_configs import _MODEL_CONFIG_BY_ID
    return _MODEL_CONFIG_BY_ID.get(model_id, _MODEL_CONFIG_BY_ID["qwen/qwen3.8-27b"])