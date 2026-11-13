"""Token usage accounting: only provider-reported values are recorded."""
from __future__ import annotations

from observability.metrics.rag_metrics import record_generation_usage


def test_openai_style_usage(memory_otel):
    resolved = record_generation_usage(
        {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140},
        provider="groq",
        model="llama-3.3-70b",
    )
    assert resolved == {"input_tokens": 100.0, "output_tokens": 40.0, "total_tokens": 140.0}


def test_ollama_style_usage(memory_otel):
    resolved = record_generation_usage(
        {"prompt_eval_count": 12, "eval_count": 8},
        provider="ollama",
        model="qwen3:0.6b",
    )
    # total is synthesized as input + output
    assert resolved == {"input_tokens": 12.0, "output_tokens": 8.0, "total_tokens": 20.0}


def test_nested_usage_dict(memory_otel):
    resolved = record_generation_usage(
        {"usage": {"prompt_tokens": 3, "completion_tokens": 7}},
        provider="groq",
        model="m",
    )
    assert resolved["input_tokens"] == 3.0
    assert resolved["output_tokens"] == 7.0


def test_partial_usage_records_only_present_keys(memory_otel):
    resolved = record_generation_usage({"completion_tokens": 7}, provider="groq", model="m")
    assert resolved == {"output_tokens": 7.0}


def test_empty_usage_records_nothing(memory_otel):
    assert record_generation_usage(None, provider="p", model="m") == {}
    assert record_generation_usage({}, provider="p", model="m") == {}


def test_malformed_usage_values_are_ignored(memory_otel):
    resolved = record_generation_usage(
        {"prompt_tokens": "not-a-number", "eval_count": 5},
        provider="p",
        model="m",
    )
    assert resolved == {"output_tokens": 5.0}


def test_metrics_contain_usage_tokens(memory_otel):
    from observability.metrics import MetricNames

    from .conftest import metric_data_points

    record_generation_usage(
        {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140},
        provider="groq",
        model="llama-3.3-70b",
    )
    data = memory_otel.metric_reader.get_metrics_data()
    totals = {
        name: sum(point.value for _, _, point in metric_data_points(data, name))
        for name in (
            MetricNames.GENERATION_INPUT_TOKENS,
            MetricNames.GENERATION_OUTPUT_TOKENS,
            MetricNames.GENERATION_TOTAL_TOKENS,
        )
    }
    assert totals[MetricNames.GENERATION_INPUT_TOKENS] == 100
    assert totals[MetricNames.GENERATION_OUTPUT_TOKENS] == 40
    assert totals[MetricNames.GENERATION_TOTAL_TOKENS] == 140
