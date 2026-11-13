"""No-op mode: default state, before/without init, and when disabled."""
from __future__ import annotations

from observability import (
    init_observability,
    is_enabled,
    metrics_active,
    operation_span,
    record_generation_usage,
    shutdown_observability,
)
from observability.configuration import ObservabilityConfig
from observability.metrics import MetricRegistry


def test_default_state_is_noop():
    assert is_enabled() is False
    assert metrics_active() is False


def test_init_disabled_is_noop():
    runtime = init_observability(ObservabilityConfig(enabled=False))
    assert runtime.enabled is False
    with operation_span(component="ingestion", operation="parse") as span:
        span.set_attribute("component", "ingestion")
        span.add_event("checkpoint")
        with operation_span(component="parser", operation="parse") as nested:
            nested.set_attribute("model", "m")
    shutdown_observability()


def test_init_without_sdk_falls_back_to_noop(monkeypatch):
    from observability import configuration

    monkeypatch.setattr(configuration, "otel_sdk_available", lambda: False)
    runtime = init_observability(ObservabilityConfig(enabled=True))
    assert runtime.otel_active is False
    assert runtime.enabled is False


def test_default_registry_disabled_is_noop():
    registry = MetricRegistry(enabled=False)
    counter = registry.counter("prome_rag.test.noop", unit="{one}")
    counter.inc()
    gauge = registry.gauge("prome_rag.test.noop_gauge", unit="By")
    gauge.set(12.0)
    assert registry.snapshot() == {}


def test_record_generation_usage_before_init_is_safe():
    resolved = record_generation_usage(
        {"prompt_tokens": 10, "completion_tokens": 5}, provider="ollama", model="qwen3"
    )
    assert resolved == {"input_tokens": 10.0, "output_tokens": 5.0, "total_tokens": 15.0}


def test_double_init_and_shutdown_are_idempotent():
    init_observability(ObservabilityConfig(enabled=False))
    init_observability(ObservabilityConfig(enabled=False))
    shutdown_observability()
    shutdown_observability()
    assert is_enabled() is False
