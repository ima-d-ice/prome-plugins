"""Shared fixtures for observability tests.

No test in this suite requires a network collector; in-memory exporters
and readers from the OpenTelemetry SDK are used instead, and are skipped
automatically when the SDK is not importable.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from observability.configuration import ObservabilityConfig, shutdown_observability


@pytest.fixture(autouse=True)
def clean_observability_state(monkeypatch):
    """Guarantee every test starts with no PROMERAG_* env and no runtime."""
    for key in list(os.environ):
        if key.startswith("PROMERAG_"):
            monkeypatch.delenv(key, raising=False)
    yield
    shutdown_observability()


@pytest.fixture
def memory_otel():
    """Init the observability stack against in-memory SDK providers."""
    pytest.importorskip("opentelemetry.sdk.trace", reason="opentelemetry-sdk not installed")

    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from observability.configuration import init_observability

    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])

    runtime = init_observability(
        ObservabilityConfig(enabled=True, exporter="none"),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )
    try:
        yield SimpleNamespace(
            runtime=runtime,
            span_exporter=span_exporter,
            metric_reader=metric_reader,
        )
    finally:
        shutdown_observability()


def metric_data_points(metrics_data, name):
    """Flatten scope/resource metrics into (name, attributes, point) tuples."""
    points = []
    for resource_metric in getattr(metrics_data, "resource_metrics", []) or []:
        for scope_metric in getattr(resource_metric, "scope_metrics", []) or []:
            for metric in getattr(scope_metric, "metrics", []) or []:
                if metric.name != name:
                    continue
                data = metric.data
                for point in getattr(data, "data_points", []) or []:
                    attrs = dict(getattr(point, "attributes", {}) or {})
                    points.append((name, attrs, point))
    return points
