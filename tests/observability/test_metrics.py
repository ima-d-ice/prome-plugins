"""Metric recording via in-memory reader and label policy enforcement."""
from __future__ import annotations

import pytest

from observability.contracts.exceptions import CardinalityViolationError
from observability.contracts.models import MAX_LABEL_VALUE_LENGTH
from observability.metrics import MetricNames
from observability.metrics._labels import sanitize_metric_labels
from observability.metrics.registry import MetricRegistry

from .conftest import metric_data_points

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_counter_records_to_otel(memory_otel):
    from observability.metrics import default_registry

    registry = default_registry()
    registry.counter(MetricNames.INGESTION_DOCUMENTS_PARSED, unit="{document}").add(
        2, {"component": "parser", "outcome": "ok"}
    )
    metrics_data = memory_otel.metric_reader.get_metrics_data()
    points = metric_data_points(metrics_data, MetricNames.INGESTION_DOCUMENTS_PARSED)
    assert points, "expected data points for documents_parsed"
    attrs, point = points[0][1], points[0][2]
    assert attrs == {"component": "parser", "outcome": "ok"}
    assert point.value == 2


def test_gauge_records(memory_otel):
    from observability.metrics import default_registry

    registry = default_registry()
    registry.gauge(MetricNames.RUNTIME_QUEUE_LENGTH, unit="{item}").set(
        42, {"component": "loader"}
    )
    metrics_data = memory_otel.metric_reader.get_metrics_data()
    points = metric_data_points(metrics_data, MetricNames.RUNTIME_QUEUE_LENGTH)
    assert points
    assert points[0][1] == {"component": "loader"}
    assert points[0][2].value == 42


def test_histogram_records(memory_otel):
    from observability.metrics import default_registry

    registry = default_registry()
    registry.histogram(MetricNames.RETRIEVAL_LATENCY, unit="s").record(
        0.25, {"component": "retriever", "retrieval_strategy": "hybrid"}
    )
    metrics_data = memory_otel.metric_reader.get_metrics_data()
    points = metric_data_points(metrics_data, MetricNames.RETRIEVAL_LATENCY)
    assert points
    assert points[0][1].get("retrieval_strategy") == "hybrid"


def test_registry_caches_instruments(memory_otel):
    from observability.metrics import default_registry

    registry = default_registry()
    first = registry.counter(MetricNames.INGESTION_CHUNKS_CREATED)
    second = registry.counter(MetricNames.INGESTION_CHUNKS_CREATED)
    assert first is second


def test_up_down_counter_for_active_requests(memory_otel):
    from observability.metrics import increment_active_requests

    increment_active_requests(1, component="orchestration")
    increment_active_requests(-1, component="orchestration")
    metrics_data = memory_otel.metric_reader.get_metrics_data()

    points = metric_data_points(metrics_data, MetricNames.RUNTIME_ACTIVE_REQUESTS)
    assert points
    # sum points: two deltas
    assert sum(point.value for _, _, point in points) == 0


def test_label_sanitizer_drops_forbidden_keys():
    labels = {
        "component": "parser",
        "user_id": "12345",
        "request_id": "req-1",
        "trace_id": "t" * 32,
        "model": "qwen3",
    }
    cleaned = sanitize_metric_labels(labels)
    assert cleaned == {"component": "parser", "model": "qwen3"}


def test_label_sanitizer_drops_unknown_keys():
    cleaned = sanitize_metric_labels({"component": "a", "surprise": "value"})
    assert cleaned == {"component": "a"}


def test_label_sanitizer_truncates_long_values():
    cleaned = sanitize_metric_labels({"model": "m" * 500})
    assert len(cleaned["model"]) <= MAX_LABEL_VALUE_LENGTH


def test_label_sanitizer_strict_mode_raises():
    with pytest.raises(CardinalityViolationError):
        sanitize_metric_labels({"user_id": "123"}, strict=True)


def test_label_sanitizer_coerces_scalar_values():
    cleaned = sanitize_metric_labels({"component": 123})  # type: ignore[dict-item]
    assert cleaned == {"component": "123"}


def test_fallback_registry_snapshot():
    registry = MetricRegistry(enabled=True, fallback=True)
    counter = registry.counter("prome_rag.test.counter")
    counter.inc({"component": "a"})
    counter.add(4, {"component": "a"})
    registry.gauge("prome_rag.test.gauge").set(9.5, {"component": "a"})
    registry.histogram("prome_rag.test.histogram").record(0.5, {"component": "a"})
    snap = registry.snapshot()
    assert snap["counter"]["prome_rag.test.counter"][0]["labels"] == {"component": "a"}
    assert snap["counter"]["prome_rag.test.counter"][0]["value"] == 5.0
    assert snap["gauge"]["prome_rag.test.gauge"][0]["value"] == 9.5
    assert snap["histogram"]["prome_rag.test.histogram"][0]["count"] == 1
