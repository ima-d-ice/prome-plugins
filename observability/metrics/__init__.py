"""Metrics for the observability ecosystem.

Instruments come in three modes selected at init time: OpenTelemetry
(when the SDK is configured), an in-memory fallback (when observability
is enabled but the SDK is unavailable), and a no-op (default). Metric
labels are always passed through the cardinality policy in
``observability.metrics._labels``.
"""
from __future__ import annotations

from ._labels import (
    ALLOWED_LABEL_KEYS,
    FORBIDDEN_LABEL_KEYS,
    MAX_LABEL_VALUE_LENGTH,
    sanitize_metric_labels,
)
from .counters import Counter, UpDownCounter
from .gauges import Gauge
from .histograms import Histogram
from .rag_metrics import (
    MetricNames,
    increment_active_requests,
    record_generation_usage,
    record_runtime_snapshot,
    set_queue_length,
)
from .registry import MetricRegistry, default_registry

__all__ = [
    "ALLOWED_LABEL_KEYS",
    "FORBIDDEN_LABEL_KEYS",
    "MAX_LABEL_VALUE_LENGTH",
    "Counter",
    "Gauge",
    "Histogram",
    "MetricNames",
    "MetricRegistry",
    "UpDownCounter",
    "default_registry",
    "increment_active_requests",
    "record_generation_usage",
    "record_runtime_snapshot",
    "sanitize_metric_labels",
    "set_queue_length",
]
