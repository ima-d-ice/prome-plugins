"""Prome RAG observability: OpenTelemetry-based tracing, metrics, structured logging.

Importing this package has no side effects: no providers are created, no
exporters are attached, and no OpenTelemetry globals are registered
until :func:`init_observability` is called. With the default
configuration every instrument is a no-op, so instrumenting a component
is always safe.

Quick start::

    from observability import init_observability, shutdown_observability

    runtime = init_observability()          # reads PROMERAG_* environment
    ...
    shutdown_observability()                # flushes and detaches everything
"""
from __future__ import annotations

__version__ = "0.1.0"

from .configuration import (
    ENV_PREFIX,
    EXPORTER_CONSOLE,
    EXPORTER_NONE,
    EXPORTER_OTLP,
    ObservabilityConfig,
    ObservabilityRuntime,
    get_config,
    get_runtime,
    init_observability,
    is_enabled,
    metrics_active,
    shutdown_observability,
)
from .contracts import (
    ALLOWED_LABEL_KEYS,
    FORBIDDEN_LABEL_KEYS,
    MAX_LABEL_VALUE_LENGTH,
    CardinalityViolationError,
    ConfigurationError,
    ExportError,
    InstrumentationError,
    MetricsError,
    NoOpModeError,
    ObservabilityError,
    RedactionError,
    RedactionRule,
    SpanKind,
    StatusCode,
    TraceContext,
    TraceContextError,
    TracingError,
)
from .instrumentation import (
    active_request,
    operation_span,
    timed_operation,
    traced,
    traced_operation,
)
from .logging import (
    HumanFormatter,
    JsonFormatter,
    Redactor,
    bind_request_id,
    get_logger,
    redact_mapping,
    redact_text,
    redact_value,
    unbind_request_id,
)
from .metrics import (
    MetricNames,
    MetricRegistry,
    default_registry,
    increment_active_requests,
    record_generation_usage,
    record_runtime_snapshot,
    sanitize_metric_labels,
    set_queue_length,
)
from .tracing import (
    Attr,
    SpanNames,
    SpanScope,
    Tracer,
    child_context,
    current_trace_context,
    extract_trace_context,
    get_tracer,
    inject_trace_context,
    root_context,
)

__all__ = [
    "ALLOWED_LABEL_KEYS",
    "ENV_PREFIX",
    "EXPORTER_CONSOLE",
    "EXPORTER_NONE",
    "EXPORTER_OTLP",
    "FORBIDDEN_LABEL_KEYS",
    "MAX_LABEL_VALUE_LENGTH",
    "Attr",
    "CardinalityViolationError",
    "ConfigurationError",
    "ExportError",
    "HumanFormatter",
    "InstrumentationError",
    "JsonFormatter",
    "MetricNames",
    "MetricRegistry",
    "MetricsError",
    "NoOpModeError",
    "ObservabilityConfig",
    "ObservabilityError",
    "ObservabilityRuntime",
    "RedactionError",
    "RedactionRule",
    "Redactor",
    "SpanKind",
    "SpanNames",
    "SpanScope",
    "StatusCode",
    "TraceContext",
    "TraceContextError",
    "Tracer",
    "TracingError",
    "__version__",
    "active_request",
    "bind_request_id",
    "child_context",
    "current_trace_context",
    "default_registry",
    "extract_trace_context",
    "get_config",
    "get_logger",
    "get_runtime",
    "get_tracer",
    "increment_active_requests",
    "init_observability",
    "inject_trace_context",
    "is_enabled",
    "metrics_active",
    "operation_span",
    "record_generation_usage",
    "record_runtime_snapshot",
    "redact_mapping",
    "redact_text",
    "redact_value",
    "root_context",
    "sanitize_metric_labels",
    "set_queue_length",
    "shutdown_observability",
    "timed_operation",
    "traced",
    "traced_operation",
    "unbind_request_id",
]
