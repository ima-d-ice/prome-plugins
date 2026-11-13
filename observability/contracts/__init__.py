"""Contracts for the observability ecosystem.

Core data models, protocols, and exceptions shared by every
observability component. Importing this package has no side effects.
"""
from __future__ import annotations

from .exceptions import (
    CardinalityViolationError,
    ConfigurationError,
    ExportError,
    InstrumentationError,
    MetricsError,
    NoOpModeError,
    ObservabilityError,
    RedactionError,
    TraceContextError,
    TracingError,
)
from .models import (
    ALLOWED_LABEL_KEYS,
    FORBIDDEN_LABEL_KEYS,
    MAX_LABEL_VALUE_LENGTH,
    ErrorCategory,
    ExportConfig,
    RedactionRule,
    SpanEvent,
    SpanKind,
    SpanRecord,
    StatusCode,
    TraceContext,
    is_valid_span_id,
    is_valid_trace_id,
)
from .protocols import (
    CounterProtocol,
    ExporterProtocol,
    GaugeProtocol,
    HistogramProtocol,
    InstrumentationHandlerProtocol,
    LoggingHandlerProtocol,
    MetricsRegistryProtocol,
    TracerProtocol,
    UpDownCounterProtocol,
)

__all__ = [
    "ALLOWED_LABEL_KEYS",
    "FORBIDDEN_LABEL_KEYS",
    "MAX_LABEL_VALUE_LENGTH",
    "CardinalityViolationError",
    "ConfigurationError",
    "CounterProtocol",
    "ErrorCategory",
    "ExportConfig",
    "ExportError",
    "ExporterProtocol",
    "GaugeProtocol",
    "HistogramProtocol",
    "InstrumentationError",
    "InstrumentationHandlerProtocol",
    "LoggingHandlerProtocol",
    "MetricsError",
    "MetricsRegistryProtocol",
    "NoOpModeError",
    "ObservabilityError",
    "RedactionError",
    "RedactionRule",
    "SpanEvent",
    "SpanKind",
    "SpanRecord",
    "StatusCode",
    "TraceContext",
    "TraceContextError",
    "TracerProtocol",
    "TracingError",
    "UpDownCounterProtocol",
    "is_valid_span_id",
    "is_valid_trace_id",
]
