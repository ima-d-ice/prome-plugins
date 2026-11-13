"""Exceptions for the observability ecosystem."""

from __future__ import annotations


class ObservabilityError(Exception):
    """Base exception for all observability-related errors."""


class ConfigurationError(ObservabilityError):
    """Raised when observability configuration is invalid."""


class TracingError(ObservabilityError):
    """Raised when tracing operations fail."""


class MetricsError(ObservabilityError):
    """Raised when metric recording fails."""


class ExportError(ObservabilityError):
    """Raised when exporter operations fail."""


class RedactionError(ObservabilityError):
    """Raised when redaction operations fail."""


class NoOpModeError(ObservabilityError):
    """Raised when attempting operation in no-op mode."""


class InstrumentationError(ObservabilityError):
    """Raised when instrumentation operations fail."""


class CardinalityViolationError(ObservabilityError):
    """Raised when high-cardinality data is used as a metric label."""


class TraceContextError(ObservabilityError):
    """Raised when trace context operations fail."""
