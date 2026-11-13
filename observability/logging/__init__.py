"""Structured logging for the observability ecosystem.

Built on Python's :mod:`logging`; this module does not replace it.
"""
from __future__ import annotations

from .correlation import (
    CorrelationFilter,
    bind_request_id,
    correlation_fields,
    get_request_id,
    get_span_id,
    get_trace_id,
    unbind_request_id,
)
from .redaction import (
    Redactor,
    configure_default_redactor,
    get_default_redactor,
    redact_mapping,
    redact_text,
    redact_value,
    reset_default_redactor,
)
from .structured import (
    LOGGER_NAMESPACE,
    HumanFormatter,
    JsonFormatter,
    configure_logging,
    get_logger,
)

__all__ = [
    "LOGGER_NAMESPACE",
    "CorrelationFilter",
    "HumanFormatter",
    "JsonFormatter",
    "Redactor",
    "bind_request_id",
    "configure_default_redactor",
    "configure_logging",
    "correlation_fields",
    "get_default_redactor",
    "get_logger",
    "get_request_id",
    "get_span_id",
    "get_trace_id",
    "redact_mapping",
    "redact_text",
    "redact_value",
    "reset_default_redactor",
    "unbind_request_id",
]
