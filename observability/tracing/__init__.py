"""Distributed tracing for the observability ecosystem.

Importing this package never creates a TracerProvider or registers
anything globally. Call :func:`observability.init_observability` to activate exporters.
"""
from __future__ import annotations

from ..contracts.models import SpanKind, StatusCode, TraceContext
from .context import (
    attach_request_id,
    child_context,
    context_from_ids,
    current_request_id,
    current_trace_context,
    detach_request_id,
    new_request_id,
    new_span_id,
    new_trace_id,
    reset_current_trace_context,
    root_context,
    set_current_trace_context,
)
from .propagation import (
    BAGGAGE_HEADER,
    REQUEST_ID_HEADER,
    TRACEPARENT_HEADER,
    TRACESTATE_HEADER,
    extract_trace_context,
    inject_trace_context,
)
from .spans import Attr, SpanNames, SpanScope, categorize_exception, sanitize_attributes
from .tracer import Tracer, get_tracer

__all__ = [
    "BAGGAGE_HEADER",
    "REQUEST_ID_HEADER",
    "TRACEPARENT_HEADER",
    "TRACESTATE_HEADER",
    "Attr",
    "SpanKind",
    "SpanNames",
    "SpanScope",
    "StatusCode",
    "TraceContext",
    "Tracer",
    "attach_request_id",
    "categorize_exception",
    "child_context",
    "context_from_ids",
    "current_request_id",
    "current_trace_context",
    "detach_request_id",
    "extract_trace_context",
    "get_tracer",
    "inject_trace_context",
    "new_request_id",
    "new_span_id",
    "new_trace_id",
    "reset_current_trace_context",
    "root_context",
    "sanitize_attributes",
    "set_current_trace_context",
]
