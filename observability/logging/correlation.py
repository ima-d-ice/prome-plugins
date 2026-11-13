"""Correlation helpers tying application logs to traces.

Reads the same context-local identifiers that the tracing layer sets, so
a log line emitted inside ``with operation_span(...)`` carries the active
request/trace/span ids without any manual plumbing. The import of the
tracing package is deliberately lazy to avoid import-time cycles.
"""
from __future__ import annotations

import logging


def get_request_id() -> str | None:
    try:
        from ..tracing.context import current_request_id

        return current_request_id()
    except Exception:
        return None


def get_trace_id() -> str | None:
    try:
        from ..tracing.context import current_trace_context

        context = current_trace_context()
        return context.trace_id if context else None
    except Exception:
        return None


def get_span_id() -> str | None:
    try:
        from ..tracing.context import current_trace_context

        context = current_trace_context()
        return context.span_id if context else None
    except Exception:
        return None


def correlation_fields() -> dict[str, str]:
    fields: dict[str, str] = {}
    request_id = get_request_id()
    if request_id:
        fields["request_id"] = request_id
    trace_id = get_trace_id()
    if trace_id:
        fields["trace_id"] = trace_id
    span_id = get_span_id()
    if span_id:
        fields["span_id"] = span_id
    return fields


class CorrelationFilter(logging.Filter):
    """Injects request_id/trace_id/span_id into every LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in correlation_fields().items():
            if not getattr(record, key, None):
                setattr(record, key, value)
        return True


def bind_request_id(request_id: str | None = None):
    """Bind a request id for the current context (generates one if omitted)."""
    from ..tracing.context import attach_request_id, new_request_id

    return attach_request_id(request_id or new_request_id())


def unbind_request_id(token) -> None:
    from ..tracing.context import detach_request_id

    detach_request_id(token)
