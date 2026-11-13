"""Reusable context managers instrumenting a single operation.

``operation_span`` is the primitive most components should use; the span
is named ``<component>.<operation>`` unless an explicit ``name`` is
given. All context managers are no-cost-when-disabled.
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from typing import Any

from ..contracts.models import SpanKind
from ..metrics.rag_metrics import increment_active_requests
from ..tracing import get_tracer
from ..tracing.spans import SpanScope


def operation_span(
    name: str | None = None,
    *,
    component: str | None = None,
    operation: str | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Mapping[str, Any] | None = None,
    request_id: str | None = None,
) -> SpanScope:
    """Open a span for one unit of work.

    Resolution order for the span name: explicit ``name``, else
    ``f"{component}.{operation}"``, else ``"component.operation"`` fallback
    of ``operation`` alone. Component is also set as a span attribute.
    """
    span_name = name or (f"{component}.{operation}" if component and operation else operation) or name
    if not span_name:
        raise ValueError("operation_span requires either 'name' or 'operation' (with optional 'component')")
    merged: dict[str, Any] = {}
    if component:
        merged["component"] = component
    if operation:
        merged["operation"] = operation
    if attributes:
        merged.update(attributes)
    return get_tracer("prome-rag").start_span(
        span_name,
        kind=kind,
        attributes=merged,
        request_id=request_id,
    )


@contextlib.contextmanager
def active_request(*, component: str) -> Iterator[None]:
    """Track an active top-level request with the runtime up/down counter."""
    increment_active_requests(1, component=component)
    try:
        yield
    finally:
        increment_active_requests(-1, component=component)


@contextlib.contextmanager
def timed_operation(
    name: str,
    *,
    component: str,
    operation: str | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[SpanScope]:
    """Alias for operation_span kept for readability at call sites."""
    with operation_span(
        name,
        component=component,
        operation=operation or name.rsplit(".", 1)[-1],
        kind=kind,
        attributes=attributes,
    ) as span:
        yield span
