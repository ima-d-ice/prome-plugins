"""Decorators for tracing functions without touching their bodies."""
from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable
from typing import Any

from ..contracts.models import SpanKind
from .context_managers import operation_span


def traced(
    _fn: Callable | None = None,
    *,
    name: str | None = None,
    component: str | None = None,
    operation: str | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Any | None = None,
) -> Callable:
    """Trace a synchronous or asynchronous callable.

    Usage::

        @traced(component="parser", operation="parse")
        def parse(doc): ...

        @traced("custom.name")
        def helper(...): ...

    Works with or without OpenTelemetry; when observability is disabled the
    decorator adds only a cheap context-manager pass-through.
    """

    def decorate(fn: Callable) -> Callable:
        span_name = name or (f"{component}.{operation}" if component and operation else fn.__qualname__)
        op = operation or fn.__name__

        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with operation_span(
                    span_name,
                    component=component,
                    operation=op,
                    kind=kind,
                    attributes=attributes,
                ):
                    return await fn(*args, **kwargs)

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with operation_span(
                span_name,
                component=component,
                operation=op,
                kind=kind,
                attributes=attributes,
            ):
                return fn(*args, **kwargs)

        return sync_wrapper

    if callable(_fn):
        return decorate(_fn)
    return decorate


def traced_operation(
    component: str,
    operation: str | None = None,
    *,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Any | None = None,
) -> Callable:
    """Specialization of :func:`traced` with an explicit component."""

    def decorate(fn: Callable) -> Callable:
        return traced(fn, name=None, component=component, operation=operation or fn.__name__, kind=kind, attributes=attributes)

    return decorate
