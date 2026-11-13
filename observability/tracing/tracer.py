"""Tracer facade used by all instrumentation code.

Resolution order at span start:

1. Runtime initialized with ``PROMERAG_OBSERVABILITY_ENABLED=true`` and
   the OpenTelemetry SDK available -> real SDK span (parentage via the
   OpenTelemetry context, provider from the initialized runtime only;
   the global proxy provider is never touched).
2. Otherwise -> no-op span that still maintains trace/correlation
   context so logging and propagation keep working.

Call :func:`get_tracer` anywhere; it never raises.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from ..configuration import get_runtime
from ..contracts.models import SpanKind, TraceContext
from .context import (
    current_trace_context,
    new_request_id,
    new_span_id,
    new_trace_id,
)
from .spans import SpanScope

logger = logging.getLogger("prome_rag.tracing")

_OTEL_KIND_MAP = {
    SpanKind.INTERNAL: "INTERNAL",
    SpanKind.SERVER: "SERVER",
    SpanKind.CLIENT: "CLIENT",
    SpanKind.PRODUCER: "PRODUCER",
    SpanKind.CONSUMER: "CONSUMER",
}


class Tracer:
    """Lightweight tracer bound to the initialized runtime (if any)."""

    def __init__(self, name: str = "prome-rag") -> None:
        self.name = name

    def start_span(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Mapping[str, Any] | None = None,
        request_id: str | None = None,
    ) -> SpanScope:
        return SpanScope(
            self, name, kind=kind, attributes=attributes, request_id=request_id
        )

    # -- internals used by SpanScope ---------------------------------------

    def _start_span(
        self,
        name: str,
        *,
        kind: SpanKind,
        attributes: Mapping[str, Any],
        request_id: str | None,
    ) -> tuple[Any, TraceContext, Any]:
        runtime = get_runtime()
        parent = current_trace_context()
        request_id = request_id or (parent.request_id if parent else None) or new_request_id()

        otel_span = self._try_start_otel_span(runtime, name, kind=kind, attributes=attributes)
        if otel_span is not None:
            ctx = otel_span.get_span_context()
            trace_id = format(ctx.trace_id, "032x")
            span_id = format(ctx.span_id, "016x")
            parent_id = parent.span_id if parent else None
            otel_token = self._attach_otel_span(otel_span)
            return (
                otel_span,
                TraceContext(
                    trace_id=trace_id,
                    span_id=span_id,
                    parent_span_id=parent_id,
                    request_id=request_id,
                    baggage=dict(parent.baggage) if parent else {},
                ),
                otel_token,
            )

        # No-op path: local identifiers keep propagation/correlation intact.
        if parent is not None:
            context = TraceContext(
                trace_id=parent.trace_id,
                span_id=new_span_id(),
                parent_span_id=parent.span_id,
                request_id=request_id,
                baggage=dict(parent.baggage),
            )
        else:
            context = TraceContext(
                trace_id=new_trace_id(),
                span_id=new_span_id(),
                parent_span_id=None,
                request_id=request_id,
            )
        return None, context, None

    def _try_start_otel_span(
        self,
        runtime: Any,
        name: str,
        *,
        kind: SpanKind,
        attributes: Mapping[str, Any],
    ) -> Any | None:
        if runtime is None or not runtime.enabled or runtime.tracer_provider is None:
            return None
        try:
            from opentelemetry import context as otel_context
            from opentelemetry.trace import SpanKind as OTelSpanKind

            otel_tracer = runtime.tracer_provider.get_tracer(self.name)
            otel_kind = getattr(OTelSpanKind, _OTEL_KIND_MAP[kind])
            parent_ctx = otel_context.get_current()
            return otel_tracer.start_span(
                name,
                kind=otel_kind,
                attributes=dict(attributes),
                context=parent_ctx,
            )
        except Exception as exc:
            logger.debug("otel span creation failed; using no-op span: %s", exc)
            return None

    @staticmethod
    def _attach_otel_span(otel_span: Any) -> Any:
        try:
            from opentelemetry import context as otel_context
            from opentelemetry.trace import set_span_in_context

            return otel_context.attach(set_span_in_context(otel_span))
        except Exception as exc:
            logger.debug("failed to attach otel span to context: %s", exc)
            return None


_tracers: dict[str, Tracer] = {}


def get_tracer(name: str = "prome-rag") -> Tracer:
    """Return a cached :class:`Tracer` for ``name``; never raises."""
    tracer = _tracers.get(name)
    if tracer is None:
        tracer = Tracer(name)
        _tracers[name] = tracer
    return tracer
