"""W3C Trace Context propagation.

Implements ``traceparent``/``tracestate``/``baggage`` injection and
extraction without requiring OpenTelemetry, so trace identity can flow
across queues, HTTP calls, and process boundaries. Invalid inbound
headers never raise; they yield ``None`` so callers start a fresh root
trace instead of crashing.
"""
from __future__ import annotations

import re
from collections.abc import MutableMapping

from ..contracts.models import TraceContext, is_valid_span_id, is_valid_trace_id
from .context import current_trace_context

TRACEPARENT_HEADER = "traceparent"
TRACESTATE_HEADER = "tracestate"
BAGGAGE_HEADER = "baggage"
REQUEST_ID_HEADER = "x-request-id"

_TRACEPARENT_RE = re.compile(
    r"^(?P<version>[0-9a-f]{2})-"
    r"(?P<trace_id>[0-9a-f]{32})-"
    r"(?P<span_id>[0-9a-f]{16})-"
    r"(?P<flags>[0-9a-f]{2})$"
)


def inject_trace_context(
    carrier: MutableMapping[str, str],
    context: TraceContext | None = None,
    *,
    sampled: bool = True,
) -> MutableMapping[str, str]:
    """Inject trace identity into ``carrier`` (e.g. HTTP headers)."""
    context = context if context is not None else current_trace_context()
    if context is None:
        return carrier
    flags = "01" if sampled else "00"
    carrier[TRACEPARENT_HEADER] = (
        f"00-{context.trace_id}-{context.span_id}-{flags}"
    )
    if context.request_id:
        carrier[REQUEST_ID_HEADER] = context.request_id
    if context.baggage:
        carrier[BAGGAGE_HEADER] = ",".join(
            f"{key}={value}" for key, value in context.baggage.items()
        )
    return carrier


def extract_trace_context(
    carrier: MutableMapping[str, str] | None,
) -> TraceContext | None:
    """Extract a validated trace context from ``carrier``; None if absent/invalid."""
    if not carrier:
        return None
    raw = None
    for key, value in carrier.items():
        if key.lower() == TRACEPARENT_HEADER:
            raw = value
            break
    if not raw:
        return None
    match = _TRACEPARENT_RE.match(raw.strip().lower())
    if not match:
        return None
    version = match.group("version")
    if version == "ff":
        return None
    trace_id = match.group("trace_id")
    span_id = match.group("span_id")
    if not is_valid_trace_id(trace_id) or not is_valid_span_id(span_id):
        return None

    request_id = None
    baggage: dict[str, str] = {}
    for key, value in carrier.items():
        lowered = key.lower()
        if lowered == REQUEST_ID_HEADER:
            request_id = value.strip() or None
        elif lowered == BAGGAGE_HEADER:
            baggage = _parse_baggage(value)

    return TraceContext(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=None,
        request_id=request_id,
        baggage=baggage,
    )


def _parse_baggage(raw: str) -> dict[str, str]:
    baggage: dict[str, str] = {}
    for part in raw.split(","):
        item = part.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            baggage[key[:128]] = value[:256]
    return baggage
