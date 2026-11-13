"""Trace and request correlation identifiers.

Provides dependency-free identifier generators plus a context-local
holder for the current :class:`TraceContext`. Nested spans in the same
thread (and across contextvars-aware async boundaries) observe the same
trace identity.
"""
from __future__ import annotations

import secrets
import uuid
from contextvars import ContextVar, Token

from ..contracts.models import (
    TraceContext,
    is_valid_span_id,
    is_valid_trace_id,
)

_current_context: ContextVar[TraceContext | None] = ContextVar(
    "prome_rag_trace_context", default=None
)
_current_request_id: ContextVar[str | None] = ContextVar(
    "prome_rag_request_id", default=None
)


def new_trace_id() -> str:
    """Return a fresh 128-bit trace identifier (32 lowercase hex chars)."""
    return secrets.token_hex(16)


def new_span_id() -> str:
    """Return a fresh 64-bit span identifier (16 lowercase hex chars)."""
    return secrets.token_hex(8)


def new_request_id() -> str:
    """Return a fresh application-level request correlation identifier."""
    return uuid.uuid4().hex


def current_trace_context() -> TraceContext | None:
    """Return the active trace context, or None when untraced."""
    return _current_context.get()


def current_request_id() -> str | None:
    """Return the active request id (context-local), or None."""
    return _current_request_id.get()


def set_current_trace_context(context: TraceContext | None) -> Token:
    """Set the active trace context; returns a token for context_attach/detach."""
    ctx_token = _current_context.set(context)
    _current_request_id.set(context.request_id if context else None)
    return ctx_token


def reset_current_trace_context(token: Token) -> None:
    _current_context.reset(token)
    context = _current_context.get()
    _current_request_id.set(context.request_id if context else None)


def attach_request_id(request_id: str) -> Token:
    """Attach only a request id; useful for entry points without a trace."""
    return _current_request_id.set(request_id)


def detach_request_id(token: Token) -> None:
    _current_request_id.reset(token)


def root_context(request_id: str | None = None) -> TraceContext:
    """Create a brand-new root trace context for an entry point."""
    return TraceContext(
        trace_id=new_trace_id(),
        span_id=new_span_id(),
        parent_span_id=None,
        request_id=request_id or new_request_id(),
    )


def child_context(
    parent: TraceContext,
    span_id: str | None = None,
    request_id: str | None = None,
) -> TraceContext:
    """Derive a nested context sharing the parent's trace and request id."""
    if not is_valid_trace_id(parent.trace_id):
        raise ValueError("parent context carries an invalid trace_id")
    return parent.child(span_id or new_span_id(), request_id=request_id)


def context_from_ids(
    trace_id: str,
    span_id: str,
    *,
    parent_span_id: str | None = None,
    request_id: str | None = None,
) -> TraceContext:
    """Build a validated context from wire values (e.g. after extraction)."""
    if not is_valid_trace_id(trace_id) or not is_valid_span_id(span_id):
        from ..contracts.exceptions import TraceContextError

        raise TraceContextError("invalid trace or span identifier supplied")
    return TraceContext(
        trace_id=trace_id.lower(),
        span_id=span_id.lower(),
        parent_span_id=parent_span_id.lower() if parent_span_id else None,
        request_id=request_id,
    )
