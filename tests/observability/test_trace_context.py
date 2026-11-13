"""Trace context creation, validation, and context propagation helpers."""
from __future__ import annotations

import pytest

from observability.contracts.exceptions import TraceContextError
from observability.contracts.models import (
    TraceContext,
    is_valid_span_id,
    is_valid_trace_id,
)
from observability.tracing.context import (
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


def test_new_ids_format():
    trace_id = new_trace_id()
    span_id = new_span_id()
    request_id = new_request_id()
    assert is_valid_trace_id(trace_id)
    assert is_valid_span_id(span_id)
    assert len(request_id) == 32
    assert is_valid_trace_id(new_trace_id())


def test_root_context_carries_request_id():
    context = root_context()
    assert is_valid_trace_id(context.trace_id)
    assert context.parent_span_id is None
    assert context.request_id


def test_child_context_shares_trace_and_request_id():
    parent = root_context()
    child = child_context(parent)
    assert child.trace_id == parent.trace_id
    assert child.request_id == parent.request_id
    assert child.parent_span_id == parent.span_id
    assert child.span_id != parent.span_id


def test_trace_context_validation_rejects_bad_ids():
    with pytest.raises(ValueError):
        TraceContext(trace_id="xyz", span_id="0123456789abcdef")
    with pytest.raises(ValueError):
        TraceContext(trace_id="0" * 32, span_id="0123456789abcdef")
    with pytest.raises(ValueError):
        TraceContext(trace_id="0" * 31, span_id="0123456789abcdef")


def test_context_from_ids_validates_and_normalizes():
    context = context_from_ids("A" * 32, "B" * 16, request_id="req-1")
    assert context.trace_id == "a" * 32
    assert context.span_id == "b" * 16
    assert context.request_id == "req-1"
    with pytest.raises(TraceContextError):
        context_from_ids("not-hex", "B" * 16)


def test_context_local_current_context():
    context = root_context()
    token = set_current_trace_context(context)
    try:
        assert current_trace_context() == context
        assert current_request_id() == context.request_id
    finally:
        reset_current_trace_context(token)
    assert current_trace_context() is None


def test_attach_detach_request_id():
    token = attach_request_id("top-level-request")
    try:
        assert current_request_id() == "top-level-request"
    finally:
        detach_request_id(token)
    assert current_request_id() is None


def test_child_of_child_nesting():
    root = root_context()
    level1 = child_context(root)
    level2 = child_context(level1)
    assert level2.trace_id == root.trace_id
    assert level2.parent_span_id == level1.span_id
    assert level1.parent_span_id == root.span_id
