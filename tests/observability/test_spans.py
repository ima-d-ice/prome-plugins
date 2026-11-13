"""Span creation, nesting, attributes, and error recording (in-memory SDK)."""
from __future__ import annotations

import pytest

from observability import operation_span
from observability.contracts.models import ErrorCategory
from observability.tracing import SpanNames
from observability.tracing.spans import Attr, categorize_exception


def _span_names(spans):
    return [span.name for span in spans]


def test_simple_span_recorded(memory_otel):
    with operation_span(component="ingestion", operation="parse"):
        pass
    spans = memory_otel.span_exporter.get_finished_spans()
    names = _span_names(spans)
    assert SpanNames.INGESTION_PARSE in names


def test_nested_spans_share_trace_and_parent_linkage(memory_otel):
    with operation_span(component="ingestion", operation="load"):
        with operation_span(component="ingestion", operation="parse"):
            pass
    spans = memory_otel.span_exporter.get_finished_spans()
    by_name = {span.name: span for span in spans}
    outer_span = by_name[SpanNames.INGESTION_LOAD]
    inner_span = by_name[SpanNames.INGESTION_PARSE]
    assert inner_span.context.trace_id == outer_span.context.trace_id
    assert inner_span.parent.span_id == outer_span.context.span_id


def test_request_id_is_stable_across_nested_operations(memory_otel):
    with operation_span(component="a", operation="one") as outer:
        with operation_span(component="b", operation="two") as inner:
            assert outer.trace_context is not None
            assert inner.trace_context is not None
            assert inner.trace_context.request_id == outer.trace_context.request_id
    spans = memory_otel.span_exporter.get_finished_spans()
    request_ids = {span.attributes.get("request_id") for span in spans}
    assert len(request_ids) == 1
    assert None not in request_ids


def test_attributes_are_truncated_and_forbidden_keys_dropped(memory_otel):
    with operation_span(component="ingestion", operation="parse") as span:
        span.set_attribute("model", "qwen3")
        span.set_attribute("document_id", "secret-doc-123")
        span.set_attribute("query", "full user query text")
        span.set_attribute("long_value", "x" * 5000)
    spans = memory_otel.span_exporter.get_finished_spans()
    attributes = spans[-1].attributes
    assert attributes["model"] == "qwen3"
    assert "document_id" not in attributes
    assert "query" not in attributes
    assert len(attributes["long_value"]) <= 512


def test_exception_marks_span_with_category_and_redaction(memory_otel):
    with pytest.raises(ValueError):
        with operation_span(component="ingestion", operation="parse"):
            raise ValueError("bad input api_key=sk-abc123456789secret")
    spans = memory_otel.span_exporter.get_finished_spans()
    span = spans[-1]
    assert span.attributes[Attr.ERROR_CATEGORY] == ErrorCategory.VALIDATION
    assert span.status.status_code.name == "ERROR"
    # exception event message must not contain the secret
    events = span.events
    assert events
    assert "sk-abc123456789secret" not in str(events)


def test_categorize_exception_mapping():
    assert categorize_exception(TimeoutError("slow")) == ErrorCategory.TIMEOUT
    assert categorize_exception(PermissionError("denied")) == ErrorCategory.AUTHORIZATION
    assert categorize_exception(ValueError("bad")) == ErrorCategory.VALIDATION
    assert categorize_exception(FileNotFoundError("missing")) == ErrorCategory.NOT_FOUND
    assert categorize_exception(RuntimeError("mystery")) == ErrorCategory.UNKNOWN


def test_span_scope_outcome_attribute(memory_otel):
    with operation_span(component="ingestion", operation="parse"):
        pass
    spans = memory_otel.span_exporter.get_finished_spans()
    outcome = spans[-1].attributes.get("outcome")
    assert outcome == "ok"


def test_outcome_error_on_exception(memory_otel):
    with pytest.raises(RuntimeError):
        with operation_span(component="ingestion", operation="parse"):
            raise RuntimeError("boom")
    span = memory_otel.span_exporter.get_finished_spans()[-1]
    assert span.attributes.get("outcome") == "error"
