"""Tests for the Observability integration adapter."""

from __future__ import annotations

import pytest

from integrations.observability_integration import (
    ensure_observability,
    shutdown_observability,
    set_request_context,
    get_request_context,
    span,
    PipelineTracer,
    trace_pipeline_stage,
    trace_request_id,
    trace_trace_id,
)


def test_set_and_get_request_context():
    """Request context can be set and retrieved."""
    set_request_context(request_id="req-123", trace_id="trace-456")
    ctx = get_request_context()
    assert ctx["request_id"] == "req-123"
    assert ctx["trace_id"] == "trace-456"


def test_set_request_context_initializes_observability():
    """Setting request context initializes observability."""
    # Should not raise
    set_request_context(request_id="req-789")
    # Context should be available
    ctx = get_request_context()
    assert ctx["request_id"] == "req-789"


def test_span_creates_context_manager():
    """Span context manager works without error."""
    with span(name="test-stage", kind="internal", attributes={"test": "value"}):
        pass  # should not raise


def test_span_without_tracer():
    """Span works even when no tracer is active (no-op)."""
    # This should not raise; span may be a no-op when observability not initialized
    with span("noop-stage"):
        pass


def test_pipeline_tracer_start_stop():
    """PipelineTracer start/stop cycle works."""
    tracer = PipelineTracer(request_id="req-tracer-001")
    tracer.start()
    # Can record stages
    tracer.record_stage(stage="test", latency_ms=42.0)
    tracer.record_error("test error")
    tracer.stop(success=True)
    # No exception should be raised


def test_pipeline_tracer_records_latency():
    """PipelineTracer records stage latencies."""
    tracer = PipelineTracer(request_id="req-latency-001")
    tracer.start()
    tracer.record_stage(stage="retrieval", latency_ms=120.5)
    tracer.record_stage(stage="generation", latency_ms=85.3)
    tracer.stop(success=True)
    assert tracer.stage_latencies["retrieval"] == 120.5
    assert tracer.stage_latencies["generation"] == 85.3


def test_pipeline_tracer_records_errors():
    """PipelineTracer records errors."""
    tracer = PipelineTracer(request_id="req-errors-001")
    tracer.start()
    tracer.record_error("something went wrong")
    tracer.stop(success=False, error="something went wrong")
    assert len(tracer.errors) == 1
    assert tracer.errors[0] == "something went wrong"


def test_trace_pipeline_stage_executes_fn():
    """trace_pipeline_stage executes the given function within a span."""
    result = trace_pipeline_stage(stage_name="test-fn", fn=lambda: 42)
    assert result == 42


def test_trace_pipeline_stage_records_error():
    """trace_pipeline_stage records and re-raises exceptions."""
    with pytest.raises(ValueError, match="error"):
        trace_pipeline_stage(stage_name="failing-fn", fn=lambda: (_ for _ in ()).throw(ValueError("error")))


def test_trace_request_id_returns_set_id():
    """trace_request_id returns the currently set request ID."""
    set_request_context(request_id="req-id-test")
    assert trace_request_id() == "req-id-test"


def test_trace_trace_id_returns_set_id():
    """trace_trace_id returns the currently set trace ID."""
    set_request_context(request_id="req-id-test", trace_id="trace-id-test")
    assert trace_trace_id() == "trace-id-test"


def test_integrations_noop_without_observability():
    """Wrappers work without observability being fully configured."""
    # enabled=False: wrappers still work, metrics/spans are no-ops
    # Just verify the functions don't crash in minimal state
    set_request_context(request_id="req-minimal")
    ctx = get_request_context()
    assert ctx["request_id"] == "req-minimal"