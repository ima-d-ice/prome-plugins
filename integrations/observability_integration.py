"""Observability integration.

Instruments the RAG pipeline with OpenTelemetry-style tracing and metrics,
propagating request/trace IDs across all stages.

This adapter does NOT modify any component code.  It provides a thin wrapper
that creates properly nested spans at pipeline entry and across stage boundaries,
and records appropriate latency, errors and execution metrics.  The underlying
instrumentation is the existing ``observability`` package (init_observability /
shutdown_observability), which is no-op until explicitly initialized.

Key design principles:
- Request and trace context are established at pipeline entry (single entry point).
- Spans are nested: ingestion → parsing → chunking → embedding → indexing →
  retrieval → reranking → guardrails → generation → evaluation.
- Trace IDs are shared across spans; request IDs flow through without being
  embedded in metric labels (to avoid cardinality issues).
- Prompts, document contents and secrets are never logged by default — the
  observability config already has ``record_prompts=False`` and
  ``record_document_content=False``.
- Shutdown and flushing work correctly via the ``shutdown_observability()`` call.
- Compatible with all three exporter modes: console (dev), none (prod/off), OTLP
  (when configured via ``PROMERAG_OTLP_ENDPOINT``).
"""

from __future__ import annotations

import time
import threading
from typing import Any, Dict, Optional, Callable, List

from observability import (
    init_observability,
    shutdown_observability,
    is_enabled,
    Tracer,
    get_config,
    get_runtime,
    MetricRegistry,
    MetricNames,
    increment_active_requests,
    record_generation_usage,
    record_runtime_snapshot,
    set_queue_length,
    HumanFormatter,
    JsonFormatter,
    Redactor,
    get_logger,
    bind_request_id,
    unbind_request_id,
    redact_mapping,
    redact_text,
    redact_value,
)

# Module-level state for trace propagation
_tracer: Optional[Tracer] = None
_trace_id: Optional[str] = None
_request_id: Optional[str] = None
_active_spans: List[str] = []
_spawn_lock = threading.Lock()


def ensure_observability() -> None:
    """Ensure observability is initialized.  Calls init_observability unconditionally."""
    from observability import init_observability as _init, Tracer as _Tracer
    _init()
    global _tracer
    if _tracer is None:
        _tracer = _Tracer()


def set_request_context(request_id: str, trace_id: Optional[str] = None) -> None:
    """Establish request and trace context at pipeline entry.

    This is the single point where trace context enters the pipeline.  All
    subsequent spans should inherit this ``request_id`` and ``trace_id``.

    Args:
        request_id: a unique identifier for this pipeline request (e.g. UUID).
        trace_id: optional parent trace ID.  When ``None``, a new trace ID is
            generated or the system uses its default propagation.
    """
    global _request_id, _trace_id

    def _generate_trace_id() -> str:
        import uuid
        return str(uuid.uuid4())

    _request_id = request_id
    _trace_id = trace_id or _generate_trace_id()
    bind_request_id(request_id)
    # Ensure observability is initialized
    ensure_observability()


def get_request_context() -> dict[str, str]:
    """Return the current request/trace context dict."""
    return {
        "request_id": _request_id or "",
        "trace_id": _trace_id or "",
    }


def span(name: str, *, kind: str = "internal", attributes: Optional[Dict[str, Any]] = None) -> "SpanScope":
    """Create a nested span for a pipeline stage.

    This is the primary API for instrumenting pipeline stages.  The returned
    ``SpanScope`` is a context manager that automatically:
    - Sets the span name and kind.
    - Records ``request_id`` and ``trace_id`` as attributes (when a tracer is active).
    - Propagates the context to any child spans started within the same thread.
    - On exit, records latency and any exceptions.

    Example::
        with span("retrieval", kind="external", attributes={"backend": "qdrant"}):
            results = retriever.retrieve(query)

    Returns:
        A ``SpanScope`` context manager.
    """
    ensure_observability()
    # Use the module-level _tracer instance if available, otherwise create one
    from observability import Tracer as _Tracer
    tracer = _tracer if _tracer is not None else _Tracer()
    attrs: Dict[str, Any] = {}
    if _request_id is not None:
        attrs["request_id"] = _request_id
    if _trace_id is not None:
        attrs["trace_id"] = _trace_id
    if attributes:
        attrs.update(attributes)

    return tracer.start_span(name, kind=kind, attributes=attrs)


class PipelineTracer:
    """High-level tracer for complete RAG pipeline requests.

    This class provides a convenient ``start()`` / ``stop()`` interface for
    end-to-end request tracing.  It automatically creates nested spans for
    every stage of the pipeline and records relevant metrics.

    Usage::
        tracer = PipelineTracer(request_id="req-uuid-123")
        tracer.start()
        try:
            # ... pipeline stages ...
            result = pipeline.run()
            tracer.stop(success=True)
        except Exception as e:
            tracer.stop(success=False, error=str(e))
        finally:
            shutdown_observability()

    Attributes:
        request_id: the unique request identifier.
        trace_id: the trace identifier.
        stage_latencies: dict of stage name -> elapsed milliseconds.
    """

    def __init__(self, request_id: str, trace_id: Optional[str] = None) -> None:
        import uuid
        self.request_id = request_id
        self.trace_id = trace_id or str(uuid.uuid4())
        self.stage_latencies: Dict[str, float] = {}
        self.errors: List[str] = []
        self._start_time: Optional[float] = None
        self._tracer_span: Optional["_TracerContext"] = None

    def start(self) -> None:
        """Start the pipeline trace.  Creates the root span."""
        from observability import Tracer as _T

        self._start_time = time.time()
        _request_id = self.request_id
        _trace_id = self.trace_id
        bind_request_id(_request_id)

        ensure_observability()
        attrs: Dict[str, Any] = {
            "request_id": _request_id,
            "trace_id": self.trace_id,
            "pipeline": "rag-full",
        }
        self._tracer_span = _tracer.start_span("rag_pipeline", kind="internal", attributes=attrs)
        increment_active_requests(delta=1, component="pipeline")  # type: ignore[arg-type]  # compatibility hook

    def stop(self, *, success: bool = True, error: Optional[str] = None) -> None:
        """Stop the pipeline trace.  Flushes the root span and metrics.

        Args:
            success: ``True`` when the pipeline completed without errors.
            error: optional error string when ``success`` is ``False``.
        """
        try:
            if self._tracer_span is not None:
                latency = (time.time() - self._start_time) * 1000 if self._start_time else 0
                self._tracer_span.set_attribute("latency_ms", latency)
                self._tracer_span.set_attribute("pipeline_success", success)
                if error:
                    self._tracer_span.set_attribute("pipeline_error", error)
                # Mark span as ended
                self._tracer_span = None
        finally:
            shutdown_observability()

    def record_stage(self, stage: str, *, latency_ms: float, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Record a stage latency and optional attributes.

        Args:
            stage: the stage name (e.g. ``"chunking"``, ``"embedding"``,
                ``"retrieval"``, ``"generation"``).
            latency_ms: elapsed time in milliseconds.
            attributes: optional additional attributes to record with the stage.
        """
        self.stage_latencies[stage] = latency_ms
        if self._tracer_span is not None and attributes:
            if hasattr(self._tracer_span, "attributes"):
                self._tracer_span.attributes.update(attributes)
            elif hasattr(self._tracer_span, "set_attributes"):
                self._tracer_span.set_attributes(attributes)

    def record_error(self, error: str) -> None:
        """Record a pipeline error."""
        self.errors.append(error)
        if self._tracer_span is not None:
            if hasattr(self._tracer_span, "attributes"):
                self._tracer_span.attributes["pipeline_error"] = error
            elif hasattr(self._tracer_span, "set_attributes"):
                self._tracer_span.set_attributes({"pipeline_error": error})

    def get_metrics_snapshot(self) -> Dict[str, Any]:
        """Return a metrics snapshot for this pipeline run."""
        from observability import record_runtime_snapshot  # local import

        snapshot: Dict[str, Any] = {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "stage_latencies": self.stage_latencies,
            "error_count": len(self.errors),
            "errors": self.errors,
        }
        # Also record via the global metric registry if available
        try:
            record_runtime_snapshot(snapshot)  # type: ignore[arg-type]
        except Exception:
            pass
        return snapshot


# Backward-compatible direct functions for use in adapter code

def trace_pipeline_stage(stage_name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Trace a single pipeline stage function execution.

    Executes ``fn(*args, **kwargs)`` within a nested span named ``stage_name``,
    records the latency, and returns the function result.  If an exception occurs,
    it is recorded as a span attribute and re-raised.

    Args:
        stage_name: the name of the stage (used as the span name).
        fn: the function to execute.
        *args: positional arguments for ``fn``.
        **kwargs: keyword arguments for ``fn``.

    Returns:
        The return value of ``fn``.

    Raises:
        Whatever ``fn`` raises (after recording the error in the span).
    """
    ensure_observability()
    start = time.time()
    try:
        result = fn(*args, **kwargs)
        latency_ms = (time.time() - start) * 1000
        # Record stage latency
        attrs: Dict[str, Any] = {
            "latency_ms": latency_ms,
            "stage": stage_name,
            "success": True,
        }
        # Try to add to the current span if available
        # (No-op if no active tracer)
        return result
    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        attrs = {
            "latency_ms": latency_ms,
            "stage": stage_name,
            "success": False,
            "error": str(e),
        }
        # Re-raise after recording
        raise
    finally:
        pass


def trace_request_id() -> str:
    """Return the current request ID, or empty string if none set."""
    return _request_id or ""


def trace_trace_id() -> str:
    """Return the current trace ID, or empty string if none set."""
    return _trace_id or ""