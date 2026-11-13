"""W3C traceparent injection/extraction."""
from __future__ import annotations

from observability.tracing import (
    current_trace_context,
    extract_trace_context,
    inject_trace_context,
)
from observability.tracing.context import root_context, set_current_trace_context


def test_round_trip():
    context = root_context(request_id="req-abc")
    carrier: dict[str, str] = {}
    inject_trace_context(carrier, context)
    assert carrier["traceparent"].startswith("00-")
    assert carrier["x-request-id"] == "req-abc"

    extracted = extract_trace_context(carrier)
    assert extracted is not None
    assert extracted.trace_id == context.trace_id
    assert extracted.span_id == context.span_id
    assert extracted.request_id == "req-abc"


def test_inject_uses_current_context(monkeypatch):
    context = root_context()
    token = set_current_trace_context(context)
    try:
        carrier: dict[str, str] = {}
        inject_trace_context(carrier)
        assert carrier["traceparent"].endswith(context.span_id + "-01")
    finally:
        from observability.tracing.context import reset_current_trace_context

        reset_current_trace_context(token)
    assert current_trace_context() is None


def test_extract_invalid_traceparent_returns_none():
    assert extract_trace_context({"traceparent": "garbage"}) is None
    assert extract_trace_context({"traceparent": "00-" + "g" * 32 + "-" + "0" * 16 + "-01"}) is None
    assert extract_trace_context({"traceparent": "ff-" + "0" * 32 + "-" + "0" * 16 + "-01"}) is None
    assert extract_trace_context({"traceparent": "00-" + "0" * 32 + "-" + "0" * 16 + "-01"}) is None
    assert extract_trace_context({}) is None
    assert extract_trace_context(None) is None


def test_extract_is_case_insensitive_on_header_name():
    context = root_context()
    carrier = {
        "Traceparent": f"00-{context.trace_id}-{context.span_id}-01",
    }
    extracted = extract_trace_context(carrier)
    assert extracted is not None
    assert extracted.trace_id == context.trace_id


def test_baggage_round_trip():
    from observability.contracts.models import TraceContext

    context = TraceContext(
        trace_id="a" * 32,
        span_id="b" * 16,
        request_id="req-1",
        baggage={"tenant": "alpha", "region": "eu"},
    )
    carrier: dict[str, str] = {}
    inject_trace_context(carrier, context)
    assert carrier["baggage"] == "tenant=alpha,region=eu"
    extracted = extract_trace_context(carrier)
    assert extracted is not None
    assert extracted.baggage == {"tenant": "alpha", "region": "eu"}
