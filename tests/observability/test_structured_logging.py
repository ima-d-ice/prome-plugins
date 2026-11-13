"""Structured logging: JSON/text output, correlation, redaction."""
from __future__ import annotations

import json
import logging

from observability.configuration import (
    ObservabilityConfig,
    shutdown_observability,
)
from observability.logging.correlation import CorrelationFilter, correlation_fields
from observability.logging.structured import (
    HumanFormatter,
    JsonFormatter,
    configure_logging,
    get_logger,
)
from observability.tracing.context import root_context, set_current_trace_context


def _make_record(msg: str, **extras) -> logging.LogRecord:
    record = logging.LogRecord(
        name="prome_rag.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )
    for key, value in extras.items():
        setattr(record, key, value)
    return record


def test_json_formatter_emits_valid_json():
    record = _make_record("hello world", component="parser", operation="parse", outcome="ok")
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["component"] == "parser"
    assert payload["outcome"] == "ok"
    assert "timestamp" in payload


def test_json_formatter_redacts_secrets():
    record = _make_record("calling with sk-abc1234567890def key")
    payload = json.loads(JsonFormatter().format(record))
    assert "abc1234567890def" not in json.dumps(payload)


def test_json_formatter_drops_prompt_payloads_by_default():
    record = _make_record("done", prompt="the complete user prompt text")
    payload = json.loads(JsonFormatter().format(record))
    assert "prompt" not in payload


def test_json_formatter_allows_prompt_when_opted_in(monkeypatch):
    from observability.configuration import get_config

    config = get_config()
    assert config.record_prompts is False
    # simulate opt-in via environment-derived config
    from observability.configuration import ObservabilityConfig as Cfg

    monkeypatch.setattr("observability.logging.structured._safe_config", lambda: Cfg(record_prompts=True))
    record = _make_record("done", prompt="sensitive but explicitly enabled")
    payload = json.loads(JsonFormatter().format(record))
    assert payload["prompt"] == "sensitive but explicitly enabled"


def test_human_formatter_readable():
    record = _make_record("done", component="retriever", outcome="ok", trace_id="t" * 32)
    line = HumanFormatter().format(record)
    assert "INFO" in line
    assert "component=retriever" in line
    assert "trace-id=" in line
    assert "done" in line


def test_correlation_filter_injects_ids():
    context = root_context()
    token = set_current_trace_context(context)
    try:
        record = _make_record("msg")
        CorrelationFilter().filter(record)
        assert record.trace_id == context.trace_id
        assert record.request_id == context.request_id
        assert record.span_id == context.span_id
    finally:
        from observability.tracing.context import reset_current_trace_context

        reset_current_trace_context(token)


def test_correlation_fields_empty_without_context():
    assert correlation_fields() == {}


def test_configure_logging_attach_and_idempotent(tmp_path):
    log_file = tmp_path / "obs.log"
    config = ObservabilityConfig(enabled=False, log_level="INFO", log_format="json", log_file=str(log_file))
    configure_logging(config)
    namespace = logging.getLogger("prome_rag")
    owned = [h for h in namespace.handlers if getattr(h, "_prome_rag_owned", False)]
    assert len(owned) == 2  # stderr + file
    configure_logging(config)  # second call replaces, does not duplicate
    owned = [h for h in namespace.handlers if getattr(h, "_prome_rag_owned", False)]
    assert len(owned) == 2
    get_logger("parser").info("hello from test", extra={"component": "parser"})
    shutdown_observability()


def test_configure_logging_does_not_touch_root_logger():
    root_handlers_before = list(logging.getLogger().handlers)
    configure_logging(ObservabilityConfig(enabled=False))
    assert logging.getLogger().handlers == root_handlers_before
    shutdown_observability()


def test_exception_is_rendered_redacted():
    record = _make_record("failed")
    try:
        raise RuntimeError("boom token=abcdef123456")
    except RuntimeError:
        import sys

        record.exc_info = sys.exc_info()
    payload = json.loads(JsonFormatter().format(record))
    assert "exception" in payload
    assert "abcdef123456" not in payload["exception"]
