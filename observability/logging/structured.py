"""Structured logging built on :mod:`logging`.

Two formats are supported: ``json`` (single-line JSON objects, suitable
for log aggregation) and ``text`` (human-readable key=value pairs).
Both redact secrets and drop prompt/document payloads by default.

``configure_logging`` attaches one or two handlers to the ``prome_rag``
logger namespace. It never touches the root logger or handlers it did not
create, and repeated calls replace previously attached handlers.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from ..configuration import ObservabilityConfig
from .redaction import get_default_redactor

LOGGER_NAMESPACE = "prome_rag"

# Fields we know how to render; anything else (minus LogRecord internals)
# is carried over as a redacted "extra".
_KNOWN_EXTRA_FIELDS = (
    "component",
    "operation",
    "request_id",
    "trace_id",
    "span_id",
    "provider",
    "model",
    "duration_ms",
    "outcome",
    "error_category",
    "retrieval_strategy",
    "stage",
)

_LOGRECORD_BUILTINS = frozenset(
    logging.makeLogRecord({}).__dict__.keys()
) | {"message", "asctime", "prome_rag_owned"}

# Free-text payloads that are never rendered unless explicitly allowed.
_DROPPED_EXTRA_KEYS = frozenset({"prompt", "document_content", "raw_prompt", "system_prompt", "retrieved_context", "completion"})


def _iso_now(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _redact_record_extras(record: logging.LogRecord) -> dict[str, Any]:
    redactor = get_default_redactor()
    extras: dict[str, Any] = {}
    for key, value in record.__dict__.items():
        if key in _LOGRECORD_BUILTINS:
            continue
        if key.startswith("_"):
            continue
        lowered = key.lower()
        if lowered in _DROPPED_EXTRA_KEYS:
            config = _safe_config()
            allowed = (lowered == "prompt" and config and config.record_prompts) or (
                lowered == "document_content" and config and config.record_document_content
            )
            if not allowed:
                continue
        extras[key] = redactor.redact_value(value)
    return extras


def _safe_config() -> ObservabilityConfig | None:
    try:
        from ..configuration import get_config

        return get_config()
    except Exception:
        return None


def _base_payload(record: logging.LogRecord) -> dict[str, Any]:
    redactor = get_default_redactor()
    payload: dict[str, Any] = {
        "timestamp": _iso_now(record.created),
        "level": record.levelname,
        "logger": record.name,
        "message": redactor.redact_text(record.getMessage()),
    }
    for field_name in _KNOWN_EXTRA_FIELDS:
        value = getattr(record, field_name, None)
        if value is not None and value != "":
            payload[field_name] = value if isinstance(value, (int, float, bool)) else redactor.redact_text(str(value))
    payload.update(_redact_record_extras(record))
    if record.exc_info:
        payload["exception"] = redactor.redact_text(
            "".join(_format_exception(record.exc_info))
        ).rstrip()
    return payload


def _format_exception(exc_info) -> list[str]:
    formatter = logging.Formatter()
    text = formatter.formatException(exc_info)
    return text.splitlines(keepends=True)


class JsonFormatter(logging.Formatter):
    """Single-line JSON per record."""

    def format(self, record: logging.LogRecord) -> str:
        payload = _base_payload(record)
        try:
            return json.dumps(payload, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as exc:  # never lose a log line
            fallback = {
                "timestamp": payload.get("timestamp"),
                "level": payload.get("level"),
                "logger": LOGGER_NAMESPACE,
                "message": f"unserializable log record: {exc}",
            }
            return json.dumps(fallback, ensure_ascii=False)


class HumanFormatter(logging.Formatter):
    """Readable ``key=value`` pairs with a UTC timestamp."""

    def format(self, record: logging.LogRecord) -> str:
        payload = _base_payload(record)
        extras = {
            key: value
            for key, value in payload.items()
            if key not in ("timestamp", "level", "logger", "message", "exception")
        }
        extras_str = " ".join(f"{_fmt_key(key)}={_fmt_value(value)}" for key, value in sorted(extras.items()))
        line = (
            f"{payload['timestamp']} {payload['level']:<7} {payload['logger']}"
            + (f" {extras_str}" if extras_str else "")
            + f" {payload['message']}"
        )
        if "exception" in payload:
            line += "\n" + payload["exception"]
        return line


def _fmt_key(key: str) -> str:
    return key.replace("_", "-")


def _fmt_value(value: Any) -> str:
    if isinstance(value, (int, float, bool)):
        return str(value)
    text = str(value)
    return json.dumps(text, ensure_ascii=False) if any(ch.isspace() for ch in text) else text


def _formatter_for(log_format: str) -> logging.Formatter:
    return JsonFormatter() if log_format == "json" else HumanFormatter()


def configure_logging(config: ObservabilityConfig) -> list[logging.Handler]:
    """Attach namespace handlers per ``config``; returns owned handlers."""
    logger = logging.getLogger(LOGGER_NAMESPACE)
    # Remove handlers we installed previously (idempotent re-init).
    for handler in list(logger.handlers):
        if getattr(handler, "_prome_rag_owned", False):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
    logger.setLevel(getattr(logging, config.log_level, logging.INFO))
    logger.propagate = False

    from .correlation import CorrelationFilter

    handlers: list[logging.Handler] = []

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(_formatter_for(config.log_format))
    stream.addFilter(CorrelationFilter())
    stream._prome_rag_owned = True  # type: ignore[attr-defined]
    logger.addHandler(stream)
    handlers.append(stream)

    if config.log_file:
        file_handler = logging.FileHandler(config.log_file, encoding="utf-8")
        file_handler.setFormatter(_formatter_for(config.log_format))
        file_handler.addFilter(CorrelationFilter())
        file_handler._prome_rag_owned = True  # type: ignore[attr-defined]
        logger.addHandler(file_handler)
        handlers.append(file_handler)

    return handlers


def get_logger(name: str) -> logging.Logger:
    if name == LOGGER_NAMESPACE or name.startswith(LOGGER_NAMESPACE + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAMESPACE}.{name}")
