"""Data models for the observability ecosystem.

Backend-independent value objects shared across all observability
components. These models do not depend on OpenTelemetry and work in
no-op mode.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SpanKind(str, Enum):
    """Kind of span, mirroring the OpenTelemetry span kinds."""

    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class StatusCode(str, Enum):
    """Terminal status of a finished span."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


def _validate_hex(value: str, length: int, field_name: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != length or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError(f"{field_name} must be {length} lowercase hex characters")
    return normalized


def is_valid_trace_id(value: str) -> bool:
    try:
        _validate_hex(value, 32, "trace_id")
    except ValueError:
        return False
    return value != "0" * 32


def is_valid_span_id(value: str) -> bool:
    try:
        _validate_hex(value, 16, "span_id")
    except ValueError:
        return False
    return value != "0" * 16


@dataclass(frozen=True)
class TraceContext:
    """Immutable carrier for trace identity across component boundaries.

    ``trace_id`` is 32 lowercase hex characters, ``span_id`` is 16. Both
    mirror the W3C Trace Context representation. ``request_id`` is the
    application-level correlation identifier carried alongside the trace.
    """

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    request_id: str | None = None
    baggage: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not is_valid_trace_id(self.trace_id):
            raise ValueError(f"invalid trace_id: {self.trace_id!r}")
        if not is_valid_span_id(self.span_id):
            raise ValueError(f"invalid span_id: {self.span_id!r}")
        if self.parent_span_id is not None:
            _validate_hex(self.parent_span_id, 16, "parent_span_id")

    def child(self, span_id: str, request_id: str | None = None) -> TraceContext:
        """Derive the context for a nested span in the same trace."""
        return TraceContext(
            trace_id=self.trace_id,
            span_id=span_id,
            parent_span_id=self.span_id,
            request_id=request_id if request_id is not None else self.request_id,
            baggage=dict(self.baggage),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "request_id": self.request_id,
            "baggage": dict(self.baggage),
        }


@dataclass
class SpanEvent:
    """Structured event attached to a span (e.g. an exception)."""

    name: str
    timestamp: float
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanRecord:
    """Flat, backend-independent snapshot of a finished span.

    Used by the no-op/in-memory fallback and by exporters that render
    spans themselves. When OpenTelemetry is configured, the SDK's own
    ReadableSpan objects are consumed instead.
    """

    name: str
    span_id: str
    trace_id: str
    parent_span_id: str | None
    kind: SpanKind
    start_time: float
    end_time: float
    attributes: dict[str, Any] = field(default_factory=dict)
    status: StatusCode = StatusCode.UNSET
    status_description: str | None = None
    events: list[SpanEvent] = field(default_factory=list)


@dataclass(frozen=True)
class RedactionRule:
    """One redaction rule: a regex pattern plus its replacement."""

    pattern: str
    replacement: str = "***REDACTED***"
    description: str = ""


@dataclass(frozen=True)
class ExportConfig:
    """Connection and batching settings for an OTLP exporter."""

    endpoint: str = ""
    protocol: str = "grpc"  # "grpc" | "http"
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 10.0
    insecure: bool = False
    batch_size: int = 512
    schedule_delay_ms: int = 5000
    metric_export_interval_ms: int = 10000


class ErrorCategory:
    """Bounded error-category labels safe for metric dimensions."""

    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    DEPENDENCY = "dependency"
    NETWORK = "network"
    STORAGE = "storage"
    PARSE = "parse"
    MODEL = "model"
    GUARDRAIL = "guardrail_block"
    NOT_FOUND = "not_found"
    UNKNOWN = "unknown"

    ALL = (
        TIMEOUT,
        RATE_LIMIT,
        AUTHENTICATION,
        AUTHORIZATION,
        VALIDATION,
        DEPENDENCY,
        NETWORK,
        STORAGE,
        PARSE,
        MODEL,
        GUARDRAIL,
        NOT_FOUND,
        UNKNOWN,
    )


# --- Metric label policy ------------------------------------------------
# Label keys that are always permitted (bounded cardinality).
ALLOWED_LABEL_KEYS = frozenset(
    {
        "component",
        "operation",
        "provider",
        "model",
        "outcome",
        "error_category",
        "retrieval_strategy",
        "stage",
        "status",
    }
)
# Label keys that must never appear on metrics (unbounded/sensitive).
FORBIDDEN_LABEL_KEYS = frozenset(
    {
        "user_id",
        "username",
        "request_id",
        "trace_id",
        "span_id",
        "chunk_id",
        "document_id",
        "doc_id",
        "chunk_ids",
        "file_name",
        "filename",
        "file_path",
        "path",
        "query",
        "prompt",
        "input",
        "output",
        "context",
        "text",
        "content",
        "metadata",
        "api_key",
        "authorization",
        "token",
        "password",
        "secret",
        "email",
        "session_id",
        "tenant_id",
        "conversation_id",
    }
)
MAX_LABEL_VALUE_LENGTH = 64
