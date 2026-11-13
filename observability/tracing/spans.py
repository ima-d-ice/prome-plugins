"""Span scopes, canonical span names, and span attribute policy."""
from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

from ..configuration import get_config
from ..contracts.models import (
    FORBIDDEN_LABEL_KEYS,
    SpanKind,
    StatusCode,
    TraceContext,
)

logger = logging.getLogger("prome_rag.tracing")


class SpanNames:
    """Canonical span names: <stage>.<operation>."""

    # Ingestion
    INGESTION_DISCOVERY = "ingestion.discovery"
    INGESTION_UPLOAD = "ingestion.upload"
    INGESTION_VALIDATE = "ingestion.validate"
    INGESTION_LOAD = "ingestion.load"
    INGESTION_PARSE = "ingestion.parse"
    INGESTION_CHUNK = "ingestion.chunk"
    INGESTION_EMBED = "ingestion.embed"
    INGESTION_INDEX_DENSE = "ingestion.index_dense"
    INGESTION_INDEX_LEXICAL = "ingestion.index_lexical"

    # Retrieval
    RETRIEVAL_VALIDATE = "retrieval.validate"
    RETRIEVAL_ENHANCE = "retrieval.enhance"
    RETRIEVAL_DENSE = "retrieval.dense"
    RETRIEVAL_BM25 = "retrieval.bm25"
    RETRIEVAL_RRF_FUSION = "retrieval.rrf_fusion"
    RETRIEVAL_PARENT_SEARCH = "retrieval.parent_search"
    RETRIEVAL_CHILD_SEARCH = "retrieval.child_search"
    RETRIEVAL_CONTEXT_EXPAND = "retrieval.context_expand"

    # Reranking
    RERANK_PREPARE = "rerank.prepare"
    RERANK_INFERENCE = "rerank.inference"
    RERANK_REORDER = "rerank.reorder"

    # Generation
    GENERATION_PROVIDER_SELECT = "generation.provider_select"
    GENERATION_MODEL_SELECT = "generation.model_select"
    GENERATION_PROMPT = "generation.prompt"
    GENERATION_RATE_LIMIT_WAIT = "generation.rate_limit_wait"
    GENERATION_REQUEST = "generation.request"
    GENERATION_FALLBACK = "generation.fallback"
    GENERATION_OUTPUT = "generation.output"

    # Guardrails
    GUARDRAILS_PROMPT_GUARD = "guardrails.prompt_guard"
    GUARDRAILS_SOURCE_TRUST = "guardrails.source_trust"
    GUARDRAILS_ACCESS_CONTROL = "guardrails.access_control"
    GUARDRAILS_PRIVACY_DETECT = "guardrails.privacy_detect"
    GUARDRAILS_SAFEGUARD = "guardrails.safeguard"
    GUARDRAILS_CITATION_VALIDATE = "guardrails.citation_validate"
    GUARDRAILS_ENFORCEMENT = "guardrails.enforcement"

    # Evaluation
    EVALUATION_DATASET_LOAD = "evaluation.dataset_load"
    EVALUATION_RAGAS = "evaluation.ragas"
    EVALUATION_JUDGE_CALL = "evaluation.judge_call"
    EVALUATION_DETERMINISTIC = "evaluation.deterministic"
    EVALUATION_REPORT = "evaluation.report"

    # Orchestration
    ORCHESTRATION_PIPELINE = "orchestration.pipeline"


class Attr:
    """Canonical attribute keys (bounded, non-sensitive)."""

    COMPONENT = "component"
    OPERATION = "operation"
    REQUEST_ID = "request_id"
    PROVIDER = "provider"
    MODEL = "model"
    OUTCOME = "outcome"
    ERROR_CATEGORY = "error.category"
    ERROR_TYPE = "error.type"
    RETRIEVAL_STRATEGY = "retrieval.strategy"
    CANDIDATE_COUNT = "candidate_count"
    BATCH_SIZE = "batch_size"
    DIMENSIONS = "dimensions"
    INPUT_TOKENS = "usage.input_tokens"
    OUTPUT_TOKENS = "usage.output_tokens"
    TOTAL_TOKENS = "usage.total_tokens"
    DECISION = "guardrails.decision"
    SAMPLE_INDEX = "evaluation.sample_index"
    CHUNKS_CREATED = "chunks_created"


_FORBIDDEN_ATTRIBUTE_KEYS = FORBIDDEN_LABEL_KEYS | frozenset(
    {
        "document_content",
        "retrieved_context",
        "raw_prompt",
        "system_prompt",
        "completion",
    }
)

# Attribute keys whose values are free text and are recorded only when the
# operator explicitly opts in via configuration.
_OPT_IN_CONTENT_KEYS = ("prompt", "document_content")


def sanitize_attributes(attributes: Mapping[str, Any] | None) -> dict[str, Any]:
    """Drop forbidden keys and truncate over-long string values."""
    if not attributes:
        return {}
    config = get_config()
    max_len = config.max_attribute_length if config else 512
    cleaned: dict[str, Any] = {}
    for key, value in attributes.items():
        lowered = str(key).lower()
        if lowered in _FORBIDDEN_ATTRIBUTE_KEYS:
            if lowered in _OPT_IN_CONTENT_KEYS:
                allowed = (lowered == "prompt" and config.record_prompts) or (
                    lowered == "document_content" and config.record_document_content
                )
                if allowed:
                    cleaned[str(key)] = _truncate(value, max_len * 4)
                else:
                    logger.debug("dropping opted-in content attribute %r", key)
                continue
            logger.debug("dropping forbidden span attribute %r", key)
            continue
        cleaned[str(key)] = _truncate(value, max_len)
    return cleaned


def _truncate(value: Any, max_len: int) -> Any:
    if isinstance(value, str) and len(value) > max_len:
        return value[: max_len - 10] + "...(trunc)"
    return value


def categorize_exception(exc: BaseException) -> str:
    """Map an exception to a bounded error category."""
    from ..contracts.exceptions import ObservabilityError
    from ..contracts.models import ErrorCategory

    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if isinstance(exc, TimeoutError) or "timeout" in name or "timed out" in text:
        return ErrorCategory.TIMEOUT
    if "ratelimit" in name or "rate_limit" in name or "429" in text:
        return ErrorCategory.RATE_LIMIT
    if "auth" in name or "credential" in name or "401" in text:
        return ErrorCategory.AUTHENTICATION
    if isinstance(exc, PermissionError) or "authori" in text or "403" in text or "forbidden" in text:
        return ErrorCategory.AUTHORIZATION
    if isinstance(exc, (ValueError, KeyError, TypeError)) or "valid" in name:
        return ErrorCategory.VALIDATION
    if isinstance(exc, FileNotFoundError) or "notfound" in name or "not found" in text:
        return ErrorCategory.NOT_FOUND
    if "connection" in name or "network" in name or "dns" in name or "ssl" in name:
        return ErrorCategory.NETWORK
    if "parse" in name or "parser" in name or "decode" in name or "unmarshal" in name:
        return ErrorCategory.PARSE
    if any(token in name for token in ("storage", "qdrant", "vector", "disk", "ioerror")):
        return ErrorCategory.STORAGE
    if any(token in name for token in ("model", "inference", "embedding", "rerank")):
        return ErrorCategory.MODEL
    if isinstance(exc, ObservabilityError):
        return ErrorCategory.UNKNOWN
    return ErrorCategory.UNKNOWN


class SpanScope:
    """Context manager wrapping a single span (OpenTelemetry or no-op).

    Attributes and exception recording are safe in both modes. On error,
    the span is marked with a bounded ``error.category`` and a redacted
    exception event; the exception always propagates.
    """

    def __init__(
        self,
        tracer: Any,
        name: str,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Mapping[str, Any] | None = None,
        request_id: str | None = None,
    ) -> None:
        self._tracer = tracer
        self.name = name
        self.kind = kind
        self._initial_attributes = dict(attributes or {})
        self._request_id = request_id
        self._otel_span: Any = None
        self._otel_token: Any = None
        self._context_token: Any = None
        self._start: float | None = None
        self.trace_context: TraceContext | None = None

    # -- entry/exit ------------------------------------------------------

    def __enter__(self) -> SpanScope:
        self._start = time.time()
        sanitized = sanitize_attributes(self._initial_attributes)
        sanitized.setdefault(Attr.OPERATION, self.name)
        self._otel_span, self.trace_context, self._otel_token = self._tracer._start_span(
            self.name,
            kind=self.kind,
            attributes=sanitized,
            request_id=self._request_id,
        )
        from .context import set_current_trace_context

        self._context_token = set_current_trace_context(self.trace_context)
        if self.trace_context is not None and self.trace_context.request_id:
            self._set_raw_attribute(Attr.REQUEST_ID, self.trace_context.request_id)
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        outcome = "ok"
        if exc is not None:
            outcome = "error"
            self.record_exception(exc)
        else:
            self.set_status(StatusCode.OK)
        self.set_attribute(Attr.OUTCOME, outcome)
        if self._start is not None:
            self.set_attribute("duration_ms", round((time.time() - self._start) * 1000.0, 3))
        self._end_span()
        from .context import reset_current_trace_context

        if self._context_token is not None:
            reset_current_trace_context(self._context_token)
        return False

    # -- attributes --------------------------------------------------------

    def set_attribute(self, key: str, value: Any) -> None:
        for k, v in sanitize_attributes({key: value}).items():
            self._set_raw_attribute(k, v)

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        for k, v in sanitize_attributes(attributes).items():
            self._set_raw_attribute(k, v)

    def _set_raw_attribute(self, key: str, value: Any) -> None:
        if self._otel_span is None:
            return
        try:
            self._otel_span.set_attribute(key, value)
        except Exception as exc:  # telemetry must never break user code
            logger.debug("failed to set span attribute %r: %s", key, exc)

    # -- events -------------------------------------------------------------

    def add_event(self, name: str, attributes: Mapping[str, Any] | None = None) -> None:
        attributes = sanitize_attributes(attributes)
        if self._otel_span is None:
            return
        try:
            self._otel_span.add_event(name, attributes=attributes or None)
        except Exception as exc:
            logger.debug("failed to add span event %r: %s", name, exc)

    def record_exception(self, exc: BaseException, category: str | None = None) -> None:
        from ..logging.redaction import redact_text

        category = category or categorize_exception(exc)
        message = redact_text(str(exc))
        if self._otel_span is None:
            return
        try:
            from opentelemetry.trace import Status
            from opentelemetry.trace import StatusCode as OTelStatusCode

            self._otel_span.set_attribute(Attr.ERROR_CATEGORY, category)
            self._otel_span.set_attribute(Attr.ERROR_TYPE, type(exc).__name__)
            self._otel_span.record_exception(
                exc,
                attributes={Attr.ERROR_CATEGORY: category, "exception.message": message},
            )
            self._otel_span.set_status(Status(OTelStatusCode.ERROR, message[:256]))
        except Exception as inner:
            logger.debug("failed to record exception on span: %s", inner)

    def set_status(self, status: StatusCode, description: str | None = None) -> None:
        if self._otel_span is None:
            return
        try:
            from opentelemetry.trace import Status
            from opentelemetry.trace import StatusCode as OTelStatusCode

            mapping = {
                StatusCode.OK: OTelStatusCode.OK,
                StatusCode.ERROR: OTelStatusCode.ERROR,
                StatusCode.UNSET: OTelStatusCode.UNSET,
            }
            self._otel_span.set_status(Status(mapping[status], description))
        except Exception as exc:
            logger.debug("failed to set span status: %s", exc)

    # -- internals -----------------------------------------------------------

    def _end_span(self) -> None:
        if self._otel_span is not None:
            try:
                self._otel_span.end()
            except Exception as exc:
                logger.debug("failed to end span: %s", exc)
            self._otel_span = None
        if self._otel_token is not None:
            try:
                from opentelemetry import context as otel_context

                otel_context.detach(self._otel_token)
            except Exception as exc:
                logger.debug("failed to detach otel context: %s", exc)
            self._otel_token = None
