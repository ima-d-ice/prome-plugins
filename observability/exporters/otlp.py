"""OTLP exporter: batch spans/metrics to an explicitly configured collector.

Nothing is sent anywhere until :func:`observability.init_observability` is
called with ``PROMERAG_EXPORTER=otlp`` (or this module's builders are used
directly). No dashboard, server, or agent is started.

Configuration via environment:

- ``PROMERAG_OTLP_ENDPOINT``   collector URL, e.g. ``http://localhost:4317`` (gRPC)
                                or ``http://localhost:4318`` (HTTP, paths
                                ``/v1/traces`` / ``/v1/metrics`` are appended when missing)
- ``PROMERAG_OTLP_PROTOCOL``   ``grpc`` (default) or ``http``
- ``PROMERAG_OTLP_HEADERS``    comma-separated ``key=value`` pairs; used for auth.
                                Values are never logged or attached to spans/metrics.
- ``PROMERAG_OTLP_TIMEOUT``    per-export timeout in seconds (default 10)
- ``PROMERAG_OTLP_INSECURE``   gRPC only: allow plaintext (default false)
- ``PROMERAG_OTLP_BATCH_SIZE`` max spans per batch (default 512)
- ``PROMERAG_OTLP_SCHEDULE_DELAY_MS``  delay between span flushes (default 5000)
- ``PROMERAG_METRICS_EXPORT_INTERVAL_MS``  metrics pull interval (default 10000)
"""
from __future__ import annotations

import logging
from typing import Any

from ..contracts.exceptions import ConfigurationError, ExportError

logger = logging.getLogger("prome_rag.exporters.otlp")

_TRACES_PATH = "/v1/traces"
_METRICS_PATH = "/v1/metrics"


def otlp_export_summary(config: Any) -> dict[str, Any]:
    """Return a redacted summary of OTLP settings (safe to log)."""
    otlp = config.otlp
    return {
        "endpoint": otlp.endpoint,
        "protocol": otlp.protocol,
        "timeout": otlp.timeout,
        "insecure": otlp.insecure,
        "header_keys": sorted(otlp.headers),
        "batch_size": otlp.batch_size,
        "schedule_delay_ms": otlp.schedule_delay_ms,
        "metric_export_interval_ms": otlp.metric_export_interval_ms,
    }


def _validate(config: Any) -> None:
    otlp = getattr(config, "otlp", None)
    if otlp is None:
        raise ConfigurationError("OTLP exporter requires exporter configuration")
    if not otlp.endpoint:
        raise ConfigurationError(
            "PROMERAG_OTLP_ENDPOINT must be set when PROMERAG_EXPORTER=otlp"
        )
    if otlp.protocol not in ("grpc", "http"):
        raise ConfigurationError(
            f"PROMERAG_OTLP_PROTOCOL must be 'grpc' or 'http', got {otlp.protocol!r}"
        )
    if otlp.timeout <= 0:
        raise ConfigurationError("PROMERAG_OTLP_TIMEOUT must be positive")


def _http_endpoint(base: str, path: str) -> str:
    base = base.rstrip("/")
    if base.endswith(path):
        return base
    return base + path


def _build_span_exporter(config: Any) -> Any:
    otlp = config.otlp
    if otlp.protocol == "grpc":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
        except Exception as exc:
            raise ExportError(
                "opentelemetry-exporter-otlp-proto-grpc is not installed; "
                "install the 'observability' extra or set PROMERAG_OTLP_PROTOCOL=http"
            ) from exc
        return OTLPSpanExporter(
            endpoint=otlp.endpoint,
            insecure=otlp.insecure,
            headers=dict(otlp.headers),
            timeout=otlp.timeout,
        )
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
    except Exception as exc:
        raise ExportError(
            "opentelemetry-exporter-otlp-proto-http is not installed; "
            "install the 'observability' extra or set PROMERAG_OTLP_PROTOCOL=grpc"
        ) from exc
    return OTLPSpanExporter(
        endpoint=_http_endpoint(otlp.endpoint, _TRACES_PATH),
        headers=dict(otlp.headers),
        timeout=otlp.timeout,
    )


def _build_metric_exporter(config: Any) -> Any:
    otlp = config.otlp
    if otlp.protocol == "grpc":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )
        except Exception as exc:
            raise ExportError(
                "opentelemetry-exporter-otlp-proto-grpc is not installed; "
                "install the 'observability' extra or set PROMERAG_OTLP_PROTOCOL=http"
            ) from exc
        return OTLPMetricExporter(
            endpoint=otlp.endpoint,
            insecure=otlp.insecure,
            headers=dict(otlp.headers),
            timeout=otlp.timeout,
            max_export_batch_size=otlp.batch_size,
        )
    try:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
    except Exception as exc:
        raise ExportError(
            "opentelemetry-exporter-otlp-proto-http is not installed; "
            "install the 'observability' extra or set PROMERAG_OTLP_PROTOCOL=grpc"
        ) from exc
    return OTLPMetricExporter(
        endpoint=_http_endpoint(otlp.endpoint, _METRICS_PATH),
        headers=dict(otlp.headers),
        timeout=otlp.timeout,
    )


def _build_sampler(config: Any) -> Any:
    from ..configuration import _build_sampler as build_sampler

    return build_sampler(config)


def build_otlp_tracer_provider(
    config: Any,
    *,
    resource: Any | None = None,
    sampler: Any | None = None,
) -> Any:
    """TracerProvider with a batching OTLP span exporter attached."""
    _validate(config)
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except Exception as exc:
        raise ExportError(f"opentelemetry-sdk is not available: {exc}") from exc

    provider = TracerProvider(
        sampler=sampler if sampler is not None else _build_sampler(config),
        resource=resource,
    )
    provider.add_span_processor(
        BatchSpanProcessor(
            _build_span_exporter(config),
            max_export_batch_size=config.otlp.batch_size,
            schedule_delay_millis=config.otlp.schedule_delay_ms,
        )
    )
    logger.debug("otlp span exporter configured: %s", otlp_export_summary(config))
    return provider


def build_otlp_meter_provider(
    config: Any,
    *,
    resource: Any | None = None,
) -> Any:
    """MeterProvider with a periodic OTLP metric exporter attached."""
    _validate(config)
    try:
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    except Exception as exc:
        raise ExportError(f"opentelemetry-sdk metrics are not available: {exc}") from exc

    reader = PeriodicExportingMetricReader(
        _build_metric_exporter(config),
        export_interval_millis=config.otlp.metric_export_interval_ms,
        export_timeout_millis=int(config.otlp.timeout * 1000),
    )
    logger.debug("otlp metric exporter configured: %s", otlp_export_summary(config))
    return MeterProvider(metric_readers=[reader], resource=resource)
