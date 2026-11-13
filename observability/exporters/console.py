"""Console exporter: print spans/metrics to stdout for local development.

Uses the OpenTelemetry SDK console exporters. No collector, server, or
background service is started. Spans are printed when they end (simple
processor); metrics are pulled on a fixed interval (8s) by a periodic
reader, or on provider shutdown.
"""
from __future__ import annotations

import logging
import sys
from typing import Any, TextIO

logger = logging.getLogger("prome_rag.exporters.console")


def configure_console_exporter(
    config: Any,
    *,
    resource: Any | None = None,
    out: TextIO | None = None,
) -> Any | None:
    """Build a TracerProvider printing spans to ``out`` (default: current stdout).

    ``config`` is an :class:`ObservabilityConfig`. Returns ``None`` when the
    OpenTelemetry SDK is unavailable (callers fall back to no-op).
    """
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )
    except Exception as exc:  # pragma: no cover - depends on environment
        logger.warning("opentelemetry-sdk not available; console exporter disabled: %s", exc)
        return None

    # Resolve stdout at call time so redirected streams (tests, notebooks,
    # embedded runtimes) receive the spans.
    stream = out if out is not None else sys.stdout
    provider = TracerProvider(sampler=_build_sampler(config), resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter(out=stream)))
    logger.debug("console span exporter configured")
    return provider


def build_console_meter_provider(
    config: Any,
    *,
    resource: Any | None = None,
    out: TextIO | None = None,
) -> Any | None:
    """Build a MeterProvider that prints aggregated metrics to ``out`` (default stdout)."""
    try:
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import (
            ConsoleMetricExporter,
            PeriodicExportingMetricReader,
        )
    except Exception as exc:  # pragma: no cover - depends on environment
        logger.warning("opentelemetry-sdk metrics not available; console metrics disabled: %s", exc)
        return None

    stream = out if out is not None else sys.stdout
    reader = PeriodicExportingMetricReader(
        ConsoleMetricExporter(out=stream),
        export_interval_millis=getattr(config.otlp, "metric_export_interval_ms", 10000),
    )
    return MeterProvider(metric_readers=[reader], resource=resource)


def _build_sampler(config: Any) -> Any:
    try:
        from ..configuration import _build_sampler as build_sampler

        return build_sampler(config)
    except Exception:
        return None
