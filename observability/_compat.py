"""Lazy optional-dependency detection for OpenTelemetry.

Importing this module never imports OpenTelemetry. Consumers call the
loader helpers, which return the imported module or ``None`` when the
dependency is not installed. No global state is modified.
"""
from __future__ import annotations

import importlib
from types import ModuleType


def _try_import(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except Exception:
        # ImportError is the expected case (OTel not installed). Any other
        # failure means OTel is unusable in this environment; treat it the
        # same way rather than crashing application startup.
        return None


def otel_api_available() -> bool:
    return _try_import("opentelemetry.trace") is not None and _try_import(
        "opentelemetry.metrics"
    ) is not None


def otel_sdk_available() -> bool:
    return _try_import("opentelemetry.sdk.trace") is not None and _try_import(
        "opentelemetry.sdk.metrics"
    ) is not None


def otel_trace() -> ModuleType | None:
    return _try_import("opentelemetry.trace")


def otel_metrics() -> ModuleType | None:
    return _try_import("opentelemetry.metrics")


def otel_sdk_trace() -> ModuleType | None:
    return _try_import("opentelemetry.sdk.trace")


def otel_sdk_metrics() -> ModuleType | None:
    return _try_import("opentelemetry.sdk.metrics")


def otel_sdk_resources() -> ModuleType | None:
    return _try_import("opentelemetry.sdk.resources")


def otel_trace_sampling() -> ModuleType | None:
    return _try_import("opentelemetry.sdk.trace.sampling")
