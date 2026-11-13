"""Protocol definitions for observability components.

Protocols enable dependency injection and keep every consumer able to
swap implementations (no-op, OpenTelemetry-backed, in-memory test
doubles) without static coupling.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from .models import SpanKind, TraceContext


@runtime_checkable
class TracerProtocol(Protocol):
    """Protocol for starting spans."""

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Mapping[str, Any] | None = None,
        parent: TraceContext | None = None,
    ) -> Any:  # context manager yielding the active span scope
        ...


@runtime_checkable
class CounterProtocol(Protocol):
    """Monotonically increasing counter instrument."""

    def add(self, amount: float, labels: Mapping[str, str] | None = None) -> None:
        ...

    def inc(self, labels: Mapping[str, str] | None = None) -> None:
        ...


@runtime_checkable
class GaugeProtocol(Protocol):
    """Last-value-wins gauge instrument."""

    def set(self, value: float, labels: Mapping[str, str] | None = None) -> None:
        ...


@runtime_checkable
class UpDownCounterProtocol(Protocol):
    """Additive-up/down instrument for levels such as active requests."""

    def add(self, amount: float, labels: Mapping[str, str] | None = None) -> None:
        ...


@runtime_checkable
class HistogramProtocol(Protocol):
    """Distribution instrument for durations and sizes."""

    def record(self, value: float, labels: Mapping[str, str] | None = None) -> None:
        ...


@runtime_checkable
class MetricsRegistryProtocol(Protocol):
    """Factory for typed metric instruments."""

    def counter(
        self,
        name: str,
        description: str = "",
        unit: str | None = None,
    ) -> CounterProtocol:
        ...

    def gauge(
        self,
        name: str,
        description: str = "",
        unit: str | None = None,
    ) -> GaugeProtocol:
        ...

    def up_down_counter(
        self,
        name: str,
        description: str = "",
        unit: str | None = None,
    ) -> UpDownCounterProtocol:
        ...

    def histogram(
        self,
        name: str,
        description: str = "",
        unit: str | None = None,
        buckets: list[float] | None = None,
    ) -> HistogramProtocol:
        ...


@runtime_checkable
class ExporterProtocol(Protocol):
    """Lifecycle for telemetry exporters."""

    def shutdown(self) -> None:
        ...


@runtime_checkable
class LoggingHandlerProtocol(Protocol):
    """Sink for structured log records."""

    def handle_event(self, event: Mapping[str, Any]) -> None:
        ...


@runtime_checkable
class InstrumentationHandlerProtocol(Protocol):
    """Apply and remove instrumentation around a component."""

    def instrument(self) -> Any:
        ...

    def uninstrument(self) -> None:
        ...


__all__ = [
    "CounterProtocol",
    "ExporterProtocol",
    "GaugeProtocol",
    "HistogramProtocol",
    "InstrumentationHandlerProtocol",
    "LoggingHandlerProtocol",
    "MetricsRegistryProtocol",
    "TracerProtocol",
    "UpDownCounterProtocol",
]
