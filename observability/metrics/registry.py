"""Metric registry: creates and caches instruments for one MeterProvider.

The registry never registers anything with OpenTelemetry globals; it
uses the provider owned by the initialized runtime (or a provider handed
to it directly, as the test-suite does).

Modes per instrument:

- OTel instrument when a provider is available;
- in-memory fallback when observability is enabled but the SDK is not
  installed (keeps records inspectable via :meth:`MetricRegistry.snapshot`);
- no-op otherwise (default before init and when metrics are disabled).
"""
from __future__ import annotations

import logging

from ._labels import sanitize_metric_labels  # re-exported convenience
from .counters import Counter, UpDownCounter
from .gauges import Gauge
from .histograms import Histogram

logger = logging.getLogger("prome_rag.metrics")


class MetricRegistry:
    """Creates instruments for a single meter; safe to use as a cache."""

    def __init__(
        self,
        meter_provider=None,
        *,
        name: str = "prome-rag",
        enabled: bool = True,
        fallback: bool = False,
        meter=None,
    ) -> None:
        self._provider = meter_provider
        self._name = name
        self._enabled = enabled
        self._fallback = fallback
        self._meter = meter
        self._instruments: dict[tuple[str, str], object] = {}

    # -- meter resolution --------------------------------------------------

    def _get_meter(self):
        if self._meter is not None:
            return self._meter
        provider = self._provider
        if provider is None:
            return None
        try:
            self._meter = provider.get_meter(self._name)
        except Exception as exc:
            logger.debug("failed to acquire meter: %s", exc)
            self._meter = None
        return self._meter

    # -- instrument factories ---------------------------------------------

    def counter(
        self,
        name: str,
        description: str = "",
        unit: str | None = None,
    ) -> Counter:
        return self._build("counter", name, description, unit)

    def up_down_counter(
        self,
        name: str,
        description: str = "",
        unit: str | None = None,
    ) -> UpDownCounter:
        return self._build("up_down_counter", name, description, unit)

    def gauge(
        self,
        name: str,
        description: str = "",
        unit: str | None = None,
    ) -> Gauge:
        return self._build("gauge", name, description, unit)

    def histogram(
        self,
        name: str,
        description: str = "",
        unit: str | None = None,
        buckets: list[float] | None = None,
    ) -> Histogram:
        return self._build("histogram", name, description, unit, buckets)

    def _build(self, kind: str, name: str, description: str, unit: str | None, buckets=None):
        key = (kind, name)
        cached = self._instruments.get(key)
        if cached is not None:
            return cached

        instrument = None
        if self._enabled:
            meter = self._get_meter()
            if meter is not None:
                try:
                    if kind == "counter":
                        instrument = meter.create_counter(name, unit=unit or "", description=description)
                    elif kind == "up_down_counter":
                        instrument = meter.create_up_down_counter(name, unit=unit or "", description=description)
                    elif kind == "gauge":
                        instrument = meter.create_gauge(name, unit=unit or "", description=description)
                    elif kind == "histogram":
                        kwargs = {}
                        if buckets:
                            kwargs["explicit_bucket_boundaries_advisory"] = buckets
                        instrument = meter.create_histogram(name, unit=unit or "", description=description, **kwargs)
                except Exception as exc:
                    logger.debug("failed to create %s instrument %r: %s", kind, name, exc)
                    instrument = None

        use_fallback = instrument is None and self._enabled and self._fallback
        if kind in ("counter", "up_down_counter"):
            built = UpDownCounter(instrument, name=name, fallback=use_fallback) if kind == "up_down_counter" else Counter(instrument, name=name, fallback=use_fallback)
        elif kind == "gauge":
            built = Gauge(instrument, name=name, fallback=use_fallback)
        else:
            built = Histogram(instrument, name=name, fallback=use_fallback)
        self._instruments[key] = built
        return built

    # -- introspection -----------------------------------------------------

    def snapshot(self) -> dict:
        """Aggregate in-memory fallback data (empty when OTel-backed)."""
        data: dict[str, object] = {}
        for (kind, name), instrument in self._instruments.items():
            snap = getattr(instrument, "snapshot", list)()
            if snap:
                data.setdefault(kind, {})[name] = snap
        return data


def default_registry() -> MetricRegistry:
    """Return the registry bound to the current runtime.

    - No runtime, or ``PROMERAG_OBSERVABILITY_ENABLED=false`` /
      ``PROMERAG_METRICS_ENABLED=false`` -> no-op instruments.
    - OTel SDK available -> instruments backed by the runtime's
      MeterProvider (which may carry no reader when no exporter is set).
    - SDK unavailable -> in-memory fallback so counters remain
      introspectable via :meth:`MetricRegistry.snapshot`.

    The registry is cached on the runtime object, so repeated calls share
    instruments.
    """
    from ..configuration import get_runtime

    runtime = get_runtime()
    if (
        runtime is None
        or not runtime.config.enabled
        or not runtime.config.metrics_enabled
    ):
        return MetricRegistry(enabled=False)

    cached = getattr(runtime, "_metric_registry", None)
    if cached is not None:
        return cached

    from .._compat import otel_sdk_available

    sdk_available = otel_sdk_available()
    if runtime.meter_provider is not None and sdk_available:
        provider = runtime.meter_provider
        fallback = False
    elif not sdk_available:
        provider = None
        fallback = True
    else:
        # SDK present but no meter provider was built (e.g. metrics disabled
        # at provider level); fall back to a plain enabled registry.
        provider = runtime.meter_provider
        fallback = False

    registry = MetricRegistry(
        provider,
        name=runtime.config.service_name,
        enabled=True,
        fallback=fallback,
    )
    runtime._metric_registry = registry  # type: ignore[attr-defined]
    return registry


__all__ = ["MetricRegistry", "default_registry", "sanitize_metric_labels"]
