"""Environment-driven configuration and explicit lifecycle for observability.

Importing this module has no side effects: no providers are created, no
loggers are configured, and no OpenTelemetry package is imported until
:func:`init_observability` is called (or a lazy helper is used).

Operating modes selected by ``PROMERAG_EXPORTER``:

- ``none``    — default; telemetry is recorded to nowhere (no-op).
- ``console`` — spans/metrics are printed to stdout (local development).
- ``otlp``    — spans/metrics are batched to an explicitly configured
                OpenTelemetry collector.

Tracing and metrics are active only when ``PROMERAG_OBSERVABILITY_ENABLED=true``
*and* the OpenTelemetry SDK is installed. Structured logging is configured
independently of that flag.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ._compat import otel_sdk_available
from .contracts.exceptions import ConfigurationError
from .contracts.models import ExportConfig

ENV_PREFIX = "PROMERAG_"

EXPORTER_NONE = "none"
EXPORTER_CONSOLE = "console"
EXPORTER_OTLP = "otlp"
VALID_EXPORTERS = (EXPORTER_NONE, EXPORTER_CONSOLE, EXPORTER_OTLP)

VALID_PROTOCOLS = ("grpc", "http")
VALID_LOG_FORMATS = ("json", "text")
VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
VALID_SAMPLINGS = (
    "parentbased_always_on",
    "parentbased_always_off",
    "always_on",
    "always_off",
    "traceidratio",
    "parentbased_traceidratio",
)

_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off"}

_HEADER_SPLIT = re.compile(r"\s*,\s*")
_HEADER_ITEM = re.compile(r"^\s*([^:=\s][^=]*?)\s*=\s*(.*)$")


def _parse_bool(raw: str, name: str, errors: list[str]) -> bool:
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    errors.append(f"{name} must be a boolean (true/false/1/0/yes/no/on/off), got {raw!r}")
    return False


def _parse_float(raw: str, name: str, errors: list[str], minimum=None, maximum=None) -> float:
    try:
        value = float(raw.strip())
    except ValueError:
        errors.append(f"{name} must be a number, got {raw!r}")
        return 0.0
    if minimum is not None and value < minimum:
        errors.append(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        errors.append(f"{name} must be <= {maximum}, got {value}")
    return value


def _parse_int(raw: str, name: str, errors: list[str], minimum=None, maximum=None) -> int:
    try:
        value = int(raw.strip())
    except ValueError:
        errors.append(f"{name} must be an integer, got {raw!r}")
        return 0
    if minimum is not None and value < minimum:
        errors.append(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        errors.append(f"{name} must be <= {maximum}, got {value}")
    return value


def parse_otlp_headers(raw: str) -> dict[str, str]:
    """Parse ``key=value,key2=value2`` header definitions.

    Values may contain ``=``; keys are trimmed and must be non-empty.
    Raises :class:`ConfigurationError` on malformed input.
    """
    headers: dict[str, str] = {}
    if not raw or not raw.strip():
        return headers
    for chunk in _HEADER_SPLIT.split(raw):
        if not chunk:
            continue
        match = _HEADER_ITEM.match(chunk)
        if not match:
            raise ConfigurationError(
                f"PROMERAG_OTLP_HEADERS entry {chunk!r} is not of the form key=value"
            )
        key, value = match.group(1).strip(), match.group(2).strip()
        if not key:
            raise ConfigurationError("PROMERAG_OTLP_HEADERS contains an empty key")
        headers[key] = value
    return headers


@dataclass(frozen=True)
class ObservabilityConfig:
    """Validated observability configuration.

    Construct with :meth:`from_env` (or defaults) and let
    :func:`init_observability` apply it. ``validate()`` is called during
    initialization; you may call it yourself for fail-fast checks.
    """

    enabled: bool = False
    service_name: str = "prome-rag"
    exporter: str = EXPORTER_NONE
    otlp: ExportConfig = field(default_factory=ExportConfig)
    metrics_enabled: bool = True
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: str = ""
    trace_sampling: str = "parentbased_always_on"
    trace_sampling_ratio: float = 1.0
    record_prompts: bool = False
    record_document_content: bool = False
    diagnostic_mode: bool = False
    strict_cardinality: bool = False
    max_attribute_length: int = 512

    # -- construction ----------------------------------------------------

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ObservabilityConfig:
        env = os.environ if environ is None else environ
        errors: list[str] = []

        def get(name: str, default: str = "") -> str:
            return env.get(ENV_PREFIX + name, default)

        enabled = _parse_bool(get("OBSERVABILITY_ENABLED", "false"), "PROMERAG_OBSERVABILITY_ENABLED", errors)
        service_name = get("OTEL_SERVICE_NAME", "prome-rag").strip() or "prome-rag"
        exporter = get("EXPORTER", EXPORTER_NONE).strip().lower() or EXPORTER_NONE
        if exporter not in VALID_EXPORTERS:
            errors.append(f"PROMERAG_EXPORTER must be one of {VALID_EXPORTERS}, got {exporter!r}")
        protocol = get("OTLP_PROTOCOL", "grpc").strip().lower() or "grpc"
        if protocol not in VALID_PROTOCOLS:
            errors.append(f"PROMERAG_OTLP_PROTOCOL must be one of {VALID_PROTOCOLS}, got {protocol!r}")
        otlp_endpoint = get("OTLP_ENDPOINT", "").strip()
        try:
            otlp_headers = parse_otlp_headers(get("OTLP_HEADERS", ""))
        except ConfigurationError as exc:
            errors.append(str(exc))
            otlp_headers = {}
        otlp_timeout = _parse_float(get("OTLP_TIMEOUT", "10.0"), "PROMERAG_OTLP_TIMEOUT", errors, minimum=0.1, maximum=300.0)
        otlp_insecure = _parse_bool(get("OTLP_INSECURE", "false"), "PROMERAG_OTLP_INSECURE", errors)
        otlp_batch_size = _parse_int(get("OTLP_BATCH_SIZE", "512"), "PROMERAG_OTLP_BATCH_SIZE", errors, minimum=1, maximum=10000)
        otlp_schedule_delay_ms = _parse_int(get("OTLP_SCHEDULE_DELAY_MS", "5000"), "PROMERAG_OTLP_SCHEDULE_DELAY_MS", errors, minimum=100, maximum=3600000)
        metric_export_interval_ms = _parse_int(get("METRICS_EXPORT_INTERVAL_MS", "10000"), "PROMERAG_METRICS_EXPORT_INTERVAL_MS", errors, minimum=100, maximum=3600000)
        metrics_enabled = _parse_bool(get("METRICS_ENABLED", "true"), "PROMERAG_METRICS_ENABLED", errors)
        log_level = get("LOG_LEVEL", "INFO").strip().upper() or "INFO"
        if log_level not in VALID_LOG_LEVELS:
            errors.append(f"PROMERAG_LOG_LEVEL must be one of {VALID_LOG_LEVELS}, got {log_level!r}")
        log_format = get("LOG_FORMAT", "json").strip().lower() or "json"
        if log_format not in VALID_LOG_FORMATS:
            errors.append(f"PROMERAG_LOG_FORMAT must be one of {VALID_LOG_FORMATS}, got {log_format!r}")
        log_file = get("LOG_FILE", "").strip()
        trace_sampling = get("TRACE_SAMPLING", "parentbased_always_on").strip().lower() or "parentbased_always_on"
        if trace_sampling not in VALID_SAMPLINGS:
            errors.append(f"PROMERAG_TRACE_SAMPLING must be one of {VALID_SAMPLINGS}, got {trace_sampling!r}")
        trace_sampling_ratio = _parse_float(get("TRACE_SAMPLING_RATIO", "1.0"), "PROMERAG_TRACE_SAMPLING_RATIO", errors, minimum=0.0, maximum=1.0)
        record_prompts = _parse_bool(get("RECORD_PROMPTS", "false"), "PROMERAG_RECORD_PROMPTS", errors)
        record_document_content = _parse_bool(get("RECORD_DOCUMENT_CONTENT", "false"), "PROMERAG_RECORD_DOCUMENT_CONTENT", errors)
        diagnostic_mode = _parse_bool(get("DIAGNOSTIC_MODE", "false"), "PROMERAG_DIAGNOSTIC_MODE", errors)
        strict_cardinality = _parse_bool(get("STRICT_CARDINALITY", "false"), "PROMERAG_STRICT_CARDINALITY", errors)
        max_attribute_length = _parse_int(get("MAX_ATTRIBUTE_LENGTH", "512"), "PROMERAG_MAX_ATTRIBUTE_LENGTH", errors, minimum=32, maximum=8192)

        if exporter == EXPORTER_OTLP and not otlp_endpoint:
            errors.append("PROMERAG_OTLP_ENDPOINT must be set when PROMERAG_EXPORTER=otlp")

        if errors:
            raise ConfigurationError(
                "Invalid observability configuration:\n- " + "\n- ".join(errors)
            )

        return cls(
            enabled=enabled,
            service_name=service_name,
            exporter=exporter,
            otlp=ExportConfig(
                endpoint=otlp_endpoint,
                protocol=protocol,
                headers=otlp_headers,
                timeout=otlp_timeout,
                insecure=otlp_insecure,
                batch_size=otlp_batch_size,
                schedule_delay_ms=otlp_schedule_delay_ms,
                metric_export_interval_ms=metric_export_interval_ms,
            ),
            metrics_enabled=metrics_enabled,
            log_level=log_level,
            log_format=log_format,
            log_file=log_file,
            trace_sampling=trace_sampling,
            trace_sampling_ratio=trace_sampling_ratio,
            record_prompts=record_prompts,
            record_document_content=record_document_content,
            diagnostic_mode=diagnostic_mode,
            strict_cardinality=strict_cardinality,
            max_attribute_length=max_attribute_length,
        )

    def validate(self) -> None:
        if not self.service_name.strip():
            raise ConfigurationError("PROMERAG_OTEL_SERVICE_NAME must not be empty")
        if self.exporter not in VALID_EXPORTERS:
            raise ConfigurationError(f"PROMERAG_EXPORTER must be one of {VALID_EXPORTERS}")
        if self.otlp.protocol not in VALID_PROTOCOLS:
            raise ConfigurationError(f"PROMERAG_OTLP_PROTOCOL must be one of {VALID_PROTOCOLS}")
        if self.exporter == EXPORTER_OTLP and not self.otlp.endpoint:
            raise ConfigurationError("PROMERAG_OTLP_ENDPOINT must be set when PROMERAG_EXPORTER=otlp")
        if self.log_level not in VALID_LOG_LEVELS:
            raise ConfigurationError(f"PROMERAG_LOG_LEVEL must be one of {VALID_LOG_LEVELS}")
        if self.log_format not in VALID_LOG_FORMATS:
            raise ConfigurationError(f"PROMERAG_LOG_FORMAT must be one of {VALID_LOG_FORMATS}")
        if self.trace_sampling not in VALID_SAMPLINGS:
            raise ConfigurationError(f"PROMERAG_TRACE_SAMPLING must be one of {VALID_SAMPLINGS}")
        if not 0.0 <= self.trace_sampling_ratio <= 1.0:
            raise ConfigurationError("PROMERAG_TRACE_SAMPLING_RATIO must be within [0, 1]")
        if self.otlp.timeout <= 0:
            raise ConfigurationError("PROMERAG_OTLP_TIMEOUT must be positive")
        if self.max_attribute_length < 32:
            raise ConfigurationError("PROMERAG_MAX_ATTRIBUTE_LENGTH must be >= 32")

    # -- redacted introspection ------------------------------------------

    def safe_dict(self) -> dict[str, Any]:
        """Configuration snapshot safe to log: header values are redacted."""
        return {
            "enabled": self.enabled,
            "service_name": self.service_name,
            "exporter": self.exporter,
            "metrics_enabled": self.metrics_enabled,
            "log_level": self.log_level,
            "log_format": self.log_format,
            "log_file": self.log_file,
            "trace_sampling": self.trace_sampling,
            "trace_sampling_ratio": self.trace_sampling_ratio,
            "record_prompts": self.record_prompts,
            "record_document_content": self.record_document_content,
            "diagnostic_mode": self.diagnostic_mode,
            "strict_cardinality": self.strict_cardinality,
            "otlp": {
                "endpoint": self.otlp.endpoint,
                "protocol": self.otlp.protocol,
                "timeout": self.otlp.timeout,
                "insecure": self.otlp.insecure,
                "header_keys": sorted(self.otlp.headers),
                "batch_size": self.otlp.batch_size,
                "schedule_delay_ms": self.otlp.schedule_delay_ms,
                "metric_export_interval_ms": self.otlp.metric_export_interval_ms,
            },
        }


class ObservabilityRuntime:
    """Handle for an initialized observability stack.

    Created by :func:`init_observability`; never created at import time.
    Providers produced here are kept in this object and are *not*
    registered in the global OpenTelemetry proxy provider by default, so
    importing this package cannot leak telemetry into unrelated libraries
    or other worktrees.
    """

    def __init__(
        self,
        config: ObservabilityConfig,
        *,
        tracer_provider: Any = None,
        meter_provider: Any = None,
        logging_handlers: tuple[logging.Handler, ...] = (),
        otel_active: bool = False,
    ) -> None:
        self.config = config
        self.tracer_provider = tracer_provider
        self.meter_provider = meter_provider
        self.logging_handlers = logging_handlers
        self.otel_active = otel_active
        self._shutdown = False

    @property
    def enabled(self) -> bool:
        return self.config.enabled and self.otel_active and not self._shutdown

    @property
    def metrics_active(self) -> bool:
        return self.enabled and self.config.metrics_enabled

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        logger = logging.getLogger("prome_rag.observability")
        for provider in (self.tracer_provider, self.meter_provider):
            if provider is None:
                continue
            try:
                provider.shutdown()
            except Exception as exc:  # shutdown must never raise
                logger.warning("error shutting down telemetry provider: %s", exc)
        self.tracer_provider = None
        self.meter_provider = None
        root = logging.getLogger("prome_rag")
        for handler in self.logging_handlers:
            try:
                root.removeHandler(handler)
                handler.close()
            except Exception as exc:
                logger.warning("error removing logging handler: %s", exc)
        self.logging_handlers = ()


_state_lock = threading.Lock()
_runtime: ObservabilityRuntime | None = None


def get_runtime() -> ObservabilityRuntime | None:
    return _runtime


def get_config() -> ObservabilityConfig:
    """Return the active config, or the env-derived default if not initialized."""
    runtime = _runtime
    if runtime is not None:
        return runtime.config
    return _UNSET_CONFIG


def is_enabled() -> bool:
    runtime = _runtime
    return bool(runtime and runtime.enabled)


def metrics_active() -> bool:
    runtime = _runtime
    return bool(runtime and runtime.metrics_active)


# Lightweight sentinel so get_config() before init returns a valid object.
_UNSET_CONFIG = ObservabilityConfig()


def init_observability(
    config: ObservabilityConfig | None = None,
    *,
    tracer_provider: Any = None,
    meter_provider: Any = None,
    environ: Mapping[str, str] | None = None,
) -> ObservabilityRuntime:
    """Explicitly initialize the observability stack.

    Safe to call multiple times: an existing runtime is shut down first,
    which makes this deterministic in tests. Returns the runtime handle.

    ``tracer_provider``/``meter_provider`` are advanced injection seams
    used by the test-suite to wire in-memory SDK providers; when omitted,
    providers are built from ``config`` (or left unset in no-op mode).
    """
    global _runtime
    with _state_lock:
        if _runtime is not None:
            _runtime.shutdown()
            _runtime = None

        if config is None:
            config = ObservabilityConfig.from_env(environ)
        else:
            config.validate()

        from .logging.structured import configure_logging

        handlers = configure_logging(config)

        if config.diagnostic_mode:
            logging.getLogger("prome_rag.observability").warning(
                "diagnostic mode enabled; prompts and document content may be "
                "recorded only if RECORD_PROMPTS/RECORD_DOCUMENT_CONTENT are also true; "
                "secrets are still redacted by default"
            )

        otel_active = False
        built_tracer = tracer_provider
        built_meter = meter_provider
        providers_injected = tracer_provider is not None or meter_provider is not None

        if config.enabled and providers_injected:
            otel_active = True
        elif config.enabled:
            if not otel_sdk_available():
                logging.getLogger("prome_rag.observability").warning(
                    "PROMERAG_OBSERVABILITY_ENABLED=true but opentelemetry-sdk is not "
                    "installed; falling back to no-op telemetry"
                )
            else:
                from .exporters.console import (
                    build_console_meter_provider,
                    configure_console_exporter,
                )
                from .exporters.otlp import (
                    build_otlp_meter_provider,
                    build_otlp_tracer_provider,
                )

                resource = _build_resource(config)
                if config.exporter == EXPORTER_CONSOLE:
                    built_tracer = configure_console_exporter(config, resource=resource)
                    if config.metrics_enabled:
                        built_meter = build_console_meter_provider(config, resource=resource)
                elif config.exporter == EXPORTER_OTLP:
                    built_tracer = build_otlp_tracer_provider(config, resource=resource)
                    if config.metrics_enabled:
                        built_meter = build_otlp_meter_provider(config, resource=resource)
                else:
                    built_tracer = _build_tracer_provider(config, resource)
                    if config.metrics_enabled:
                        built_meter = _build_meter_provider(config, resource)
                otel_active = built_tracer is not None

        runtime = ObservabilityRuntime(
            config,
            tracer_provider=built_tracer,
            meter_provider=built_meter,
            logging_handlers=tuple(handlers),
            otel_active=otel_active,
        )
        _runtime = runtime
        logging.getLogger("prome_rag.observability").debug(
            "observability initialized: %s", config.safe_dict()
        )
        return runtime


def shutdown_observability() -> None:
    """Flush and tear down the initialized runtime, if any. Idempotent."""
    global _runtime
    with _state_lock:
        if _runtime is not None:
            _runtime.shutdown()
            _runtime = None


def _build_resource(config: ObservabilityConfig) -> Any:
    from ._compat import otel_sdk_resources

    resources = otel_sdk_resources()
    if resources is not None:
        return resources.Resource.create({"service.name": config.service_name})
    return None


def _build_sampler(config: ObservabilityConfig) -> Any:
    from ._compat import otel_trace_sampling

    sampling = otel_trace_sampling()
    if sampling is None:
        return None
    name = config.trace_sampling
    if name == "always_on":
        return sampling.ALWAYS_ON
    if name == "always_off":
        return sampling.ALWAYS_OFF
    if name == "parentbased_always_on":
        return sampling.ParentBased(sampling.ALWAYS_ON)
    if name == "parentbased_always_off":
        return sampling.ParentBased(sampling.ALWAYS_OFF)
    if name == "traceidratio":
        return sampling.TraceIdRatioBased(config.trace_sampling_ratio)
    if name == "parentbased_traceidratio":
        return sampling.ParentBasedTraceIdRatio(config.trace_sampling_ratio)
    return sampling.ALWAYS_ON


def _build_tracer_provider(config: ObservabilityConfig, resource: Any) -> Any:
    from ._compat import otel_sdk_trace

    sdk_trace = otel_sdk_trace()
    if sdk_trace is None:
        return None
    sampler = _build_sampler(config)
    return sdk_trace.TracerProvider(sampler=sampler, resource=resource)


def _build_meter_provider(config: ObservabilityConfig, resource: Any) -> Any:
    from ._compat import otel_sdk_metrics

    sdk_metrics = otel_sdk_metrics()
    if sdk_metrics is None:
        return None
    # No reader: instruments record into the SDK and are dropped on export.
    # This keeps code paths uniform without requiring a collector.
    return sdk_metrics.MeterProvider(metric_readers=[], resource=resource)
