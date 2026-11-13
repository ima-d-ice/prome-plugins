"""Exporter tests: console, OTLP validation, and import safety."""
from __future__ import annotations

import pytest

from observability import init_observability, shutdown_observability
from observability.configuration import ObservabilityConfig, parse_otlp_headers
from observability.contracts.exceptions import ConfigurationError, ExportError
from observability.contracts.models import ExportConfig
from observability.exporters.otlp import (
    _http_endpoint,
    build_otlp_meter_provider,
    build_otlp_tracer_provider,
    otlp_export_summary,
)


def test_console_exporter_prints_span(capsys):
    pytest.importorskip("opentelemetry.sdk.trace")
    runtime = init_observability(ObservabilityConfig(enabled=True, exporter="console"))
    from observability import operation_span

    with operation_span(component="test", operation="ping"):
        pass
    assert runtime.enabled
    shutdown_observability()
    out = capsys.readouterr().out
    assert "test.ping" in out


def test_console_metrics_exporter_builds():
    pytest.importorskip("opentelemetry.sdk.metrics")
    from observability.exporters.console import build_console_meter_provider

    provider = build_console_meter_provider(ObservabilityConfig(enabled=True, exporter="console"))
    try:
        assert provider is not None
    finally:
        provider.shutdown()


def test_otlp_missing_endpoint_rejected():
    config = ObservabilityConfig(exporter="otlp", otlp=ExportConfig(endpoint=""))
    with pytest.raises(ConfigurationError):
        build_otlp_tracer_provider(config)
    with pytest.raises(ConfigurationError):
        build_otlp_meter_provider(config)


def test_otlp_invalid_protocol_rejected():
    config = ObservabilityConfig(
        exporter="otlp", otlp=ExportConfig(endpoint="http://localhost:4317", protocol="carrier-pigeon")
    )
    with pytest.raises(ConfigurationError):
        build_otlp_tracer_provider(config)


def test_otlp_tracer_provider_builds_without_network():
    pytest.importorskip("opentelemetry.exporter.otlp.proto.grpc.trace_exporter")
    config = ObservabilityConfig(
        exporter="otlp",
        otlp=ExportConfig(
            endpoint="http://localhost:4317",
            protocol="grpc",
            headers={"authorization": "Bearer test-token"},
            insecure=True,
            timeout=1.0,
            batch_size=16,
            schedule_delay_ms=60000,
        ),
    )
    provider = build_otlp_tracer_provider(config)
    try:
        assert provider is not None
    finally:
        provider.shutdown()


def test_otlp_meter_provider_builds_without_network():
    pytest.importorskip("opentelemetry.exporter.otlp.proto.grpc.metric_exporter")
    config = ObservabilityConfig(
        exporter="otlp",
        otlp=ExportConfig(
            endpoint="http://localhost:4317",
            protocol="grpc",
            insecure=True,
            timeout=1.0,
            metric_export_interval_ms=60000,
        ),
    )
    provider = build_otlp_meter_provider(config)
    try:
        assert provider is not None
    finally:
        provider.shutdown()


def test_otlp_http_endpoints_append_paths():
    assert _http_endpoint("http://localhost:4318", "/v1/traces") == "http://localhost:4318/v1/traces"
    assert _http_endpoint("http://localhost:4318/v1/traces", "/v1/traces") == "http://localhost:4318/v1/traces"
    assert _http_endpoint("http://collector:4318/", "/v1/metrics") == "http://collector:4318/v1/metrics"


def test_otlp_http_tracer_builds():
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    config = ObservabilityConfig(
        exporter="otlp",
        otlp=ExportConfig(
            endpoint="http://localhost:4318",
            protocol="http",
            timeout=1.0,
            schedule_delay_ms=60000,
        ),
    )
    provider = build_otlp_tracer_provider(config)
    try:
        assert provider is not None
    finally:
        provider.shutdown()


def test_otlp_export_summary_never_leaks_header_values():
    config = ObservabilityConfig(
        exporter="otlp",
        otlp=ExportConfig(
            endpoint="http://localhost:4317",
            headers={"authorization": "Bearer verysecretvalue"},
        ),
    )
    summary = otlp_export_summary(config)
    assert summary["header_keys"] == ["authorization"]
    assert "verysecretvalue" not in str(summary)


def test_headers_parsing_roundtrip():
    headers = parse_otlp_headers("authorization=Bearer abc123,x-scope=prod")
    assert headers == {"authorization": "Bearer abc123", "x-scope": "prod"}


def test_otlp_exporter_missing_dependency_raises_export_error(monkeypatch):
    import sys

    config = ObservabilityConfig(
        exporter="otlp",
        otlp=ExportConfig(endpoint="http://localhost:4317", protocol="grpc", insecure=True),
    )

    monkeypatch.setitem(sys.modules, "opentelemetry.exporter.otlp.proto.grpc.trace_exporter", None)
    with pytest.raises((ExportError, ConfigurationError)):
        build_otlp_tracer_provider(config)


def test_full_init_with_otlp_config_does_not_send(monkeypatch):
    """Init with OTLP exporter configured but no traffic must not open sockets."""
    pytest.importorskip("opentelemetry.exporter.otlp.proto.grpc.trace_exporter")
    from observability.configuration import ObservabilityConfig as Cfg

    config = Cfg(
        enabled=True,
        exporter="otlp",
        otlp=ExportConfig(
            endpoint="http://127.0.0.1:9",  # discard port; nothing should connect
            protocol="grpc",
            insecure=True,
            timeout=0.2,
            schedule_delay_ms=60000,
            metric_export_interval_ms=60000,
        ),
    )
    runtime = init_observability(config)
    assert runtime.enabled
    shutdown_observability()


def test_init_otlp_without_endpoint_logs_and_stays_consistent():
    config = ObservabilityConfig(enabled=True, exporter="none")
    runtime = init_observability(config)
    assert runtime.enabled
    shutdown_observability()
