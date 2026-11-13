"""Configuration parsing and validation."""
from __future__ import annotations

import pytest

from observability.configuration import (
    EXPORTER_CONSOLE,
    EXPORTER_NONE,
    EXPORTER_OTLP,
    ObservabilityConfig,
    init_observability,
    parse_otlp_headers,
    shutdown_observability,
)
from observability.contracts.exceptions import ConfigurationError


def make_env(**overrides):
    env = {}
    env.update(overrides)
    return env


def test_defaults_are_safe():
    config = ObservabilityConfig.from_env({})
    assert config.enabled is False
    assert config.exporter == EXPORTER_NONE
    assert config.service_name == "prome-rag"
    assert config.log_level == "INFO"
    assert config.log_format == "json"
    assert config.trace_sampling == "parentbased_always_on"
    assert config.record_prompts is False
    assert config.record_document_content is False
    assert config.metrics_enabled is True


def test_explicit_values_parse():
    config = ObservabilityConfig.from_env(
        {
            "PROMERAG_OBSERVABILITY_ENABLED": "true",
            "PROMERAG_EXPORTER": "otlp",
            "PROMERAG_OTEL_SERVICE_NAME": "my-service",
            "PROMERAG_OTLP_ENDPOINT": "http://localhost:4317",
            "PROMERAG_OTLP_PROTOCOL": "grpc",
            "PROMERAG_OTLP_TIMEOUT": "5.5",
            "PROMERAG_OTLP_HEADERS": "authorization=Bearer abc,x-tenant=red",
            "PROMERAG_LOG_LEVEL": "debug",
            "PROMERAG_LOG_FORMAT": "text",
            "PROMERAG_TRACE_SAMPLING": "parentbased_traceidratio",
            "PROMERAG_TRACE_SAMPLING_RATIO": "0.25",
            "PROMERAG_RECORD_PROMPTS": "true",
            "PROMERAG_METRICS_ENABLED": "false",
        }
    )
    assert config.enabled is True
    assert config.service_name == "my-service"
    assert config.exporter == EXPORTER_OTLP
    assert config.otlp.endpoint == "http://localhost:4317"
    assert config.otlp.timeout == 5.5
    assert config.otlp.headers == {"authorization": "Bearer abc", "x-tenant": "red"}
    assert config.log_level == "DEBUG"
    assert config.log_format == "text"
    assert config.trace_sampling_ratio == 0.25
    assert config.record_prompts is True
    assert config.metrics_enabled is False


def test_invalid_values_are_collected_with_context():
    with pytest.raises(ConfigurationError) as excinfo:
        ObservabilityConfig.from_env(
            {
                "PROMERAG_OBSERVABILITY_ENABLED": "maybe",
                "PROMERAG_EXPORTER": "datadog",
                "PROMERAG_LOG_LEVEL": "CHATTY",
                "PROMERAG_LOG_FORMAT": "yaml",
                "PROMERAG_TRACE_SAMPLING": "sometimes",
                "PROMERAG_TRACE_SAMPLING_RATIO": "7",
                "PROMERAG_MAX_ATTRIBUTE_LENGTH": "4",
            }
        )
    message = str(excinfo.value)
    assert "PROMERAG_OBSERVABILITY_ENABLED" in message
    assert "PROMERAG_EXPORTER" in message
    assert "PROMERAG_LOG_LEVEL" in message
    assert "PROMERAG_LOG_FORMAT" in message
    assert "PROMERAG_TRACE_SAMPLING" in message
    assert "PROMERAG_TRACE_SAMPLING_RATIO" in message
    assert "PROMERAG_MAX_ATTRIBUTE_LENGTH" in message


def test_otlp_requires_endpoint():
    with pytest.raises(ConfigurationError) as excinfo:
        ObservabilityConfig.from_env({"PROMERAG_EXPORTER": "otlp"})
    assert "PROMERAG_OTLP_ENDPOINT" in str(excinfo.value)


def test_bool_parsing_accepts_common_spellings():
    for value in ("true", "1", "yes", "on", "TRUE", "On"):
        assert ObservabilityConfig.from_env({"PROMERAG_OBSERVABILITY_ENABLED": value}).enabled is True
    for value in ("false", "0", "no", "off", "FALSE"):
        assert ObservabilityConfig.from_env({"PROMERAG_OBSERVABILITY_ENABLED": value}).enabled is False


def test_parse_otlp_headers_rejects_malformed_entries():
    with pytest.raises(ConfigurationError):
        parse_otlp_headers("not-a-pair")
    assert parse_otlp_headers("") == {}
    assert parse_otlp_headers("k=v") == {"k": "v"}
    assert parse_otlp_headers("a=b=c") == {"a": "b=c"}  # values may contain '='


def test_safe_dict_redacts_header_values():
    config = ObservabilityConfig.from_env(
        {
            "PROMERAG_EXPORTER": "otlp",
            "PROMERAG_OTLP_ENDPOINT": "http://localhost:4317",
            "PROMERAG_OTLP_HEADERS": "authorization=Bearer supersecret",
        }
    )
    safe = config.safe_dict()
    assert safe["otlp"]["header_keys"] == ["authorization"]
    assert "supersecret" not in str(safe)


def test_validate_rejects_manual_construction():
    config = ObservabilityConfig(exporter="otlp")
    with pytest.raises(ConfigurationError):
        config.validate()


def test_console_mode_initializes_without_collector():
    runtime = init_observability(ObservabilityConfig(enabled=True, exporter=EXPORTER_CONSOLE))
    assert runtime.enabled is True
    shutdown_observability()


def test_no_endpoint_config_init_with_defaults():
    runtime = init_observability(ObservabilityConfig.from_env({}))
    assert runtime.enabled is False
    shutdown_observability()
