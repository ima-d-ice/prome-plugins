"""Integration checks between the observability and loaders packages.

Uses the in-memory OpenTelemetry providers from ``conftest.py`` and real
``loaders`` components (not duck-typed fakes) to verify the seam exposed
by ``observability.instrumentation.integrations.instrument_loader``.
"""

from __future__ import annotations

import pytest

from loaders.filesystem.file_loader import FileLoader
from observability.instrumentation.integrations import instrument_loader
from observability.metrics import MetricNames

from .conftest import metric_data_points


def test_instrument_loader_wraps_real_file_loader_success(memory_otel, tmp_path):
    target = tmp_path / "doc.txt"
    target.write_text("hello prome", encoding="utf-8")

    wrapped = instrument_loader(FileLoader())
    result = wrapped.load(target)

    assert result.resources, "real FileLoader should load the file"
    assert result.resources[0].content == b"hello prome"

    spans = memory_otel.span_exporter.get_finished_spans()
    assert any(span.name == "ingestion.load" for span in spans)

    data = memory_otel.metric_reader.get_metrics_data()
    assert metric_data_points(data, MetricNames.INGESTION_DOCUMENTS_LOADED)


def test_instrument_loader_wraps_real_file_loader_error(memory_otel, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("nope", encoding="utf-8")

    wrapped = instrument_loader(FileLoader(permitted_roots=[allowed]))
    from loaders.contracts.exceptions import PathTraversalError

    with pytest.raises(PathTraversalError):
        wrapped.load(outside)

    data = memory_otel.metric_reader.get_metrics_data()
    assert metric_data_points(data, MetricNames.INGESTION_DOCUMENTS_REJECTED)
