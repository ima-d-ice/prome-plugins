"""Integration tests: mock pipelines traced end-to-end (in-memory SDK)."""
from __future__ import annotations

from observability import operation_span
from observability.tracing import SpanNames


def _spans_by_name(spans):
    by_name: dict[str, list] = {}
    for span in spans:
        by_name.setdefault(span.name, []).append(span)
    return by_name


def _request_ids(spans):
    return {span.attributes.get("request_id") for span in spans}


def test_mock_pipeline_single_trace_and_request(memory_otel):
    with operation_span(component="orchestration", operation="pipeline"):
        with operation_span(component="ingestion", operation="parse"):
            with operation_span(component="ingestion", operation="chunk"):
                with operation_span(component="ingestion", operation="embed"):
                    pass
        with operation_span(component="retrieval", operation="dense"):
            with operation_span(component="retrieval", operation="rrf_fusion"):
                pass
        with operation_span(component="generation", operation="request"):
            pass
    spans = memory_otel.span_exporter.get_finished_spans()
    assert len(spans) == 7
    trace_ids = {span.context.trace_id for span in spans}
    assert len(trace_ids) == 1
    assert len(_request_ids(spans)) == 1


def test_ingestion_pipeline_span_tree(memory_otel):
    with operation_span(component="orchestration", operation="ingest"):
        with operation_span(component="ingestion", operation="discovery"):
            pass
        with operation_span(component="ingestion", operation="validate"):
            pass
        with operation_span(component="ingestion", operation="load"):
            pass
        with operation_span(component="ingestion", operation="parse"):
            pass
        with operation_span(component="ingestion", operation="chunk"):
            pass
        with operation_span(component="ingestion", operation="embed"):
            pass
        with operation_span(component="ingestion", operation="index_dense"):
            pass
        with operation_span(component="ingestion", operation="index_lexical"):
            pass
    spans = memory_otel.span_exporter.get_finished_spans()
    names = _spans_by_name(spans)
    for expected in (
        SpanNames.INGESTION_DISCOVERY,
        SpanNames.INGESTION_VALIDATE,
        SpanNames.INGESTION_LOAD,
        SpanNames.INGESTION_PARSE,
        SpanNames.INGESTION_CHUNK,
        SpanNames.INGESTION_EMBED,
        SpanNames.INGESTION_INDEX_DENSE,
        SpanNames.INGESTION_INDEX_LEXICAL,
    ):
        assert expected in names


def test_hybrid_retrieval_sibling_spans(memory_otel):
    with operation_span(component="orchestration", operation="query"):
        with operation_span(component="retrieval", operation="validate"):
            pass
        with operation_span(component="retrieval", operation="dense"):
            pass
        with operation_span(component="retrieval", operation="bm25"):
            pass
        with operation_span(component="retrieval", operation="rrf_fusion"):
            pass
    spans = memory_otel.span_exporter.get_finished_spans()
    names = _spans_by_name(spans)
    assert SpanNames.RETRIEVAL_DENSE in names
    assert SpanNames.RETRIEVAL_BM25 in names
    assert SpanNames.RETRIEVAL_RRF_FUSION in names
    fuse = names[SpanNames.RETRIEVAL_RRF_FUSION][0]
    dense = names[SpanNames.RETRIEVAL_DENSE][0]
    # siblings: fusion and dense share the same parent
    assert fuse.parent.span_id == dense.parent.span_id


def test_hierarchical_retrieval_parent_child_flow(memory_otel):
    with operation_span(component="orchestration", operation="query"):
        with operation_span(component="retrieval", operation="parent_search"):
            pass
        with operation_span(component="retrieval", operation="child_search"):
            pass
        with operation_span(component="retrieval", operation="context_expand"):
            pass
    spans = memory_otel.span_exporter.get_finished_spans()
    names = _spans_by_name(spans)
    assert SpanNames.RETRIEVAL_PARENT_SEARCH in names
    assert SpanNames.RETRIEVAL_CHILD_SEARCH in names
    assert SpanNames.RETRIEVAL_CONTEXT_EXPAND in names
    assert names[SpanNames.RETRIEVAL_PARENT_SEARCH][0].context.trace_id == names[
        SpanNames.RETRIEVAL_CHILD_SEARCH
    ][0].context.trace_id


def test_generation_fallback_attempt_trace(memory_otel):
    with operation_span(component="generation", operation="provider_select"):
        pass
    try:
        with operation_span(component="generation", operation="request", attributes={"provider": "groq", "model": "m1"}):
            raise TimeoutError("provider timed out")
    except TimeoutError:
        pass
    with operation_span(component="generation", operation="fallback"):
        pass
    with operation_span(component="generation", operation="request", attributes={"provider": "ollama", "model": "m2"}):
        pass
    spans = memory_otel.span_exporter.get_finished_spans()
    names = _spans_by_name(spans)
    requests = names[SpanNames.GENERATION_REQUEST]
    assert len(requests) == 2
    first, second = requests[0], requests[1]
    assert first.attributes["provider"] == "groq"
    assert second.attributes["provider"] == "ollama"
    # Groq failure is visible on the span
    assert first.status.status_code.name == "ERROR"
    assert SpanNames.GENERATION_FALLBACK in names


def test_guardrails_trace(memory_otel):
    with operation_span(component="orchestration", operation="answer"):
        with operation_span(component="guardrails", operation="prompt_guard"):
            pass
        with operation_span(component="guardrails", operation="privacy_detect"):
            pass
        with operation_span(component="guardrails", operation="safeguard"):
            pass
        with operation_span(component="guardrails", operation="citation_validate"):
            pass
        with operation_span(component="guardrails", operation="enforcement"):
            pass
    spans = memory_otel.span_exporter.get_finished_spans()
    names = _spans_by_name(spans)
    for expected in (
        SpanNames.GUARDRAILS_PROMPT_GUARD,
        SpanNames.GUARDRAILS_PRIVACY_DETECT,
        SpanNames.GUARDRAILS_SAFEGUARD,
        SpanNames.GUARDRAILS_CITATION_VALIDATE,
        SpanNames.GUARDRAILS_ENFORCEMENT,
    ):
        assert expected in names


def test_evaluation_trace(memory_otel):
    with operation_span(component="evaluation", operation="dataset_load"):
        pass
    with operation_span(component="evaluation", operation="ragas"):
        with operation_span(component="evaluation", operation="judge_call"):
            pass
        with operation_span(component="evaluation", operation="deterministic"):
            pass
    with operation_span(component="evaluation", operation="report"):
        pass
    spans = memory_otel.span_exporter.get_finished_spans()
    names = _spans_by_name(spans)
    for expected in (
        SpanNames.EVALUATION_DATASET_LOAD,
        SpanNames.EVALUATION_RAGAS,
        SpanNames.EVALUATION_JUDGE_CALL,
        SpanNames.EVALUATION_DETERMINISTIC,
        SpanNames.EVALUATION_REPORT,
    ):
        assert expected in names
    # judge call is nested under ragas
    assert names[SpanNames.EVALUATION_JUDGE_CALL][0].parent.span_id == names[
        SpanNames.EVALUATION_RAGAS
    ][0].context.span_id


def test_reranking_trace(memory_otel):
    from observability.tracing import SpanNames as Names

    with operation_span(Names.RERANK_PREPARE, operation="prepare"):
        pass
    with operation_span(Names.RERANK_INFERENCE, component="rerank", operation="inference"):
        pass
    with operation_span(Names.RERANK_REORDER, component="rerank", operation="reorder"):
        pass
    spans = memory_otel.span_exporter.get_finished_spans()
    names = _spans_by_name(spans)
    assert SpanNames.RERANK_PREPARE in names
    assert SpanNames.RERANK_INFERENCE in names
    assert SpanNames.RERANK_REORDER in names


def test_metrics_emit_alongside_spans(memory_otel):
    """A full query shows up in both traces and metrics."""
    from observability.metrics import MetricNames, default_registry

    registry = default_registry()
    with operation_span(component="orchestration", operation="query"):
        with operation_span(component="retrieval", operation="dense"):
            pass
        with operation_span(component="generation", operation="request"):
            pass
    registry.counter(MetricNames.RETRIEVAL_QUERIES, unit="{query}").inc(
        {"component": "retrieval", "retrieval_strategy": "hybrid"}
    )

    from .conftest import metric_data_points

    data = memory_otel.metric_reader.get_metrics_data()
    points = metric_data_points(data, MetricNames.RETRIEVAL_QUERIES)
    assert points
