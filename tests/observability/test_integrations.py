"""Integration helpers wrap duck-typed components without touching packages."""
from __future__ import annotations

import pytest

from observability.instrumentation import (
    instrument_chunker,
    instrument_dense_retriever,
    instrument_embedding,
    instrument_evaluation,
    instrument_generation,
    instrument_guardrails,
    instrument_loader,
    instrument_orchestration,
    instrument_parser,
    instrument_reranker,
    instrument_vector_store,
)
from observability.metrics import MetricNames

from .conftest import metric_data_points


class FakeParser:
    name = "fake_parser"

    def parse(self, source):
        return {"source": source, "blocks": ["a", "b"]}


class FakeFailingParser:
    def parse(self, source):
        raise RuntimeError("parse exploded api_key=sk-abc1234567890def")


def test_instrument_parser_success(memory_otel):
    wrapped = instrument_parser(FakeParser())
    assert wrapped.name == "fake_parser"  # attribute passthrough
    assert wrapped.parse("doc.pdf")["blocks"] == ["a", "b"]
    spans = memory_otel.span_exporter.get_finished_spans()
    assert any(span.name == "ingestion.parse" for span in spans)
    data = memory_otel.metric_reader.get_metrics_data()
    points = metric_data_points(data, MetricNames.INGESTION_DOCUMENTS_PARSED)
    assert points


def test_instrument_parser_failure_counts_and_propagates(memory_otel):
    wrapped = instrument_parser(FakeFailingParser())
    with pytest.raises(RuntimeError):
        wrapped.parse("doc.pdf")
    data = memory_otel.metric_reader.get_metrics_data()
    assert metric_data_points(data, MetricNames.INGESTION_PARSE_FAILURES)


def test_instrument_chunker_counts_output(memory_otel):
    class FakeChunker:
        name = "fake"

        def chunk(self, document):
            return ["c1", "c2", "c3"]

    wrapped = instrument_chunker(FakeChunker())
    assert wrapped.chunk("doc") == ["c1", "c2", "c3"]
    data = memory_otel.metric_reader.get_metrics_data()
    points = metric_data_points(data, MetricNames.INGESTION_CHUNKS_CREATED)
    assert points[0][2].value == 3


def test_instrument_embedding_counts_requests(memory_otel):
    class FakeEmbedding:
        model_id = "qwen3-embedding-0.6b"

        def embed_documents(self, texts):
            return [[0.1, 0.2] for _ in texts]

        def embed_query(self, text):
            return [0.1]

    wrapped = instrument_embedding(FakeEmbedding())
    assert wrapped.embed_documents(["a", "b"]) == [[0.1, 0.2], [0.1, 0.2]]
    assert wrapped.embed_query("q") == [0.1]
    data = memory_otel.metric_reader.get_metrics_data()
    points = metric_data_points(data, MetricNames.INGESTION_EMBEDDING_REQUESTS)
    # both calls share labels, so the cumulative point is 2
    assert sum(point.value for _, _, point in points) == 2


def test_instrument_vector_store_upsert_and_search(memory_otel):
    class FakeStore:
        name = "fake_store"

        def upsert(self, records):
            return len(records)

        def search(self, request):
            return ["hit1", "hit2"]

    wrapped = instrument_vector_store(FakeStore())
    wrapped.upsert(["r1", "r2", "r3"])
    wrapped.search("query")
    data = memory_otel.metric_reader.get_metrics_data()
    assert metric_data_points(data, MetricNames.INGESTION_DENSE_INDEX_UPSERTS)
    assert metric_data_points(data, MetricNames.RETRIEVAL_DENSE_CANDIDATES)


def test_instrument_dense_retriever_counts_and_empty(memory_otel):
    class FakeRetriever:
        name = "dense"

        def __init__(self, results):
            self._results = results

        def retrieve(self, query):
            return type("Resp", (), {"results": self._results})()

    wrapped = instrument_dense_retriever(FakeRetriever(["a", "b"]))
    response = wrapped.retrieve("anything")
    assert response.results == ["a", "b"]
    data = memory_otel.metric_reader.get_metrics_data()
    assert metric_data_points(data, MetricNames.RETRIEVAL_QUERIES)
    assert metric_data_points(data, MetricNames.RETRIEVAL_DENSE_CANDIDATES)

    empty = instrument_dense_retriever(FakeRetriever([]))
    empty.retrieve("anything")
    data = memory_otel.metric_reader.get_metrics_data()
    assert metric_data_points(data, MetricNames.RETRIEVAL_EMPTY_RESULTS)


def test_instrument_reranker_records(memory_otel):
    class FakeResponse:
        def __init__(self):
            self.candidates = None
            self.candidates_scored = 4
            self.reranked_items = [1, 2, 3, 4]

    class FakeRerankRequest:
        def __init__(self):
            self.candidates = ["c1", "c2", "c3", "c4"]
            self.query = "q"

    class FakeReranker:
        model_id = "qwen3-reranker-0.6b"

        def rerank(self, request):
            return FakeResponse()

    wrapped = instrument_reranker(FakeReranker())
    response = wrapped.rerank(FakeRerankRequest())
    assert response.candidates_scored == 4
    data = memory_otel.metric_reader.get_metrics_data()
    points = metric_data_points(data, MetricNames.RERANK_CANDIDATES)
    assert points[0][2].value == 4
    assert metric_data_points(data, MetricNames.RERANK_DURATION)


def test_instrument_generation_records_usage_and_fallback(memory_otel):
    class FakeResult:
        def __init__(self):
            self.provider = "groq"
            self.model = "llama"
            self.usage = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
            self.fallback_used = True

    class FakeClient:
        provider = "groq"
        model = "llama"

        def generate(self, prompt):
            return FakeResult()

    wrapped = instrument_generation(FakeClient())
    result = wrapped.generate("prompt text")
    assert result.usage["total_tokens"] == 12
    data = memory_otel.metric_reader.get_metrics_data()
    assert metric_data_points(data, MetricNames.GENERATION_REQUESTS)
    assert metric_data_points(data, MetricNames.GENERATION_TOTAL_TOKENS)
    assert metric_data_points(data, MetricNames.GENERATION_FALLBACKS)
    spans = memory_otel.span_exporter.get_finished_spans()
    gen = [span for span in spans if span.name == "generation.request"][0]
    assert gen.attributes["provider"] == "groq"
    assert gen.attributes["usage.total_tokens"] == 12


def test_instrument_generation_failure(memory_otel):
    class FakeClient:
        provider = "groq"
        model = "llama"

        def generate(self, prompt):
            raise TimeoutError("timed out")

    wrapped = instrument_generation(FakeClient())
    with pytest.raises(TimeoutError):
        wrapped.generate("prompt")
    data = memory_otel.metric_reader.get_metrics_data()
    points = metric_data_points(data, MetricNames.GENERATION_ERRORS)
    assert points


def test_instrument_guardrails_decision_metrics(memory_otel):
    class FakeGuard:
        def evaluate(self, text):
            return {"decision": "block", "pii_detected": True}

    wrapped = instrument_guardrails(FakeGuard())
    result = wrapped.evaluate("something bad")
    assert result["decision"] == "block"
    data = memory_otel.metric_reader.get_metrics_data()
    assert metric_data_points(data, MetricNames.GUARDRAILS_BLOCKS)
    assert metric_data_points(data, MetricNames.GUARDRAILS_PRIVACY_DETECTIONS)
    assert metric_data_points(data, MetricNames.GUARDRAILS_SAFEGUARD_INVOCATIONS)


def test_instrument_loader_counts(memory_otel):
    class FakeLoader:
        def load(self, path):
            return {"path": path}

    class FailingLoader:
        def load(self, path):
            raise FileNotFoundError("missing")

    ok = instrument_loader(FakeLoader())
    ok.load("doc.pdf")
    failing = instrument_loader(FailingLoader())
    with pytest.raises(FileNotFoundError):
        failing.load("nope.pdf")
    data = memory_otel.metric_reader.get_metrics_data()
    assert metric_data_points(data, MetricNames.INGESTION_DOCUMENTS_LOADED)
    assert metric_data_points(data, MetricNames.INGESTION_DOCUMENTS_REJECTED)


def test_instrument_orchestration_tracks_active_requests(memory_otel):
    class FakePipeline:
        def run(self, query):
            return "answer"

    wrapped = instrument_orchestration(FakePipeline())
    assert wrapped.run("q") == "answer"
    data = memory_otel.metric_reader.get_metrics_data()
    points = metric_data_points(data, MetricNames.RUNTIME_ACTIVE_REQUESTS)
    assert points
    assert sum(point.value for _, _, point in points) == 0  # inc + dec


def test_instrument_evaluation_records(memory_otel):
    class FakeEvaluation:
        def evaluate(self, dataset):
            return type("Result", (), {"results": [1, 2, 3]})()

        def judge(self, prompt):
            return {
                "usage": {"total_tokens": 99},
                "provider": "groq",
                "model": "judge-1",
            }

    wrapped = instrument_evaluation(FakeEvaluation())
    wrapped.evaluate("dataset.jsonl")
    wrapped.judge("judge this")
    data = memory_otel.metric_reader.get_metrics_data()
    assert metric_data_points(data, MetricNames.EVALUATION_EXPERIMENT_RUNS)
    assert metric_data_points(data, MetricNames.EVALUATION_SAMPLES_EVALUATED)
    assert metric_data_points(data, MetricNames.EVALUATION_JUDGE_TOKENS)


def test_wrapped_object_is_transparent_and_reversible(memory_otel):
    parser = FakeParser()
    wrapped = instrument_parser(parser)
    assert wrapped.parse is not parser.parse
    assert wrapped.uninstrument() is parser
    assert wrapped.__wrapped_methods__ == ("parse",)


def test_missing_methods_are_skipped(memory_otel):
    class Empty:
        pass

    wrapped = instrument_parser(Empty())  # has no .parse
    assert wrapped.__wrapped_methods__ == ()


def test_integrations_noop_without_observability():
    # enabled=False: wrappers still work, metrics/spans are no-ops
    wrapped = instrument_parser(FakeParser(), component="parser")
    assert wrapped.parse("doc.pdf")["blocks"] == ["a", "b"]


def test_instrument_generation_reads_top_level_token_attributes(memory_otel):
    """GenerationResponse-style objects expose tokens as plain attributes."""

    class GenerationResponse:
        def __init__(self):
            self.text = "hello"
            self.actual_provider = "ollama"
            self.actual_model = "qwen3"
            self.input_tokens = 15
            self.output_tokens = 5
            self.total_tokens = 20

    class FakeClient:
        provider = "ollama"
        model = "qwen3"

        def generate(self, prompt):
            return GenerationResponse()

    wrapped = instrument_generation(FakeClient())
    result = wrapped.generate("prompt")
    assert result.total_tokens == 20
    data = memory_otel.metric_reader.get_metrics_data()
    assert metric_data_points(data, MetricNames.GENERATION_TOTAL_TOKENS)
    spans = memory_otel.span_exporter.get_finished_spans()
    gen = [span for span in spans if span.name == "generation.request"][0]
    assert gen.attributes["usage.total_tokens"] == 20


def test_instrument_generation_reads_groq_usage_dicts(memory_otel):
    class FakeClient:
        provider = "groq"
        model = "llama"

        def generate(self, prompt):
            return {
                "text": "hi",
                "usage": {"prompt_tokens": 8, "completion_tokens": 2},
            }

    wrapped = instrument_generation(FakeClient())
    wrapped.generate("prompt")
    data = memory_otel.metric_reader.get_metrics_data()
    points = metric_data_points(data, MetricNames.GENERATION_TOTAL_TOKENS)
    # total synthesized as input + output = 10
    assert points[0][2].value == 10


def test_instrument_generation_ollama_flat_response(memory_otel):
    class FakeClient:
        provider = "ollama"
        model = "qwen3"

        def generate(self, prompt):
            return {"response": "hi", "prompt_eval_count": 33, "eval_count": 7}

    wrapped = instrument_generation(FakeClient())
    wrapped.generate("prompt")
    data = memory_otel.metric_reader.get_metrics_data()
    points = metric_data_points(data, MetricNames.GENERATION_INPUT_TOKENS)
    assert points[0][2].value == 33
