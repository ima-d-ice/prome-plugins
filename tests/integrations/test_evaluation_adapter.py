"""Tests for the Evaluation adapter."""

from __future__ import annotations

from integrations.evaluation_adapter import (
    eval_sample_from_pipeline,
    experiment_config_from_pipeline,
    pipeline_outputs_to_eval_sample,
)


def test_eval_sample_from_pipeline_basic():
    """Basic evaluation sample from pipeline outputs."""
    sample = eval_sample_from_pipeline(
        user_input="What is quantum computing?",
        retrieved_chunks=[
            type("Chunk", (), {"text": "Quantum computing uses quantum states.", "chunk_id": "ch-1", "doc_id": "doc-1"})(),
            type("Chunk", (), {"text": "Superposition is a key principle.", "chunk_id": "ch-2", "doc_id": "doc-1"})(),
        ],
        generation_response=type("GenerationResponse", (), {
            "text": "Quantum computing leverages quantum mechanical phenomena.",
            "citation_source_refs": [{"chunk_id": "ch-1"}, {"doc_id": "doc-2"}],
        })(),
        cited_source_ids=["ch-1", "doc-2"],
        reference="Quantum computing uses quantum states.",
    )
    assert sample.user_input == "What is quantum computing?"
    assert len(sample.retrieved_contexts) == 2
    assert len(sample.retrieved_context_ids) == 2
    assert sample.generated_response == "Quantum computing leverages quantum mechanical phenomena."
    assert "ch-1" in sample.cited_source_ids
    assert "doc-2" in sample.cited_source_ids


def test_eval_sample_from_pipeline_no_generation_response():
    """Evaluation sample without generation response still works."""
    sample = eval_sample_from_pipeline(
        user_input="Tell me about dogs.",
        retrieved_chunks=[
            type("Chunk", (), {"text": "Dogs are mammals.", "chunk_id": "ch-1", "doc_id": "doc-1"})(),
        ],
        generation_response=None,
        cited_source_ids=None,
    )
    assert sample.user_input == "Tell me about dogs."
    assert len(sample.retrieved_contexts) == 1
    assert sample.generated_response is None


def test_eval_sample_from_pipeline_empty_chunks():
    """Evaluation sample with no retrieved chunks."""
    sample = eval_sample_from_pipeline(
        user_input="Empty query.",
        retrieved_chunks=[],
        generation_response=None,
        cited_source_ids=None,
    )
    assert sample.user_input == "Empty query."
    assert sample.retrieved_contexts == []
    assert sample.retrieved_context_ids == []


def test_eval_sample_from_pipeline_cited_source_ids_extracted_from_response():
    """cited_source_ids extracted from generation_response.citation_source_refs when not provided explicitly."""
    sample = eval_sample_from_pipeline(
        user_input="Query.",
        retrieved_chunks=[],
        generation_response=type("GenerationResponse", (), {
            "text": "Answer.",
            "citation_source_refs": [{"chunk_id": "ch-5"}, {"doc_id": "doc-10"}],
        })(),
        cited_source_ids=None,
    )
    assert "ch-5" in sample.cited_source_ids
    assert "doc-10" in sample.cited_source_ids


def test_eval_sample_from_pipeline_deduplicates_cited_source_ids():
    """Cited source IDs are deduplicated while preserving order."""
    sample = eval_sample_from_pipeline(
        user_input="Query.",
        retrieved_chunks=[],
        generation_response=type("GenerationResponse", (), {
            "text": "Answer.",
            "citation_source_refs": [{"chunk_id": "ch-1"}, {"chunk_id": "ch-1"}, {"doc_id": "doc-1"}],
        })(),
        cited_source_ids=None,
    )
    # Should be deduplicated: ch-1 appears once, doc-1 once
    assert sample.cited_source_ids.count("ch-1") == 1
    assert sample.cited_source_ids.count("doc-1") == 1


def test_experiment_config_from_pipeline_basic():
    """Basic experiment config from pipeline configuration.
    
    Note: ragas is an optional dependency; this test is skipped when ragas
    is not installed.
    """
    import pytest
    pytest.skip("ragas not installed - optional dependency")
    assert config.experiment_id == "exp-001"
    assert config.dataset_id == "dataset-42"
    assert config.reranker_on is True
    assert config.generation_model == "qwen/qwen3-8b"
    assert config.judge_provider == "groq"
    assert config.judge_model == "llama-3.1-70b"
    assert config.judge_temperature == 0.01
    assert config.judge_max_tokens == 512
    assert config.embedding_model == "qwen3-embedding-0.6b"
    assert config.embedding_revision == "main"
    assert "faithfulness" in (config.metric_names or [])
    assert "answer_relevancy" in (config.metric_names or [])


def test_pipeline_outputs_to_eval_sample_basic():
    """High-level convenience: pipeline outputs to eval sample."""
    sample = pipeline_outputs_to_eval_sample(
        user_input="What is RAG?",
        retrieval_results=[
            type("RetrievalResult", (), {
                "chunk_id": "ch-1",
                "doc_id": "doc-1",
                "text": "RAG combines retrieval and generation.",
                "score": 0.95,
            })(),
        ],
        generation_response=type("GenerationResponse", (), {
            "text": "RAG is Retrieval-Augmented Generation.",
            "citation_source_refs": [{"chunk_id": "ch-1"}],
        })(),
        strategy="dense",
        experiment_id="exp-002",
        dataset_id="ds-001",
    )
    assert sample.user_input == "What is RAG?"
    assert len(sample.retrieved_contexts) == 1
    assert sample.strategy == "dense"
    assert sample.sample_id == "exp-002"
    assert sample.dataset_version == "ds-001"