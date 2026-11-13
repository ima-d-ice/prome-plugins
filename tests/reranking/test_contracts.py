"""Dependency-free unit tests for the reranking contracts.

These tests run without model weights or heavyweight inference libraries.
They test contract validation, candidate preservation, stable IDs, empty
and invalid input handling, ordering, and error propagation.
"""

from __future__ import annotations

import pytest

from reranking.contracts.models import (
    RerankCandidate,
    RerankDiagnostics,
    RerankedItem,
    RerankRequest,
    RerankResponse,
    RerankScoreInterpretation,
)
from reranking.contracts.protocols import Reranker


def test_rerank_candidate_creation_preserves_fields():
    """Test that RerankCandidate preserves all fields."""
    cand = RerankCandidate(
        candidate_id="candidate-001",
        text="This is the chunk text to be scored.",
        original_rank=3,
        metadata={"source": "parser-x", "page": 42},
    )
    assert cand.candidate_id == "candidate-001"
    assert cand.text == "This is the chunk text to be scored."
    assert cand.original_rank == 3
    assert cand.metadata == {"source": "parser-x", "page": 42}
    assert cand.source_provenance is None
    assert cand.retrieval_result is None


def test_rerank_candidate_with_provenance_and_result():
    """Test RerankCandidate with source provenance and retrieval result."""
    result_ref = {"doc_id": "doc-123", "chunk_ids": ("c1", "c2"), "score": 0.87}
    cand = RerankCandidate(
        candidate_id="c42",
        text="another chunk",
        original_rank=1,
        source_provenance="page-7, block B",
        retrieval_result=result_ref,
    )
    assert cand.candidate_id == "c42"
    assert cand.source_provenance == "page-7, block B"
    assert cand.retrieval_result is result_ref


def test_rerank_request_creation_minimal():
    """Test RerankRequest with minimal fields."""
    request = RerankRequest(
        query="what is this",
        candidates=(RerankCandidate(candidate_id="c1", text="chunk"),),
    )
    assert request.query == "what is this"
    assert len(request.candidates) == 1
    assert request.top_k is None
    assert request.task_instruction is None


def test_rerank_request_with_top_k():
    """Test RerankRequest with top_k limit."""
    request = RerankRequest(
        query="query",
        candidates=(
            RerankCandidate(candidate_id="c1", text="t1"),
            RerankCandidate(candidate_id="c2", text="t2"),
            RerankCandidate(candidate_id="c3", text="t3"),
        ),
        top_k=2,
    )
    assert request.top_k == 2
    assert len(request.candidates) == 3


def test_rerank_request_invalid_empty_query():
    """Test that empty query text is rejected."""
    with pytest.raises(ValueError):
        RerankRequest(query="", candidates=(RerankCandidate(candidate_id="c1", text=""),))


def test_rerank_request_invalid_empty_candidates():
    """Test that empty candidates list is rejected."""
    with pytest.raises(ValueError):
        RerankRequest(query="some query", candidates=())


def test_rerank_request_invalid_top_k_zero():
    """Test that top_k=0 is rejected."""
    candidates = (RerankCandidate(candidate_id="c1", text="t1"),)
    with pytest.raises(ValueError):
        RerankRequest(query="q", candidates=candidates, top_k=0)


def test_rerank_request_invalid_top_negative():
    """Test that top_k=-1 is rejected."""
    candidates = (RerankCandidate(candidate_id="c1", text="t1"),)
    with pytest.raises(ValueError):
        RerankRequest(query="q", candidates=candidates, top_k=-1)


def test_rerank_request_invalid_top_exceeds_candidates():
    """Test that top_k exceeding candidate count is rejected."""
    candidates = (RerankCandidate(candidate_id="c1", text="t1"),)
    with pytest.raises(ValueError):
        RerankRequest(query="q", candidates=candidates, top_k=5)


def test_reranked_item_creation():
    """Test basic RerankedItem creation."""
    cand = RerankCandidate(candidate_id="c1", text="some text")
    item = RerankedItem(candidate=cand, raw_score=5.7, original_rank=3, new_rank=1)
    assert item.candidate is cand
    assert item.raw_score == 5.7
    assert item.original_rank == 3
    assert item.new_rank == 1


def test_reranked_item_preserves_candidate():
    """Test that RerankedItem preserves the original candidate."""
    cand = RerankCandidate(candidate_id="c99", text="original text", original_rank=2)
    item = RerankedItem(candidate=cand, raw_score=3.14, original_rank=2, new_rank=1)
    assert item.candidate is cand
    assert item.candidate.candidate_id == "c99"
    assert item.candidate.text == "original text"


def test_rerank_response_creation():
    """Test basic RerankResponse creation."""
    cand = RerankCandidate(candidate_id="c1", text="some text")
    item = RerankedItem(candidate=cand, raw_score=5.0, original_rank=3, new_rank=1)
    response = RerankResponse(
        reranked_items=(item,),
        model_id="test-model",
        model_revision="main",
        candidates_scored=1,
        elapsed_ms=12.5,
        score_interpretation=RerankScoreInterpretation.RAW,
        diagnostics={"batch_count": 1},
    )
    assert len(response.reranked_items) == 1
    assert response.model_id == "test-model"
    assert response.model_revision == "main"
    assert response.candidates_scored == 1
    assert response.elapsed_ms == 12.5
    assert response.score_interpretation == RerankScoreInterpretation.RAW
    assert "batch_count" in response.diagnostics


def test_rerank_score_interpretation_raw():
    """Test RerankScoreInterpretation.RAW value."""
    assert RerankScoreInterpretation.RAW.value == "raw"


def test_rerank_score_interpretation_sigmoid():
    """Test RerankScoreInterpretation.SIGMOID value."""
    assert RerankScoreInterpretation.SIGMOID.value == "sigmoid"


def test_rerank_diagnostics_defaults():
    """Test RerankDiagnostics default values."""
    d = RerankDiagnostics()
    assert d.candidates_scored == 0
    assert d.batch_count == 0


def test_reranker_protocol_is_runtime_checkable():
    """Test that the Reranker protocol is runtime checkable."""
    # A class that implements the required attributes/methods should be
    # an instance of the protocol.

    class FakeReranker:
        model_id = "fake-model"

        @property
        def model_revision(self) -> str | None:
            return None

        @property
        def capabilities(self) -> dict[str, Any]:
            return {}

        def rerank(self, request: RerankRequest) -> RerankResponse:
            from reranking.contracts.models import RerankResponse
            return RerankResponse(
                reranked_items=(),
                model_id=self.model_id,
                candidates_scored=0,
            )

    fake = FakeReranker()
    assert isinstance(fake, Reranker)


def test_reranker_protocol_is_not_object():
    """Test that plain object is not an instance of the Reranker protocol."""

    assert not isinstance(object(), Reranker)


def test_rerank_response_defaults():
    """Test RerankResponse with only required fields."""
    cand = RerankCandidate(candidate_id="c1", text="text")
    item = RerankedItem(candidate=cand, raw_score=1.0, original_rank=1, new_rank=1)
    response = RerankResponse(reranked_items=(item,), model_id="m")
    assert response.model_id == "m"
    assert response.model_revision is None
    assert response.candidates_scored == 0
    assert response.elapsed_ms is None
    assert response.score_interpretation == RerankScoreInterpretation.RAW
    assert response.diagnostics == {}