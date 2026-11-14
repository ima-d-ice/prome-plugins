"""Tests for the Reranking → Generation adapter."""

from __future__ import annotations

from reranking.contracts.models import RerankCandidate, RerankedItem, RerankRequest, RerankResponse
from integrations.reranking_generation_adapter import reranked_to_generation_context


def test_reranked_to_generation_context_basic():
    """Basic conversion from reranked items to generation context items."""
    candidate1 = RerankCandidate(
        candidate_id="ch-1",
        text="first evidence",
        original_rank=1,
        metadata={"source": "doc1.pdf"},
        source_provenance={"doc_id": "doc-1", "chunk_id": "ch-1"},
        retrieval_result={"chunk_id": "ch-1", "doc_id": "doc-1", "score": 0.93, "rank": 1, "strategy": "dense"},
    )
    candidate2 = RerankCandidate(
        candidate_id="ch-2",
        text="second evidence",
        original_rank=2,
        metadata={"source": "doc2.pdf"},
        source_provenance={"doc_id": "doc-2", "chunk_id": "ch-2"},
        retrieval_result={"chunk_id": "ch-2", "doc_id": "doc-2", "score": 0.87, "rank": 2, "strategy": "bm25"},
    )

    ri1 = RerankedItem(candidate=candidate1, raw_score=0.95, original_rank=1, new_rank=1)
    ri2 = RerankedItem(candidate=candidate2, raw_score=0.88, original_rank=2, new_rank=2)

    context_items = reranked_to_generation_context((ri1, ri2))
    assert len(context_items) == 2
    # Order preserved: index 0 = most relevant
    assert context_items[0]["chunk_id"] == "ch-1"
    assert context_items[1]["chunk_id"] == "ch-2"


def test_reranked_to_generation_context_preserves_doc_id():
    """doc_id extracted from source_provenance when available."""
    candidate = RerankCandidate(
        candidate_id="ch-1",
        text="doc identity test",
        original_rank=1,
        metadata={},
        source_provenance={"doc_id": "doc-42", "chunk_id": "ch-1"},
        retrieval_result=None,
    )
    ri = RerankedItem(candidate=candidate, raw_score=0.9, original_rank=1, new_rank=1)

    context_items = reranked_to_generation_context((ri,))
    assert context_items[0]["doc_id"] == "doc-42"


def test_reranked_to_generation_context_falls_back_to_candidate_id_heuristic():
    """When source_provenance has no doc_id, heuristic from candidate_id attempts extraction."""
    candidate = RerankCandidate(
        candidate_id="doc77.chunk-003",  # contains "." to trigger heuristic
        text="heuristic doc id",
        original_rank=1,
        metadata={},
        source_provenance={},
        retrieval_result=None,
    )
    ri = RerankedItem(candidate=candidate, raw_score=0.7, original_rank=1, new_rank=1)

    context_items = reranked_to_generation_context((ri,))
    # Heuristic: split on "-" and join all but last part
    # "doc77.chunk-003" -> parts=["doc77", "chunk", "003"] -> "doc77.chunk"
    assert context_items[0]["doc_id"] == "doc77.chunk"


def test_reranked_to_generation_context_provenance_included():
    """Provenance (retrieval_result) included in context item when available."""
    candidate = RerankCandidate(
        candidate_id="ch-1",
        text="provenance test",
        original_rank=1,
        metadata={},
        source_provenance={"doc_id": "doc-1", "chunk_id": "ch-1"},
        retrieval_result={"chunk_id": "ch-1", "doc_id": "doc-1", "score": 0.93, "rank": 1, "strategy": "dense"},
    )
    ri = RerankedItem(candidate=candidate, raw_score=0.95, original_rank=1, new_rank=1)

    context_items = reranked_to_generation_context((ri,))
    assert "provenance" in context_items[0]
    assert context_items[0]["provenance"] == {
        "chunk_id": "ch-1",
        "doc_id": "doc-1",
        "score": 0.93,
        "rank": 1,
        "strategy": "dense",
    }


def test_reranked_to_generation_context_empty_tuple():
    """Empty input returns empty tuple."""
    context_items = reranked_to_generation_context(())
    assert context_items == ()