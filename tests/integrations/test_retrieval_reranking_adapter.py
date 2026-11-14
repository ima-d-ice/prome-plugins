"""Tests for the Retrieval → Reranking adapter."""

from __future__ import annotations

from retrieval.contracts.models import RetrievedItem
from reranking.contracts.models import RerankCandidate
from integrations.retrieval_reranking_adapter import (
    retrieved_item_to_rerank_candidate,
    reranked_items_to_context_items,
)


def test_retrieved_item_to_rerank_candidate_basic():
    """Basic conversion preserves chunk_id, text, rank, score, metadata, provenance."""
    item = RetrievedItem(
        chunk_id="ch-1",
        doc_id="doc-1",
        text="some passage",
        score=0.93,
        score_direction="higher_is_better",
        rank=1,
        strategy="dense",
        metadata={"source": "doc1.pdf"},
        provenance={"doc_id": "doc-1", "chunk_id": "ch-1", "original_uri": "/path/to/doc1.pdf"},
    )
    candidate = retrieved_item_to_rerank_candidate(item)
    assert candidate.candidate_id == "ch-1"
    assert candidate.text == "some passage"
    assert candidate.original_rank == 1
    # Note: RerankCandidate model does not have a raw_score field;
    # the adapter computes it but does not store it in the model.
    assert candidate.metadata == {"source": "doc1.pdf"}
    assert candidate.source_provenance == {"doc_id": "doc-1", "chunk_id": "ch-1", "original_uri": "/path/to/doc1.pdf"}
    assert candidate.retrieval_result == {
        "chunk_id": "ch-1",
        "doc_id": "doc-1",
        "score": 0.93,
        "rank": 1,
        "strategy": "dense",
    }


def test_retrieved_item_to_rerank_candidate_preserves_provenance_explicit():
    """When preserve_provenance=False, source_provenance is None but retrieval_result still set if provenance exists."""
    item = RetrievedItem(
        chunk_id="ch-2",
        doc_id="doc-2",
        text="another passage",
        score=0.87,
        score_direction="higher_is_better",
        rank=2,
        strategy="bm25",
        metadata={},
        provenance={"doc_id": "doc-2", "chunk_id": "ch-2"},
    )
    candidate = retrieved_item_to_rerank_candidate(item, preserve_provenance=False)
    # source_provenance is None when preserve_provenance=False
    assert candidate.source_provenance is None
    # retrieval_result is still set because it's derived from item.provenance, not preserve_provenance
    assert candidate.retrieval_result is not None


def test_retrieved_item_to_rerank_candidate_minimal():
    """Item with minimal fields still produces a valid candidate."""
    item = RetrievedItem(
        chunk_id="ch-3",
        doc_id="doc-3",
        text="minimal text",
        score=0.5,
        score_direction="higher_is_better",
        rank=1,
        strategy="",
        metadata={},
        provenance=None,
    )
    candidate = retrieved_item_to_rerank_candidate(item)
    assert candidate.candidate_id == "ch-3"
    assert candidate.text == "minimal text"
    assert candidate.source_provenance is None
    assert candidate.retrieval_result is None


def test_reranked_items_to_context_items_basic():
    """Conversion from reranked items to context items preserves chunk_id, doc_id, text, metadata, provenance."""
    from reranking.contracts.models import RerankCandidate, RerankedItem

    candidate1 = RerankCandidate(
        candidate_id="ch-1",
        text="first passage",
        original_rank=1,
        metadata={"source": "doc1.pdf"},
        source_provenance={"doc_id": "doc-1", "chunk_id": "ch-1"},
        retrieval_result={"chunk_id": "ch-1", "doc_id": "doc-1", "score": 0.93, "rank": 1, "strategy": "dense"},
    )
    candidate2 = RerankCandidate(
        candidate_id="ch-2",
        text="second passage",
        original_rank=2,
        metadata={"source": "doc2.pdf"},
        source_provenance={"doc_id": "doc-2", "chunk_id": "ch-2"},
        retrieval_result={"chunk_id": "ch-2", "doc_id": "doc-2", "score": 0.87, "rank": 2, "strategy": "bm25"},
    )

    ri1 = RerankedItem(candidate=candidate1, raw_score=0.95, original_rank=1, new_rank=1)
    ri2 = RerankedItem(candidate=candidate2, raw_score=0.88, original_rank=2, new_rank=2)

    context_items = reranked_items_to_context_items((ri1, ri2))
    assert len(context_items) == 2

    # First item: chunk_id from candidate_id, doc_id from source_provenance
    assert context_items[0]["chunk_id"] == "ch-1"
    assert context_items[0]["doc_id"] == "doc-1"
    assert context_items[0]["text"] == "first passage"
    assert context_items[0]["metadata"] == {"source": "doc1.pdf"}
    assert "provenance" in context_items[0]

    # Second item
    assert context_items[1]["chunk_id"] == "ch-2"
    assert context_items[1]["doc_id"] == "doc-2"
    assert context_items[1]["text"] == "second passage"
    assert context_items[1]["metadata"] == {"source": "doc2.pdf"}
    assert "provenance" in context_items[1]


def test_reranked_items_to_context_items_doc_id_from_provenance():
    """doc_id extracted from source_provenance when available."""
    from reranking.contracts.models import RerankCandidate, RerankedItem

    candidate = RerankCandidate(
        candidate_id="ch-1",
        text="doc identity test",
        original_rank=1,
        metadata={},
        source_provenance={"doc_id": "doc-42", "chunk_id": "ch-1"},
        retrieval_result=None,
    )
    ri = RerankedItem(candidate=candidate, raw_score=0.9, original_rank=1, new_rank=1)

    context_items = reranked_items_to_context_items((ri,))
    assert context_items[0]["doc_id"] == "doc-42"


def test_reranked_items_to_context_items_provenance_priority():
    """Provenance preference: retrieval_result > source_provenance."""
    from reranking.contracts.models import RerankCandidate, RerankedItem

    candidate = RerankCandidate(
        candidate_id="ch-1",
        text=" provenance test",
        original_rank=1,
        metadata={},
        source_provenance={"doc_id": "doc-1", "chunk_id": "ch-1"},
        retrieval_result={"chunk_id": "ch-1", "doc_id": "doc-1", "score": 0.93, "rank": 1, "strategy": "dense"},
    )
    ri = RerankedItem(candidate=candidate, raw_score=0.95, original_rank=1, new_rank=1)

    context_items = reranked_items_to_context_items((ri,))
    # Should use retrieval_result since it's present
    assert "provenance" in context_items[0]
    assert context_items[0]["provenance"] == {
        "chunk_id": "ch-1",
        "doc_id": "doc-1",
        "score": 0.93,
        "rank": 1,
        "strategy": "dense",
    }


def test_reranked_items_to_context_items_no_doc_id_no_heuristic():
    """When neither source_provenance nor heuristic is available, doc_id is absent."""
    from reranking.contracts.models import RerankCandidate, RerankedItem

    candidate = RerankCandidate(
        candidate_id="ch-99",
        text="no doc id",
        original_rank=1,
        metadata={},
        source_provenance={},
        retrieval_result=None,
    )
    ri = RerankedItem(candidate=candidate, raw_score=0.5, original_rank=1, new_rank=1)

    context_items = reranked_items_to_context_items((ri,))
    assert context_items[0]["chunk_id"] == "ch-99"
    # doc_id may be absent
    assert "doc_id" not in context_items[0]