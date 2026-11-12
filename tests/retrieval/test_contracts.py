"""Tests of the shared retrieval contracts.

These tests are dependency free: no vector store, no model, no database.
"""

from __future__ import annotations

import pytest

from retrieval.contracts import (
    ContextChunk,
    RetrievalCapabilities,
    RetrievalConfigError,
    RetrievalDiagnostics,
    RetrievalQuery,
    RetrievalResponse,
    RetrievedItem,
    Retriever,
)
from vector_stores.contracts.filters import FilterOp, eq, is_in


def test_the_query_carries_every_restriction():
    """A query carries the limits, the filter and the restrictions."""
    query = RetrievalQuery(
        text="how are the keys rotated",
        query_id="q-1",
        limit=5,
        dense_limit=50,
        lexical_limit=30,
        filters=eq("tenant", "acme"),
        doc_ids=("a.md", "b.md"),
        tenant_id="acme",
        params={"hnsw_ef": 128},
    )

    assert query.text == "how are the keys rotated"
    assert query.query_id == "q-1"
    assert query.limit == 5
    assert query.dense_limit == 50
    assert query.lexical_limit == 30
    assert query.filters is not None
    assert query.doc_ids == ("a.md", "b.md")
    assert query.tenant_id == "acme"
    assert query.params["hnsw_ef"] == 128


def test_the_candidate_limits_default_to_the_result_limit():
    """Without an explicit limit the channels return the requested count."""
    query = RetrievalQuery(text="hello", limit=7)

    assert query.effective_dense_limit == 7
    assert query.effective_lexical_limit == 7


def test_an_empty_limit_is_rejected():
    """A limit below one is a configuration error."""
    with pytest.raises(ValueError):
        RetrievalQuery(text="hello", limit=0)


def test_a_negative_candidate_limit_is_rejected():
    """A negative candidate limit is rejected."""
    with pytest.raises(ValueError):
        RetrievalQuery(text="hello", dense_limit=-1)


def test_a_non_text_query_is_rejected():
    """The query text must be a string."""
    with pytest.raises(TypeError):
        RetrievalQuery(text=123)  # type: ignore[arg-type]


def test_the_item_keeps_the_evidence_apart_from_the_context():
    """The matched chunk is the evidence; the context is separate."""
    parent = ContextChunk(
        chunk_id="p0", text="the section", role="parent", doc_id="doc.md"
    )
    adjacent = ContextChunk(
        chunk_id="c1", text="the neighbour", role="adjacent", doc_id="doc.md"
    )
    item = RetrievedItem(
        chunk_id="c0",
        doc_id="doc.md",
        text="the evidence",
        score=0.5,
        rank=1,
        strategy="dense",
        context=(parent, adjacent),
    )

    assert item.chunk_id == "c0"
    assert item.text == "the evidence"
    assert item.parent_context is parent
    assert item.adjacent_context == (adjacent,)


def test_an_item_without_context_has_none():
    """A strategy that attaches no context leaves the fields empty."""
    item = RetrievedItem(
        chunk_id="c0", doc_id="doc.md", text="text", score=1.0, rank=1, strategy="dense"
    )

    assert item.parent_context is None
    assert item.adjacent_context == ()


def test_the_item_records_the_channels_that_found_it():
    """The per-channel ranks and scores are kept."""
    item = RetrievedItem(
        chunk_id="c0",
        doc_id="doc.md",
        text="text",
        score=0.03,
        rank=1,
        strategy="hybrid",
        methods=("dense", "bm25"),
        channel_ranks={"dense": 1, "bm25": 4},
        channel_scores={"dense": 0.9, "bm25": -3.2},
    )

    assert item.methods == ("dense", "bm25")
    assert item.channel_ranks["bm25"] == 4
    assert item.channel_scores["dense"] == 0.9


def test_the_response_is_iterable():
    """The response behaves like a list of results."""
    results = [
        RetrievedItem(
            chunk_id=f"c{index}", doc_id="doc.md", text="t", score=1.0, rank=index + 1
        )
        for index in range(3)
    ]
    response = RetrievalResponse(results=results, strategy="dense")

    assert len(response) == 3
    assert [r.chunk_id for r in response] == ["c0", "c1", "c2"]
    assert response[0].chunk_id == "c0"
    assert response.chunk_ids == ("c0", "c1", "c2")


def test_the_candidate_reduction_ratio():
    """The reduction ratio is the fraction of the candidates that was skipped."""
    diagnostics = RetrievalDiagnostics(candidates_searched=25, _candidates_total=100)

    assert diagnostics.candidates_total == 100
    assert diagnostics.candidate_reduction_ratio == pytest.approx(0.75)


def test_an_unknown_candidate_total_gives_no_ratio():
    """An unknown total is reported as no ratio, not as a zero reduction."""
    diagnostics = RetrievalDiagnostics(candidates_searched=25)

    assert diagnostics.candidates_total == 0
    assert diagnostics.candidate_reduction_ratio == 0.0


def test_the_candidate_total_is_resolved_lazily():
    """A callable total is only resolved when it is read."""
    calls: list[int] = []

    def total() -> int:
        calls.append(1)
        return 100

    diagnostics = RetrievalDiagnostics(candidates_searched=50, _candidates_total=total)
    assert calls == []
    assert diagnostics.candidates_total == 100
    assert calls == [1]


def test_the_channel_failures_are_reported():
    """A failing channel is visible in the diagnostics."""
    diagnostics = RetrievalDiagnostics(
        channel_candidates={"dense": 10, "bm25": 0},
        channel_failures={"bm25": "RuntimeError: boom"},
    )

    assert diagnostics.channel_candidates["dense"] == 10
    assert "bm25" in diagnostics.channel_failures


def test_the_capabilities_report_what_is_supported():
    """The capabilities describe the retrieval abilities."""
    capabilities = RetrievalCapabilities(
        dense=True,
        lexical=True,
        fusion=True,
        metadata_filtering=True,
        filter_ops=frozenset({FilterOp.EQ, FilterOp.IN}),
        postprocessing=frozenset({"deduplication"}),
    )

    assert capabilities.dense
    assert capabilities.lexical
    assert capabilities.fusion
    assert capabilities.supports(FilterOp.EQ)
    assert not capabilities.supports(FilterOp.GT)
    assert capabilities.supports_filtering
    assert "deduplication" in capabilities.postprocessing


def test_the_capabilities_are_serializable():
    """The capabilities can be reported as a dictionary."""
    capabilities = RetrievalCapabilities(dense=True)

    data = capabilities.as_dict()
    assert data["dense"] is True
    assert data["lexical"] is False
    assert isinstance(data["filter_ops"], list)


def test_the_retriever_protocol_is_runtime_checkable():
    """The retriever protocol can be used with isinstance."""

    class Fake:
        name = "fake"

        @property
        def capabilities(self) -> RetrievalCapabilities:
            return RetrievalCapabilities()

        def retrieve(self, query: RetrievalQuery) -> RetrievalResponse:
            return RetrievalResponse()

    assert isinstance(Fake(), Retriever)
    assert not isinstance(object(), Retriever)
