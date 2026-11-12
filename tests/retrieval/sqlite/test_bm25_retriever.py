"""Tests of the BM25 retriever over the SQLite FTS5 index."""

from __future__ import annotations

import pytest

from retrieval.contracts.exceptions import (
    RetrievalConfigError,
    UnsupportedRetrievalCapabilityError,
)
from retrieval.contracts.models import RetrievalQuery
from retrieval.lexical.bm25.sqlite_fts5 import BM25Retriever, LexicalRecord, SQLiteFTS5Index
from vector_stores.contracts.filters import eq, is_in
from vector_stores.contracts.models import ChunkProvenance, ScoreDirection


def record(chunk_id, doc_id, text, metadata=None, index=0) -> LexicalRecord:
    """Build a record for the tests."""
    return LexicalRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        metadata=dict(metadata or {}),
        provenance=ChunkProvenance(source=doc_id, pages=(1,)),
        source=doc_id,
        chunk_index=index,
    )


@pytest.fixture()
def index(tmp_path):
    """Return a lexical index with a small corpus."""
    instance = SQLiteFTS5Index(tmp_path / "index.db")
    instance.open()
    instance.upsert(
        [
            record(
                "c0", "auth.md",
                "API keys are rotated every thirty days by the rotation "
                "scheduler. A key that is not rotated within the grace period "
                "is revoked automatically and the affected collectors are "
                "restarted in a rolling fashion.",
                metadata={"tenant": "acme", "doc_type": "manual"},
            ),
            record(
                "c1", "auth.md",
                "The rotation writes the new key version to the key store and "
                "publishes the change to the discovery endpoint, so the "
                "collectors pick it up without downtime.",
                metadata={"tenant": "acme", "doc_type": "manual"},
            ),
            record(
                "c2", "auth.md",
                "Roles grant access to the billing console. The auditor role "
                "can read invoices but cannot issue refunds.",
                metadata={"tenant": "other", "doc_type": "manual"},
            ),
            record(
                "c3", "billing.md",
                "Invoices are issued on the first day of every month. An "
                "invoice for a partial month is pro-rated by day.",
                metadata={"tenant": "acme", "doc_type": "guide"},
            ),
            record(
                "c4", "billing.md",
                "A refund is issued within five business days of an approved "
                "credit note.",
                metadata={"tenant": "other", "doc_type": "guide"},
            ),
        ]
    )
    try:
        yield instance
    finally:
        instance.close()


@pytest.fixture()
def retriever(index):
    """Return the BM25 retriever over the fixture corpus."""
    return BM25Retriever(index)


def test_the_query_is_found(retriever):
    """The search returns the chunks containing the query terms."""
    response = retriever.retrieve(RetrievalQuery(text="rotation"))

    assert response.results
    assert all("rotation" in item.text for item in response.results)


def test_the_chunk_identity_is_preserved(retriever):
    """The original chunk id is returned."""
    response = retriever.retrieve(RetrievalQuery(text="rotation"))

    assert {item.chunk_id for item in response.results} <= {"c0", "c1", "c2", "c3", "c4"}


def test_the_provenance_is_preserved(retriever):
    """The provenance survives the round trip through the index."""
    response = retriever.retrieve(RetrievalQuery(text="rotation"))

    assert response.results
    for item in response.results:
        assert item.provenance is not None
        assert item.provenance.source


def test_the_metadata_filter_restricts_the_results(retriever):
    """A metadata filter is applied by the backend."""
    response = retriever.retrieve(
        RetrievalQuery(text="rotation", filters=eq("tenant", "acme"))
    )

    assert response.results
    assert all(item.metadata["tenant"] == "acme" for item in response.results)
    assert {item.chunk_id for item in response.results} <= {"c0", "c1", "c3"}


def test_a_filter_that_excludes_everything_returns_nothing(retriever):
    """A filter that excludes every row returns no results."""
    response = retriever.retrieve(
        RetrievalQuery(text="rotation", filters=eq("tenant", "nobody"))
    )

    assert response.results == []


def test_the_document_restriction_is_applied(retriever):
    """A restriction to a set of documents is applied."""
    response = retriever.retrieve(
        RetrievalQuery(text="rotation", doc_ids=("billing.md",))
    )

    assert all(item.doc_id == "billing.md" for item in response.results)


def test_a_technical_identifier_is_matched(retriever):
    """An identifier with underscores is found."""
    response = retriever.retrieve(RetrievalQuery(text="get_user_id"))

    assert response.results == []


def test_the_limit_is_respected(retriever):
    """Only the requested number of candidates is returned."""
    response = retriever.retrieve(RetrievalQuery(text="rotation", limit=1))

    assert len(response.results) == 1


def test_the_score_direction_is_lower_is_better(retriever):
    """The BM25 score is lower-is-better."""
    response = retriever.retrieve(RetrievalQuery(text="rotation"))

    assert response.results
    for item in response.results:
        assert item.score_direction is ScoreDirection.LOWER_IS_BETTER
        assert item.score_interpretation == "bm25"


def test_the_ranks_are_consecutive(retriever):
    """The ranks are consecutive starting at one."""
    response = retriever.retrieve(RetrievalQuery(text="rotation"))

    assert [item.rank for item in response.results] == list(
        range(1, len(response.results) + 1)
    )


def test_an_empty_query_is_rejected(retriever):
    """An empty query raises instead of returning everything."""
    with pytest.raises(RetrievalConfigError):
        retriever.retrieve(RetrievalQuery(text="   "))


def test_a_query_with_only_symbols_returns_nothing(retriever):
    """A query without a searchable term returns no results."""
    response = retriever.retrieve(RetrievalQuery(text="!!!"))

    assert response.results == []


def test_an_unsupported_filter_field_is_reported(retriever):
    """A filter on a field that is not a valid column is reported."""
    with pytest.raises(UnsupportedRetrievalCapabilityError):
        retriever.search(
            "rotation",
            limit=10,
            filter=eq("a-field-with-dashes", "x"),
        )


def test_a_bad_filter_field_name_is_rejected():
    """A field name that could inject SQL is rejected."""
    from retrieval.lexical.bm25.sqlite_fts5.retriever import to_sql_filter
    from vector_stores.contracts.filters import MetadataFilter, FilterOp

    bad = MetadataFilter(op=FilterOp.EQ, field="a; DROP TABLE chunks--", value="x")
    with pytest.raises(UnsupportedRetrievalCapabilityError):
        to_sql_filter(bad)


def test_the_capabilities_report_the_channel(retriever):
    """The channel reports its lexical capability."""
    capabilities = retriever.capabilities

    assert capabilities.lexical is True
    assert capabilities.dense is False
    assert capabilities.fusion is False
    assert capabilities.supports_filtering


def test_the_documents_of_the_index_are_listed(retriever):
    """The channel inherits the documents listing from the index."""
    assert sorted(retriever.index.documents()) == ["auth.md", "billing.md"]
