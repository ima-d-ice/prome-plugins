"""Tests of the SQLite FTS5 lexical index.

These are integration tests: the index is a real SQLite database, written
to disk so that the persistence behaviour is verified.
"""

from __future__ import annotations

import pytest

from retrieval.contracts.exceptions import (
    FTS5UnavailableError,
    LexicalIndexError,
)
from retrieval.lexical.bm25.sqlite_fts5 import (
    LexicalRecord,
    SQLiteFTS5Index,
    fts5_available,
)


def record(chunk_id, doc_id, text, metadata=None, index=0) -> LexicalRecord:
    """Build a record for the tests."""
    return LexicalRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        metadata=dict(metadata or {}),
        chunk_index=index,
    )


@pytest.fixture()
def index(tmp_path):
    """Return an open lexical index in a temporary directory."""
    instance = SQLiteFTS5Index(tmp_path / "index.db")
    instance.open()
    try:
        yield instance
    finally:
        instance.close()


# ----------------------------------------------------------------------
# capabilities and lifecycle
# ----------------------------------------------------------------------
def test_fts5_is_available_in_this_python():
    """The SQLite build of this interpreter provides FTS5."""
    assert fts5_available()


def test_the_index_is_persistent_across_close_and_reopen(index):
    """The records survive a close and a reopen."""
    index.upsert([record("c0", "doc.md", "the api key is rotated")])
    index.close()

    reopened = SQLiteFTS5Index(index.path)
    reopened.open()
    try:
        assert reopened.count() == 1
        records = reopened.get(["c0"])
        assert len(records) == 1
        assert records[0].text == "the api key is rotated"
    finally:
        reopened.close()


def test_an_unavailable_fts5_is_reported_not_worked_around(monkeypatch):
    """A missing FTS5 is an explicit error, never a silent fallback."""
    monkeypatch.setattr(
        "retrieval.lexical.bm25.sqlite_fts5.index.fts5_available",
        lambda connection=None: False,
    )
    with pytest.raises(FTS5UnavailableError):
        SQLiteFTS5Index(":memory:").open()


# ----------------------------------------------------------------------
# writing
# ----------------------------------------------------------------------
def test_the_index_inserts_updates_and_deletes(index):
    """The index supports insert, update and delete."""
    index.upsert([record("c0", "doc.md", "alpha beta")])
    assert index.count() == 1

    index.upsert([record("c0", "doc.md", "gamma delta")])
    records = index.get(["c0"])
    assert records[0].text == "gamma delta"

    index.delete(["c0"])
    assert index.count() == 0


def test_upsert_is_idempotent(index):
    """Writing the same record twice keeps one row."""
    index.upsert([record("c0", "doc.md", "alpha")])
    index.upsert([record("c0", "doc.md", "alpha")])
    assert index.count() == 1


def test_deleting_documents_removes_every_chunk(index):
    """Deleting a document removes all of its chunks."""
    index.upsert(
        [
            record("a0", "a.md", "alpha"),
            record("a1", "a.md", "bravo"),
            record("b0", "b.md", "charlie"),
        ]
    )
    index.delete_documents(["a.md"])
    assert index.count() == 1
    assert index.get(["b0"])[0].doc_id == "b.md"


def test_the_integrity_check_passes_on_a_healthy_index(index):
    """A healthy index passes the integrity check."""
    index.upsert(
        [
            record("c0", "doc.md", "alpha beta"),
            record("c1", "doc.md", "gamma delta"),
        ]
    )
    assert index.integrity_check() is True


def test_an_integrity_check_detects_a_corrupt_index(tmp_path):
    """An index whose content and FTS index disagree fails the check."""
    instance = SQLiteFTS5Index(tmp_path / "index.db")
    instance.open()
    try:
        instance.upsert(
            [
                record("c0", "doc.md", "alpha beta"),
                record("c1", "doc.md", "gamma delta"),
            ]
        )
        assert instance.integrity_check() is True
        # Drop the FTS entry of c0 directly, leaving the content table
        # untouched: the index can no longer answer a query for those terms.
        rowid = instance.connection.execute(
            "SELECT rowid FROM chunks WHERE chunk_id = 'c0'"
        ).fetchone()[0]
        instance.connection.execute(
            "INSERT INTO chunks_fts(chunks_fts, rowid, text) "
            "VALUES ('delete', ?, 'alpha beta')",
            (rowid,),
        )
        instance.connection.commit()
        assert instance.integrity_check() is False
    finally:
        instance.close()


def test_the_documents_of_the_index_are_listed(index):
    """The index reports the ids of the documents it holds."""
    index.upsert(
        [
            record("c0", "a.md", "alpha"),
            record("c1", "b.md", "beta"),
        ]
    )
    assert index.documents() == ["a.md", "b.md"]
