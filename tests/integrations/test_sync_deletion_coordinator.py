"""Tests for the Sync/Deletion coordinator."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from integrations.sync_deletion_coordinator import IndexSyncCoordinator


class FakeVectorStore:
    """Minimal vector store fake for coordinator testing."""

    def __init__(self, name="fake", delete_by_document_raises=True, delete_by_document_return=0, delete_by_filter_return=0):
        self.name = name
        self.upsert_call_count = 0
        self.delete_by_document_call_count = 0
        self.delete_by_filter_call_count = 0
        self.upsert_return = 0
        # When delete_by_document_raises is True, delete_by_document always raises TypeError
        # When False, it returns the delete_by_document_return value
        self._delete_by_document_raises = delete_by_document_raises
        self.delete_by_document_return = delete_by_document_return
        self.delete_by_filter_return = delete_by_filter_return

    def upsert(self, collection, records):
        self.upsert_call_count += 1
        return self.upsert_return

    def delete_by_document(self, collection, doc_id):
        self.delete_by_document_call_count += 1
        if self._delete_by_document_raises:
            raise TypeError("delete_by_document not available")
        return self.delete_by_document_return

    def delete_by_filter(self, collection, filter):
        self.delete_by_filter_call_count += 1
        return self.delete_by_filter_return


def test_coordinator_deletes_by_document_when_available():
    """Coordinator uses delete_by_document when the store supports it."""
    vector_store = FakeVectorStore("test", delete_by_document_raises=False, delete_by_document_return=5)

    coord = IndexSyncCoordinator(vector_store=vector_store, collection="kb-1")
    result = coord.delete_source_by_id(source_id="s-1", doc_id="doc-1")

    assert result["vector_store"]["success"] is True
    assert result["vector_store"]["deleted"] == 5
    # Coordinator calls delete_by_document on both vector and lexical stores;
    # lexical_store defaults to vector_store, so 2 calls total
    assert vector_store.delete_by_document_call_count == 2


def test_coordinator_falls_back_to_filter_when_delete_by_document_unavailable():
    """Coordinator falls back to delete_by_filter when delete_by_document raises TypeError."""
    vector_store = FakeVectorStore("filter-store", delete_by_document_raises=True, delete_by_filter_return=2)

    coord = IndexSyncCoordinator(vector_store=vector_store, collection="kb-1")
    result = coord.delete_source_by_id(source_id="s-1", doc_id=None)

    assert result["vector_store"]["success"] is True
    assert vector_store.delete_by_filter_call_count >= 1


def test_coordinator_derives_doc_id_from_source_when_not_supplied():
    """When doc_id not supplied, coordinator derives it from source_id."""
    vector_store = FakeVectorStore("test", delete_by_document_raises=False, delete_by_document_return=4)

    coord = IndexSyncCoordinator(vector_store=vector_store, collection="kb-1")
    result = coord.delete_source_by_id(source_id="s-99")

    assert result["vector_store"]["success"] is True
    assert result["vector_store"]["deleted"] == 4
    assert vector_store.delete_by_document_call_count == 2  # vector + lexical (same store)


def test_coordinator_deletes_from_lexical_store_separately():
    """Coordinator deletes from both vector and lexical stores."""
    vector_store = FakeVectorStore("vector", delete_by_document_raises=False, delete_by_document_return=3)

    lexical_store = FakeVectorStore("lexical", delete_by_document_raises=False, delete_by_document_return=3)

    coord = IndexSyncCoordinator(
        vector_store=vector_store,
        lexical_store=lexical_store,
        collection="kb-1",
    )
    result = coord.delete_source_by_id(source_id="s-1", doc_id="doc-1")

    assert result["vector_store"]["success"] is True
    assert result["lexical_store"]["success"] is True
    assert result["vector_store"]["deleted"] == 3
    assert result["lexical_store"]["deleted"] == 3


def test_coordinator_idempotent_repeated_deletion():
    """Repeated deletion is idempotent."""
    vector_store = FakeVectorStore("test", delete_by_document_raises=True, delete_by_document_return=0)  # already deleted

    coord = IndexSyncCoordinator(vector_store=vector_store, collection="kb-1")
    result1 = coord.delete_source_by_id(source_id="s-1", doc_id="doc-1")
    result2 = coord.delete_source_by_id(source_id="s-1", doc_id="doc-1")

    assert result1["vector_store"]["success"] is True
    assert result2["vector_store"]["success"] is True
    # 2 direct calls + 2 filter fallbacks (TypeError on both vector + lexical) = 4
    assert vector_store.delete_by_document_call_count == 4


def test_coordinator_upsert_records():
    """Coordinator upserts records to vector and lexical stores."""
    vector_store = FakeVectorStore("vector", delete_by_document_raises=False, delete_by_document_return=0)
    vector_store.upsert_return = 5

    lexical_store = FakeVectorStore("lexical", delete_by_document_raises=False, delete_by_document_return=0)
    lexical_store.upsert_return = 5

    coord = IndexSyncCoordinator(
        vector_store=vector_store,
        lexical_store=lexical_store,
        collection="kb-1",
    )
    records = [{"chunk": {"id": "ch-1", "doc_id": "doc-1"}, "metadata": {}} for _ in range(5)]
    result = coord.upsert_records(records)

    assert result["vector_store"]["success"] is True
    assert result["vector_store"]["written"] == 5
    assert result["lexical_store"]["success"] is True
    assert result["lexical_store"]["written"] == 5


def test_coordinator_propagates_changes():
    """Coordinator propagates changes via change detection."""
    from loaders.sync.state import SyncState
    from loaders.sync.change_detector import ChangeDetector

    prev_state = SyncState(
        source_id="s-1",
        source_type="local_file",
    )

    curr_resources = [
        type("Res", (), {"reference": type("Ref", (), {"source_id": "s-1"})(), "content": b"test", "content_hash": "hash-123"})(),
    ]

    vector_store = MagicMock()
    vector_store.upsert.return_value = 1
    vector_store.delete_by_document.return_value = 0

    lexical_store = MagicMock()
    lexical_store.upsert.return_value = 1
    lexical_store.delete_by_document.return_value = 0

    coord = IndexSyncCoordinator(
        vector_store=vector_store,
        lexical_store=lexical_store,
        collection="kb-1",
        persister=None,
    )
    report = coord.propagate_changes(prev_state, curr_resources)

    assert "new" in report
    assert "deleted" in report
    assert "vector_store" in report
    assert "lexical_store" in report
