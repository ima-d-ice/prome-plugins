"""Tests for incremental source synchronization (SQLite state, changes)."""

from __future__ import annotations

import hashlib

import pytest

from loaders.contracts.models import (
    LoadedResource,
    SourceProvenance,
    SourceReference,
    SourceType,
)
from loaders.sync import ChangeDetector, Reconciler, SyncState, SyncStatePersister


def make_resource(source_id: str, content: bytes) -> LoadedResource:
    return LoadedResource(
        reference=SourceReference(
            source_id=source_id,
            source_type=SourceType.LOCAL_FILE,
            original_uri_or_path=f"/data/{source_id}",
        ),
        content=content,
        content_type="text",
        content_hash=hashlib.sha256(content).hexdigest(),
        provenance=SourceProvenance(
            content_hash=hashlib.sha256(content).hexdigest(),
        ),
    )


@pytest.fixture()
def persister(tmp_path):
    return SyncStatePersister(tmp_path / "sync-state.db")


def test_initial_sync_creates_state(persister):
    reconciler = Reconciler(persister)
    result = reconciler.reconcile(
        "source-1",
        SourceType.LOCAL_DIRECTORY,
        [make_resource("a", b"content-a"), make_resource("b", b"content-b")],
    )
    assert result["reconciled"] is True
    assert set(result["change_report"]["new"]) == {"a", "b"}
    state = persister.get_state("source-1")
    assert state is not None
    assert state.operation in ("initial", "reconcile")


def test_modified_file_detected(persister):
    reconciler = Reconciler(persister)
    reconciler.reconcile(
        "source-1",
        SourceType.LOCAL_DIRECTORY,
        [make_resource("a", b"one")],
    )
    result = reconciler.reconcile(
        "source-1",
        SourceType.LOCAL_DIRECTORY,
        [make_resource("a", b"ONE")],
    )
    assert "a" in result["change_report"]["modified"]


def test_unchanged_file_detected(persister):
    reconciler = Reconciler(persister)
    reconciler.reconcile(
        "source-1",
        SourceType.LOCAL_DIRECTORY,
        [make_resource("a", b"same")],
    )
    result = reconciler.reconcile(
        "source-1",
        SourceType.LOCAL_DIRECTORY,
        [make_resource("a", b"same")],
    )
    assert "a" in result["change_report"]["unchanged"]
    assert result["change_report"]["new"] == []


def test_deleted_file_detected(persister):
    reconciler = Reconciler(persister)
    reconciler.reconcile(
        "source-1",
        SourceType.LOCAL_DIRECTORY,
        [make_resource("a", b"a"), make_resource("b", b"b")],
    )
    result = reconciler.reconcile(
        "source-1",
        SourceType.LOCAL_DIRECTORY,
        [make_resource("a", b"a")],
    )
    assert "b" in result["change_report"]["deleted"]


def test_state_persists_across_reopens(tmp_path):
    db = tmp_path / "reopen.db"
    persister1 = SyncStatePersister(db)
    reconciler = Reconciler(persister1)
    reconciler.reconcile(
        "source-1",
        SourceType.LOCAL_DIRECTORY,
        [make_resource("a", b"hello")],
    )
    del persister1

    persister2 = SyncStatePersister(db)
    state = persister2.get_state("source-1")
    assert state is not None
    assert state.tracked_urls["a"]["content_hash"] == hashlib.sha256(b"hello").hexdigest()


def test_explicit_deletion_handling(persister):
    reconciler = Reconciler(persister)
    reconciler.reconcile(
        "source-1",
        SourceType.LOCAL_DIRECTORY,
        [make_resource("a", b"a")],
    )
    result = reconciler.handle_deletion("source-1", {"a"}, reason="user_request")
    assert "a" in result["deleted_source_ids"]
    state = persister.get_state("source-1")
    assert "a" in state.deleted_files


def test_etag_change_marks_modified(persister):
    reconciler = Reconciler(persister)
    res = make_resource("a", b"content")
    res.provenance.etag = type("ETag", (), {"tag": "etag-1"})()
    # Simpler: construct real ETag
    from loaders.contracts.models import ETag

    res.provenance.etag = ETag(tag="etag-1")
    reconciler.reconcile("source-1", SourceType.URL, [res])

    res2 = make_resource("a", b"content")
    res2.provenance.etag = ETag(tag="etag-2")
    result = reconciler.reconcile("source-1", SourceType.URL, [res2])
    assert "a" in result["change_report"]["modified"]


def test_change_detector_pure(persister):
    detector = ChangeDetector()
    previous = SyncState(source_id="s", source_type=SourceType.LOCAL_DIRECTORY)
    previous.tracked_urls = {
        "keep": {"content_hash": hashlib.sha256(b"keep").hexdigest(), "etag": None},
        "gone": {"content_hash": hashlib.sha256(b"gone").hexdigest(), "etag": None},
    }
    current = [make_resource("keep", b"keep"), make_resource("new", b"new")]
    detection = detector.detect_changes(previous, current)
    assert detection["new"] == ["new"]
    assert detection["unchanged"] == ["keep"]
    assert detection["deleted"] == ["gone"]


def test_delete_state(persister):
    reconciler = Reconciler(persister)
    reconciler.reconcile(
        "source-1", SourceType.LOCAL_DIRECTORY, [make_resource("a", b"x")]
    )
    persister.delete_state("source-1")
    assert persister.get_state("source-1") is None
