"""Tests for the upload manager: batches, scopes, atomicity, lifecycle."""

from __future__ import annotations

import pytest

from loaders.uploads import (
    UploadedFileStatus,
    UploadManager,
    UploadScope,
)
from tests.loaders.conftest import make_uploaded_file


@pytest.fixture()
def manager(tmp_path):
    return UploadManager(
        storage_root=tmp_path / "store",
        session_root=tmp_path / "sessions",
    )


def test_upload_single_file_conversation_scope(manager):
    result = manager.upload_files(
        conversation_id="conv-1",
        files=[make_uploaded_file("a.txt", b"hello world")],
    )
    assert len(result.successful) == 1
    assert len(result.rejected) == 0
    assert result.successful[0].status is UploadedFileStatus.STORED
    assert result.successful[0].processing_status is UploadedFileStatus.READY_FOR_INGESTION
    # Uploaded is not the same as available for retrieval
    assert result.successful[0].status is not UploadedFileStatus.AVAILABLE


def test_upload_multiple_files_partial_success(manager):
    result = manager.upload_files(
        conversation_id="conv-1",
        files=[
            make_uploaded_file("good.txt", b"good content"),
            make_uploaded_file("bad.exe", b"MZ"),
        ],
    )
    assert [f.sanitized_filename for f in result.successful] == ["good.txt"]
    assert [f.sanitized_filename for f in result.rejected] == ["bad.exe"]
    assert "bad.exe" in result.per_file_errors


def test_atomic_upload_all_or_nothing(manager):
    result = manager.upload_files(
        conversation_id="conv-1",
        files=[
            make_uploaded_file("ok.txt", b"ok content"),
            make_uploaded_file("bad.exe", b"MZ"),
        ],
        atomic=True,
    )
    assert len(result.successful) == 0
    assert len(result.rejected) == 2
    assert manager.list_attachments("conv-1") == []


def test_atomic_upload_all_valid(manager):
    result = manager.upload_files(
        conversation_id="conv-1",
        files=[
            make_uploaded_file("a.txt", b"content a"),
            make_uploaded_file("b.txt", b"content b"),
        ],
        atomic=True,
    )
    assert len(result.successful) == 2
    assert len(result.rejected) == 0


def test_conversation_isolation(manager):
    result_a = manager.upload_files(
        conversation_id="conv-a",
        files=[make_uploaded_file("secret-a.txt", b"conversation A secret")],
    )
    fid = result_a.successful[0].file_id

    # Conversation B must not see A's file
    assert manager.list_attachments("conv-b") == []
    assert manager.get_attachment("conv-b", fid) is None
    assert manager.delete_attachment("conv-b", fid) is False

    # Conversation A sees it
    attachments = manager.list_attachments("conv-a")
    assert len(attachments) == 1
    assert manager.get_attachment("conv-a", fid)["sanitized_filename"] == "secret-a.txt"


def test_knowledge_base_upload_separate_from_conversation(manager):
    result = manager.upload_files(
        knowledge_base_id="kb-1",
        scope=UploadScope.KNOWLEDGE_BASE,
        files=[make_uploaded_file("kb.txt", b"knowledge base content")],
    )
    assert len(result.successful) == 1
    assert result.successful[0].knowledge_base_id == "kb-1"
    assert result.successful[0].conversation_id is None
    # KB uploads are not listed as conversation attachments
    assert manager.list_attachments("kb-1") == []


def test_knowledge_base_duplicate_detection(manager):
    manager.upload_files(
        knowledge_base_id="kb-1",
        scope=UploadScope.KNOWLEDGE_BASE,
        files=[make_uploaded_file("a.txt", b"duplicate me", file_id="fid-dup-1")],
    )
    result = manager.upload_files(
        knowledge_base_id="kb-1",
        scope=UploadScope.KNOWLEDGE_BASE,
        files=[make_uploaded_file("b.txt", b"duplicate me", file_id="fid-dup-2")],
    )
    assert len(result.successful) == 0
    assert result.duplicate_status == "duplicate"
    assert any("Duplicate" in e for e in result.per_file_errors.values())


def test_same_conversation_duplicate_detection(manager):
    manager.upload_files(
        conversation_id="conv-1",
        files=[make_uploaded_file("a.txt", b"same content", file_id="fid-d1")],
    )
    result = manager.upload_files(
        conversation_id="conv-1",
        files=[make_uploaded_file("b.txt", b"same content", file_id="fid-d2")],
    )
    assert len(result.successful) == 0
    assert result.duplicate_status == "duplicate"


def test_cross_conversation_same_content_allowed(manager):
    manager.upload_files(
        conversation_id="conv-a",
        files=[make_uploaded_file("a.txt", b"shared content", file_id="fid-x1")],
    )
    result = manager.upload_files(
        conversation_id="conv-b",
        files=[make_uploaded_file("b.txt", b"shared content", file_id="fid-x2")],
    )
    # No cross-conversation hash leakage: same content is fine in another conversation
    assert len(result.successful) == 1


def test_batch_file_count_limit(tmp_path):
    manager = UploadManager(
        storage_root=tmp_path / "store",
        session_root=tmp_path / "sessions",
        max_files_per_batch=2,
    )
    files = [make_uploaded_file(f"f{i}.txt", f"content {i}", file_id=f"fid-{i}") for i in range(5)]
    result = manager.upload_files(conversation_id="conv-1", files=files, atomic=True)
    assert len(result.rejected) == 5
    assert result.successful == []


def test_upload_result_has_batch_id(manager):
    result = manager.upload_files(
        conversation_id="conv-1",
        files=[make_uploaded_file("a.txt", b"abc")],
    )
    assert result.batch_id
    assert result.scope is UploadScope.CONVERSATION


def test_never_silently_truncates(tmp_path):
    manager = UploadManager(
        storage_root=tmp_path / "store",
        session_root=tmp_path / "sessions",
        max_file_size=5,
    )
    result = manager.upload_files(
        conversation_id="conv-1",
        files=[make_uploaded_file("big.txt", b"x" * 50)],
    )
    assert result.successful == []
    assert result.rejected, "oversize file must be rejected, never truncated"
