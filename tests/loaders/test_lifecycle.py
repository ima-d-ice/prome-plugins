"""Tests for attachment lifecycle: statuses, expiry, isolation, deletion."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from loaders.uploads import UploadManager
from loaders.uploads.models import UploadedFileStatus
from tests.loaders.conftest import make_uploaded_file


@pytest.fixture()
def manager(tmp_path):
    return UploadManager(
        storage_root=tmp_path / "store",
        session_root=tmp_path / "sessions",
        default_expiry_hours=1,
    )


def test_attachment_available_after_upload(manager):
    result = manager.upload_files(
        conversation_id="conv-1",
        files=[make_uploaded_file("a.txt", b"content")],
    )
    file = result.successful[0]
    assert file.status is UploadedFileStatus.STORED
    assert file.processing_status is UploadedFileStatus.READY_FOR_INGESTION


def test_attachments_list_and_fetch(manager):
    manager.upload_files(
        conversation_id="conv-1",
        files=[
            make_uploaded_file("a.txt", b"aaa", file_id="lf-1"),
            make_uploaded_file("b.txt", b"bbb", file_id="lf-2"),
        ],
    )
    attachments = manager.list_attachments("conv-1")
    assert {a["sanitized_filename"] for a in attachments} == {"a.txt", "b.txt"}

    meta = manager.get_attachment("conv-1", "lf-1")
    assert meta["file_id"] == "lf-1"
    assert meta["content_hash"]


def test_delete_attachment(manager):
    manager.upload_files(
        conversation_id="conv-1",
        files=[make_uploaded_file("a.txt", b"aaa", file_id="del-1")],
    )
    assert manager.list_attachments("conv-1") != []
    assert manager.delete_attachment("conv-1", "del-1") is True
    assert manager.list_attachments("conv-1") == []
    assert manager.get_attachment("conv-1", "del-1") is None


def test_delete_conversation_attachments(manager):
    manager.upload_files(
        conversation_id="conv-1",
        files=[
            make_uploaded_file("a.txt", b"a", file_id="cd-1"),
            make_uploaded_file("b.txt", b"b", file_id="cd-2"),
        ],
    )
    manager.upload_files(
        conversation_id="conv-2",
        files=[make_uploaded_file("c.txt", b"c", file_id="cd-3")],
    )
    deleted = manager.delete_conversation_attachments("conv-1")
    assert deleted == 2
    assert manager.list_attachments("conv-1") == []
    assert len(manager.list_attachments("conv-2")) == 1


def test_attachment_expiry(manager, tmp_path):
    manager.upload_files(
        conversation_id="conv-1",
        files=[make_uploaded_file("expiry.txt", b"expiring", file_id="exp-1")],
    )
    attachment = manager.lifecycle.get_attachment("exp-1")
    attachment.expiry_time = datetime.now(timezone.utc) - timedelta(hours=1)

    expired = manager.lifecycle.expire_old_attachments()
    assert expired == 1
    assert attachment.status is UploadedFileStatus.EXPIRED
    # Expired files are deleted from storage
    assert manager.storage.retrieve("exp-1") is None


def test_failed_attachment_preserves_diagnostics(manager):
    result = manager.upload_files(
        conversation_id="conv-1",
        files=[make_uploaded_file("bad.exe", b"MZ")],
    )
    rejected = result.rejected[0]
    assert rejected.status is UploadedFileStatus.FAILED
    assert rejected.metadata.get("last_error")


def test_upload_status_not_available_until_processed(manager):
    result = manager.upload_files(
        conversation_id="conv-1",
        files=[make_uploaded_file("a.txt", b"abc")],
    )
    file = result.successful[0]
    # Upload success alone does not imply RAG availability
    assert file.status is not UploadedFileStatus.AVAILABLE
    assert file.processing_status is UploadedFileStatus.READY_FOR_INGESTION
