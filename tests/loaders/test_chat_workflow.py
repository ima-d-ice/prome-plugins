"""End-to-end chat upload workflow tests.

Models how a future chat application will drive the upload backend:
upload multiple files to one conversation, list attachments, remove
attachments, expire attachments, and enforce cross-conversation scope.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from loaders.uploads import UploadManager
from loaders.uploads.models import UploadedFileStatus
from tests.loaders.conftest import make_uploaded_file


@pytest.fixture()
def app(tmp_path):
    """Simulated chat application session with the upload backend."""
    return UploadManager(
        storage_root=tmp_path / "store",
        session_root=tmp_path / "sessions",
        default_expiry_hours=1,
    )


def test_upload_files_to_conversation(app):
    """The upload button flow: several files, one conversation."""
    result = app.upload_files(
        conversation_id="chat-42",
        files=[
            make_uploaded_file("report.pdf", b"%PDF-1.4\nreport body", file_id="chat-1"),
            make_uploaded_file("notes.txt", b"conversation notes", file_id="chat-2"),
        ],
    )
    assert len(result.successful) == 2
    attachments = app.list_attachments("chat-42")
    assert len(attachments) == 2
    assert all(a["conversation_id"] == "chat-42" for a in attachments)


def test_workflow_list_and_remove_attachment(app):
    app.upload_files(
        conversation_id="chat-42",
        files=[make_uploaded_file("keep.txt", b"keep", file_id="rm-1"),
               make_uploaded_file("remove.txt", b"remove", file_id="rm-2")],
    )
    names = {a["sanitized_filename"] for a in app.list_attachments("chat-42")}
    assert names == {"keep.txt", "remove.txt"}

    assert app.delete_attachment("chat-42", "rm-2") is True
    names = {a["sanitized_filename"] for a in app.list_attachments("chat-42")}
    assert names == {"keep.txt"}


def test_workflow_expire_attachments(app):
    app.upload_files(
        conversation_id="chat-42",
        files=[make_uploaded_file("expiring.txt", b"bye", file_id="exp-1")],
    )
    attachment = app.lifecycle.get_attachment("exp-1")
    assert attachment is not None
    attachment.expiry_time = datetime.now(timezone.utc) - timedelta(minutes=1)

    expired = app.lifecycle.expire_old_attachments()
    assert expired == 1
    assert app.list_attachments("chat-42") == []


def test_expired_attachments_not_listed(app):
    app.upload_files(
        conversation_id="chat-42",
        files=[make_uploaded_file("gone.txt", b"bye", file_id="exp-2")],
    )
    attachment = app.lifecycle.get_attachment("exp-2")
    attachment.expiry_time = datetime.now(timezone.utc) - timedelta(minutes=1)
    app.lifecycle.expire_old_attachments()
    assert app.get_attachment("chat-42", "exp-2") is None


def test_workflow_cross_conversation_isolation(app):
    result = app.upload_files(
        conversation_id="chat-A",
        files=[make_uploaded_file("secret.txt", b"A secret", file_id="iso-1")],
    )
    fid = result.successful[0].file_id

    # Conversation B sees nothing
    assert app.list_attachments("chat-B") == []
    assert app.get_attachment("chat-B", fid) is None
    # Conversation A sees the attachment
    assert len(app.list_attachments("chat-A")) == 1


def test_uploaded_vs_processed_distinction(app):
    """Upload success does not claim RAG availability; processing is separate."""
    result = app.upload_files(
        conversation_id="chat-42",
        files=[make_uploaded_file("pending.txt", b"pending", file_id="proc-1")],
    )
    file = result.successful[0]
    assert file.processing_status is UploadedFileStatus.READY_FOR_INGESTION
    assert file.status is not UploadedFileStatus.AVAILABLE

    # Simulate the downstream pipeline marking the file available
    app.lifecycle.mark_processing(file)
    app.lifecycle.mark_available(file)
    assert file.processing_status is UploadedFileStatus.AVAILABLE
    assert file.status is UploadedFileStatus.AVAILABLE


def test_delete_all_conversation_attachments(app):
    app.upload_files(
        conversation_id="chat-42",
        files=[
            make_uploaded_file("a.txt", b"a", file_id="all-1"),
            make_uploaded_file("b.txt", b"b", file_id="all-2"),
        ],
    )
    deleted = app.delete_conversation_attachments("chat-42")
    assert deleted == 2
    assert app.list_attachments("chat-42") == []


def test_failed_upload_keeps_diagnostics(app):
    result = app.upload_files(
        conversation_id="chat-42",
        files=[make_uploaded_file("corrupt.exe", b"MZ")],
    )
    assert result.rejected
    rejected = result.rejected[0]
    assert rejected.metadata.get("last_error")
    # A failed file is not available to the conversation
    assert app.list_attachments("chat-42") == []
