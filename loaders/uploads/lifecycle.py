"""Attachment lifecycle management for Prome RAG Plugins."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from loaders.uploads.models import UploadedFile, UploadedFileStatus


class AttachmentStatus:
    """Status of a conversation attachment."""

    RECEIVED = "received"
    VALIDATING = "validating"
    STORED = "stored"
    READY_FOR_INGESTION = "ready_for_ingestion"
    PROCESSING = "processing"
    AVAILABLE = "available"
    FAILED = "failed"
    EXPIRED = "expired"
    DELETED = "deleted"


class AttachmentLifecycle:
    """Manages the lifecycle of a conversation attachment."""

    def __init__(
        self,
        storage_backend,
        session_manager,
        *,
        default_expiry_hours: int = 168,  # 7 days
        expiry_on_ingest: bool = True,
    ):
        self.storage = storage_backend
        self.sessions = session_manager
        self.default_expiry_hours = default_expiry_hours
        self.expiry_on_ingest = expiry_on_ingest
        self._attachments: dict[str, UploadedFile] = {}  # attachment_id -> UploadedFile

    def register(self, file: UploadedFile, conversation_id: str | None = None) -> UploadedFile:
        """Register an already-stored file in the lifecycle tracker."""
        if conversation_id:
            file.conversation_id = conversation_id
        self._attachments[file.file_id] = file
        return file

    def start_processing(self, file: UploadedFile, conversation_id: str) -> UploadedFile:
        """Move a file through the attachment lifecycle."""
        file.conversation_id = conversation_id
        file.upload_timestamp = datetime.now(timezone.utc)
        file.status = UploadedFileStatus.RECEIVED
        file.validation_status = "pending"

        # Store the file
        storage_path = self.storage.store(file)
        file.storage_reference = storage_path
        file.status = UploadedFileStatus.STORED

        # Mark as ready for ingestion
        file.processing_status = UploadedFileStatus.READY_FOR_INGESTION

        # Track the attachment
        self._attachments[file.file_id] = file

        return file

    def mark_validating(self, file: UploadedFile) -> UploadedFile:
        """Mark file as being validated."""
        file.validation_status = "validating"
        file.status = UploadedFileStatus.VALIDATING
        return file

    def mark_validated(self, file: UploadedFile) -> UploadedFile:
        """Mark file as validated and ready for ingestion."""
        file.validation_status = "valid"
        file.status = UploadedFileStatus.STORED
        file.processing_status = UploadedFileStatus.READY_FOR_INGESTION
        return file

    def mark_processing(self, file: UploadedFile) -> UploadedFile:
        """Mark file as being processed."""
        file.processing_status = UploadedFileStatus.PROCESSING
        file.status = UploadedFileStatus.PROCESSING
        return file

    def mark_available(self, file: UploadedFile) -> UploadedFile:
        """Mark file as available for conversation use."""
        file.status = UploadedFileStatus.AVAILABLE
        file.processing_status = UploadedFileStatus.AVAILABLE
        return file

    def mark_failed(self, file: UploadedFile, error: str) -> UploadedFile:
        """Mark file as failed with diagnostics."""
        file.status = UploadedFileStatus.FAILED
        file.processing_status = UploadedFileStatus.FAILED
        # Preserve failure diagnostics in metadata
        if file.metadata is None:
            file.metadata = {}
        file.metadata["last_error"] = error
        file.metadata["last_error_timestamp"] = datetime.now(timezone.utc).isoformat()
        return file

    def mark_expired(self, file: UploadedFile) -> UploadedFile:
        """Mark file as expired."""
        file.status = UploadedFileStatus.EXPIRED
        file.processing_status = UploadedFileStatus.EXPIRED
        return file

    def mark_deleted(self, file: UploadedFile) -> UploadedFile:
        """Mark file as deleted and clean up storage."""
        # Delete from storage
        self.storage.delete(file.file_id)

        # Remove from tracking
        file.status = UploadedFileStatus.DELETED
        file.processing_status = UploadedFileStatus.DELETED
        self._attachments.pop(file.file_id, None)
        return file

    def get_attachment(self, file_id: str) -> UploadedFile | None:
        """Get an attachment by file ID."""
        return self._attachments.get(file_id)

    def list_attachments(
        self,
        conversation_id: str,
        *,
        include_expired: bool = False,
    ) -> list[UploadedFile]:
        """List attachments for a conversation."""
        attachments = [
            f for f in self._attachments.values()
            if f.conversation_id == conversation_id
        ]
        if not include_expired:
            attachments = [
                a for a in attachments
                if a.status not in (UploadedFileStatus.EXPIRED, UploadedFileStatus.DELETED)
            ]
        return attachments

    def delete_attachment(self, file_id: str) -> bool:
        """Delete a single attachment."""
        file = self._attachments.pop(file_id, None)
        if file is None:
            return False
        return self.storage.delete(file.file_id)

    def delete_conversation_attachments(self, conversation_id: str) -> int:
        """Delete all attachments for a conversation. Returns count deleted."""
        to_delete = [
            f for f in self._attachments.values()
            if f.conversation_id == conversation_id
        ]
        for file in to_delete:
            self.storage.delete(file.file_id)
        for file in to_delete:
            self._attachments.pop(file.file_id, None)
        return len(to_delete)

    def check_expiry(self, file: UploadedFile) -> bool:
        """Check if a file has expired based on its expiry configuration."""
        expiry_time = getattr(file, "expiry_time", None)
        if expiry_time is None and file.upload_timestamp:
            # Use default expiry
            expiry_time = file.upload_timestamp + timedelta(hours=self.default_expiry_hours)
        if expiry_time:
            return datetime.now(timezone.utc) > expiry_time
        return False

    def expire_old_attachments(self) -> int:
        """Expire attachments that have exceeded their expiry. Returns count expired."""
        expired_count = 0
        to_expire: list[UploadedFile] = []

        for file in self._attachments.values():
            if self.check_expiry(file) and file.status not in (UploadedFileStatus.EXPIRED, UploadedFileStatus.DELETED):
                to_expire.append(file)

        for file in to_expire:
            self.mark_expired(file)
            self.storage.delete(file.file_id)
            expired_count += 1

        return expired_count