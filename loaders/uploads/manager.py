"""Upload service manager for Prome RAG Plugins."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loaders.contracts.exceptions import (
    FileValidationError,
    LoaderError,
)
from loaders.uploads.lifecycle import AttachmentLifecycle
from loaders.uploads.models import (
    UploadedFile,
    UploadedFileStatus,
    UploadRequest,
    UploadResult,
    UploadScope,
)
from loaders.uploads.sessions import UploadSessionManager
from loaders.uploads.storage import StorageBackend
from loaders.uploads.validation import DEFAULT_MAX_FILE_SIZE, FileValidator


class UploadManager:
    """Manages file uploads with validation, storage, and lifecycle.

    Conversation-scoped uploads are isolated to their conversation.
    Knowledge-base uploads are accepted into persistent storage and
    can be indexed for reuse subject to access control.
    """

    def __init__(
        self,
        *,
        storage_root: str | Path,
        session_root: str | Path | None = None,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
        max_files_per_batch: int = 20,
        max_total_batch_size: int = 500 * 1024 * 1024,
        max_conversation_attachments: int = 50,
        allowed_extensions: Sequence[str] | None = None,
        allowed_mime_types: Sequence[str] | None = None,
        default_expiry_hours: int = 168,
    ):
        self.storage = StorageBackend(storage_root)
        self.validator = FileValidator(
            max_file_size=max_file_size,
            max_files_per_batch=max_files_per_batch,
            max_total_batch_size=max_total_batch_size,
            allowed_extensions=allowed_extensions,
            allowed_mime_types=allowed_mime_types,
        )
        self.session_manager = UploadSessionManager(session_root or storage_root)
        self.lifecycle = AttachmentLifecycle(
            storage_backend=self.storage,
            session_manager=self.session_manager,
            default_expiry_hours=default_expiry_hours,
        )

    def upload_files(
        self,
        conversation_id: str | None = None,
        files: Sequence[UploadedFile] | None = None,
        *,
        knowledge_base_id: str | None = None,
        tenant_id: str | None = None,
        scope: UploadScope = UploadScope.CONVERSATION,
        metadata: dict[str, Any] | None = None,
        request_id: str | None = None,
        atomic: bool = False,
    ) -> UploadResult:
        """Upload one or more files.

        Returns an UploadResult with per-file success/rejection details.
        A single invalid file does not invalidate unrelated valid files
        unless atomic=True.
        """
        request = UploadRequest(
            request_id=request_id or str(uuid.uuid4()),
            conversation_id=conversation_id,
            knowledge_base_id=knowledge_base_id,
            tenant_id=tenant_id,
            scope=scope,
            files=list(files or []),
            metadata=metadata or {},
            atomic=atomic,
        )
        return self.process_request(request)

    def process_request(self, request: UploadRequest) -> UploadResult:
        """Process an entire upload request."""
        result = UploadResult(
            batch_id=str(uuid.uuid4()),
            scope=request.scope,
        )

        is_valid, validation_errors = self.validator.validate_request(request)
        if not is_valid:
            result.diagnostics.extend(validation_errors)
            if request.atomic:
                result.rejected = list(request.files)
                return result
            # Non-atomic: still process what we can, but report errors
            for err in validation_errors:
                result.diagnostics.append(err)

        existing_hashes = self._existing_content_hashes(
            scope=request.scope, conversation_id=request.conversation_id
        )

        if request.atomic:
            return self._process_atomic(request, result, existing_hashes)

        for uploaded_file in request.files:
            try:
                processed = self._process_single_file(uploaded_file, existing_hashes)
                result.successful.append(processed)
            except (LoaderError, OSError, ValueError, TypeError) as e:
                self._reject_file(result, uploaded_file, str(e))

        return self._finalize_result(result, request)

    @staticmethod
    def _reject_file(result: UploadResult, uploaded_file: UploadedFile, error: str) -> None:
        """Mark a file failed and record per-file diagnostics."""
        uploaded_file.status = UploadedFileStatus.FAILED
        uploaded_file.processing_status = UploadedFileStatus.FAILED
        uploaded_file.metadata = uploaded_file.metadata or {}
        uploaded_file.metadata["last_error"] = error
        result.rejected.append(uploaded_file)
        result.per_file_errors[
            uploaded_file.sanitized_filename or uploaded_file.original_filename
        ] = error

    def _process_single_file(
        self, file: UploadedFile, existing_hashes: set[str]
    ) -> UploadedFile:
        """Validate, store and register a single file."""
        self._validate_single_file(file, existing_hashes)
        self._store_file(file)
        return file

    def _process_atomic(
        self, request: UploadRequest, result: UploadResult, existing_hashes: set[str]
    ) -> UploadResult:
        """Atomic upload: either all files are accepted or none are stored."""
        validated: list[UploadedFile] = []

        for uploaded_file in request.files:
            try:
                validated.append(self._validate_single_file(uploaded_file, existing_hashes))
            except (LoaderError, OSError, ValueError, TypeError) as e:
                self._reject_file(result, uploaded_file, str(e))

        if result.rejected:
            # Nothing is stored — the whole batch is rejected
            for f in validated:
                uploaded_file = f
                uploaded_file.status = UploadedFileStatus.FAILED
                uploaded_file.processing_status = UploadedFileStatus.FAILED
                uploaded_file.metadata = uploaded_file.metadata or {}
                uploaded_file.metadata["last_error"] = (
                    "Atomic upload aborted: a sibling file failed validation"
                )
                result.rejected.append(uploaded_file)
                result.per_file_errors[
                    uploaded_file.sanitized_filename or uploaded_file.original_filename
                ] = uploaded_file.metadata["last_error"]
            result.diagnostics.append(
                f"Atomic upload rejected {len(result.rejected)} file(s); no files were stored"
            )
            return result

        for f in validated:
            self._store_file(f)
        result.successful = validated
        return self._finalize_result(result, request)

    def _validate_single_file(
        self, file: UploadedFile, existing_hashes: set[str]
    ) -> UploadedFile:
        """Validate a file and populate hash/sanitized name fields."""
        # Sanitize filename
        is_valid, sanitized, warnings = self.validator.validate_sanitized_filename(
            file.original_filename
        )
        if not is_valid:
            raise FileValidationError(f"Invalid filename: {warnings}")
        file.sanitized_filename = sanitized

        # Compute content hash
        content = file.content
        if isinstance(content, str):
            data = content.encode("utf-8")
        else:
            data = content
        file.content_hash = hashlib.sha256(data).hexdigest()

        # Validate file (size, extension, magic bytes)
        self.validator.validate_file(file)

        # Duplicate detection for knowledge base / same-conversation uploads
        if file.content_hash in existing_hashes:
            file.metadata = file.metadata or {}
            file.metadata["duplicate"] = True
            raise FileValidationError(
                "Duplicate content detected (content hash already exists)"
            )

        return file

    def _store_file(self, file: UploadedFile) -> UploadedFile:
        """Store a validated file and register it for lifecycle tracking."""
        storage_path = self.storage.store(file)
        file.storage_reference = storage_path
        file.status = UploadedFileStatus.STORED
        file.processing_status = UploadedFileStatus.READY_FOR_INGESTION
        file.upload_timestamp = datetime.now(timezone.utc)

        # Track in lifecycle for listing / conversation isolation
        self.lifecycle.register(file)
        return file

    def _finalize_result(self, result: UploadResult, request: UploadRequest) -> UploadResult:
        """Apply scope assignment and duplicate status to a result."""
        if request.scope == UploadScope.CONVERSATION:
            for f in result.successful:
                f.conversation_id = request.conversation_id
        else:
            for f in result.successful:
                f.knowledge_base_id = request.knowledge_base_id

        dup_errors = [
            v for v in result.per_file_errors.values()
            if "duplicate" in v.lower()
        ]
        result.duplicate_status = "duplicate" if dup_errors else "none"
        return result

    def _existing_content_hashes(
        self,
        scope: UploadScope | None = None,
        conversation_id: str | None = None,
    ) -> set[str]:
        """Content hashes already present in the relevant scope.

        Knowledge-base uploads deduplicate globally; conversation uploads
        deduplicate only within their own conversation. Cross-conversation
        hashes never leak between scopes.
        """
        hashes: set[str] = set()

        if scope == UploadScope.KNOWLEDGE_BASE:
            for meta in self.storage.list_files():
                h = meta.get("content_hash")
                if h:
                    hashes.add(h)
            return hashes

        if scope == UploadScope.CONVERSATION and conversation_id:
            for attachment in self.lifecycle.list_attachments(conversation_id):
                if attachment.content_hash:
                    hashes.add(attachment.content_hash)
            return hashes

        # Unknown scope: be conservative, no cross-scope leakage
        return hashes

    def list_attachments(self, conversation_id: str) -> list[dict[str, Any]]:
        """List attachment metadata for a conversation."""
        return [
            self._attachment_to_dict(f)
            for f in self.lifecycle.list_attachments(conversation_id)
        ]

    def list_conversation_attachments(self, conversation_id: str) -> list[dict[str, Any]]:
        """Alias for list_attachments."""
        return self.list_attachments(conversation_id)

    def get_attachment(self, conversation_id: str, attachment_id: str) -> dict[str, Any] | None:
        """Get attachment metadata, verifying conversation isolation."""
        file = self.lifecycle.get_attachment(attachment_id)
        if file is None or file.conversation_id != conversation_id:
            return None
        if file.status in (UploadedFileStatus.EXPIRED, UploadedFileStatus.DELETED):
            return None
        return self._attachment_to_dict(file)

    def get_attachment_metadata(self, file_id: str) -> dict[str, Any] | None:
        """Get attachment metadata by file ID (internal use)."""
        file = self.lifecycle.get_attachment(file_id)
        if file is None:
            return None
        return self._attachment_to_dict(file)

    def delete_attachment(self, conversation_id: str, attachment_id: str) -> bool:
        """Delete an attachment, verifying conversation isolation."""
        file = self.lifecycle.get_attachment(attachment_id)
        if file is None or file.conversation_id != conversation_id:
            return False
        return self.lifecycle.delete_attachment(attachment_id)

    def delete_conversation_attachments(self, conversation_id: str) -> int:
        """Delete all attachments for a conversation. Returns count deleted."""
        return self.lifecycle.delete_conversation_attachments(conversation_id)

    def get_content(self, file_id: str) -> bytes | None:
        """Get raw content of a stored attachment."""
        return self.storage.retrieve(file_id)

    def _attachment_to_dict(self, file: UploadedFile) -> dict[str, Any]:
        """Convert an UploadedFile to a safe API dictionary."""
        return {
            "file_id": file.file_id,
            "original_filename": file.original_filename,
            "sanitized_filename": file.sanitized_filename,
            "content_type": file.content_type,
            "file_size": file.file_size,
            "content_hash": file.content_hash,
            "upload_timestamp": file.upload_timestamp.isoformat() if file.upload_timestamp else None,
            "storage_reference": file.storage_reference,
            "status": file.status.value,
            "validation_status": file.validation_status,
            "processing_status": file.processing_status.value,
            "conversation_id": file.conversation_id,
            "knowledge_base_id": file.knowledge_base_id,
            "expiry_time": file.expiry_time.isoformat() if file.expiry_time else None,
        }
