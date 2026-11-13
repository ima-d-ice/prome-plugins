"""Upload service data models for Prome RAG Plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class UploadScope(str, Enum):
    """Scope of an upload."""

    CONVERSATION = "conversation"
    KNOWLEDGE_BASE = "knowledge_base"


class UploadedFileStatus(str, Enum):
    """Processing status of an uploaded file."""

    RECEIVED = "received"
    VALIDATING = "validating"
    STORED = "stored"
    READY_FOR_INGESTION = "ready_for_ingestion"
    PROCESSING = "processing"
    AVAILABLE = "available"
    FAILED = "failed"
    EXPIRED = "expired"
    DELETED = "deleted"


@dataclass
class UploadedFile:
    """A file received via the upload API."""

    file_id: str
    original_filename: str
    sanitized_filename: str
    content_type: str
    file_size: int
    content_hash: str
    upload_timestamp: datetime
    storage_reference: str
    status: UploadedFileStatus
    content: bytes | str = b""
    validation_status: str = "pending"
    processing_status: UploadedFileStatus = UploadedFileStatus.RECEIVED
    metadata: dict[str, Any] = field(default_factory=dict)
    expiry_time: datetime | None = None
    conversation_id: str | None = None
    knowledge_base_id: str | None = None


@dataclass
class UploadRequest:
    """A request to upload one or more files."""

    request_id: str
    conversation_id: str | None = None
    knowledge_base_id: str | None = None
    tenant_id: str | None = None
    scope: UploadScope = UploadScope.CONVERSATION
    files: list[UploadedFile] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    expiry_policy: dict[str, Any] | None = None
    atomic: bool = False


@dataclass
class UploadResult:
    """Result of an upload batch operation."""

    batch_id: str
    scope: UploadScope
    successful: list[UploadedFile] = field(default_factory=list)
    rejected: list[UploadedFile] = field(default_factory=list)
    per_file_errors: dict[str, str] = field(default_factory=dict)
    duplicate_status: str = "none"
    diagnostics: list[str] = field(default_factory=list)


@dataclass
class DuplicateCheckResult:
    """Result of duplicate content detection."""

    is_duplicate: bool
    existing_file_id: str | None = None
    confidence: float = 1.0
