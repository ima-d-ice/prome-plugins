"""ChatGPT-style file upload backend for Prome RAG Plugins."""

from loaders.uploads.lifecycle import AttachmentLifecycle, AttachmentStatus
from loaders.uploads.manager import UploadManager
from loaders.uploads.models import (
    DuplicateCheckResult,
    UploadedFile,
    UploadedFileStatus,
    UploadRequest,
    UploadResult,
    UploadScope,
)
from loaders.uploads.sessions import UploadSession, UploadSessionManager
from loaders.uploads.storage import StorageBackend
from loaders.uploads.validation import FileValidator

__all__ = [
    "AttachmentLifecycle",
    "AttachmentStatus",
    "DuplicateCheckResult",
    "FileValidator",
    "StorageBackend",
    "UploadManager",
    "UploadRequest",
    "UploadResult",
    "UploadScope",
    "UploadSession",
    "UploadSessionManager",
    "UploadedFile",
    "UploadedFileStatus",
]
