"""File upload validation for Prome RAG Plugins."""

from __future__ import annotations

import re
from collections.abc import Sequence

from loaders.contracts.exceptions import (
    EmptyContentError,
    FileValidationError,
    MaxFileSizeExceededError,
    UnsupportedFormatError,
)
from loaders.uploads.models import UploadedFile, UploadScope

DEFAULT_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MiB
DEFAULT_MAX_FILES_PER_BATCH = 20
DEFAULT_MAX_TOTAL_BATCH_SIZE = 500 * 1024 * 1024  # 500 MiB
DEFAULT_MAX_CONVERSATION_ATTACHMENTS = 50
DEFAULT_ALLOWED_EXTENSIONS: tuple[str, ...] = (
    ".txt",
    ".md",
    ".pdf",
    ".csv",
    ".json",
    ".xml",
    ".html",
    ".htm",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".toml",
    ".yaml",
    ".yml",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".sh",
    ".cfg",
    ".ini",
    ".conf",
    ".dockerfile",
)
DEFAULT_ALLOWED_MIME_TYPES: tuple[str, ...] = (
    "text/plain",
    "text/markdown",
    "text/csv",
    "application/json",
    "application/xml",
    "application/xml+dtd",
    "text/html",
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "audio/mpeg",
    "audio/wav",
    "audio/ogg",
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

MAGIC_BYTES: dict[str, list[bytes]] = {
    ".pdf": [b"%PDF"],
    ".png": [b"\x89PNG\r\n\x1a\n"],
    ".jpg": [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
    ".gif": [b"GIF87a", b"GIF89a"],
    ".webp": [b"RIFF"],
    ".zip": [b"PK\x03\x04", b"PK\x05\x06"],
}


def detect_magic(extension: str, content: bytes) -> bool:
    """Check if content matches known magic bytes for the extension.

    Returns True if the content looks like the expected format, or if the
    extension has no known magic bytes (text formats have no signatures).
    """
    signatures = MAGIC_BYTES.get(extension)
    if not signatures:
        return True
    return any(content.startswith(sig) for sig in signatures)


def sniff_content_kind(content: bytes) -> str:
    """Lightweight content sniffing from magic bytes."""
    if content.startswith(b"%PDF"):
        return "pdf"
    if content.startswith(b"\x89PNG"):
        return "png"
    if content.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if content.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if content.startswith((b"PK\x03\x04", b"PK\x05\x06")):
        return "zip"
    if content.startswith((b"<!DOCTYPE html", b"<!doctype html")):
        return "html"
    if content.startswith((b"<html", b"<!--")):
        return "html"
    if content.startswith((b"{", b"[")):
        return "json"
    return "unknown"


class FileValidator:
    """Validates uploaded files against configurable policies."""

    def __init__(
        self,
        *,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
        max_files_per_batch: int = DEFAULT_MAX_FILES_PER_BATCH,
        max_total_batch_size: int = DEFAULT_MAX_TOTAL_BATCH_SIZE,
        max_conversation_attachments: int = DEFAULT_MAX_CONVERSATION_ATTACHMENTS,
        allowed_extensions: Sequence[str] | None = None,
        allowed_mime_types: Sequence[str] | None = None,
    ):
        self.max_file_size = max_file_size
        self.max_files_per_batch = max_files_per_batch
        self.max_total_batch_size = max_total_batch_size
        self.max_conversation_attachments = max_conversation_attachments
        self.allowed_extensions = {
            ext.lower() for ext in (allowed_extensions or DEFAULT_ALLOWED_EXTENSIONS)
        }
        self.allowed_mime_types = set(
            allowed_mime_types or DEFAULT_ALLOWED_MIME_TYPES
        )

    def validate_request(self, request: UploadRequest) -> tuple[bool, list[str]]:  # noqa: F821
        """Validate an entire upload request.

        Returns (is_valid, errors).
        """
        errors: list[str] = []

        if request.scope == UploadScope.CONVERSATION and not request.conversation_id:
            errors.append("Conversation ID is required for conversation-scoped uploads")

        if request.scope == UploadScope.KNOWLEDGE_BASE and not request.knowledge_base_id:
            errors.append("Knowledge base ID is required for knowledge-base uploads")

        if len(request.files) > self.max_files_per_batch:
            errors.append(
                f"Files per batch exceeds maximum of {self.max_files_per_batch}"
            )

        total_size = sum(f.file_size for f in request.files)
        if total_size > self.max_total_batch_size:
            errors.append(
                f"Total batch size exceeds maximum of {self.max_total_batch_size} bytes"
            )

        return len(errors) == 0, errors

    def validate_file(self, file: UploadedFile) -> tuple[bool, list[str]]:
        """Validate a single uploaded file.

        Checks size, emptiness, extension allowlist and content magic bytes.
        Never trusts client-provided MIME type alone.
        """
        errors: list[str] = []

        if file.file_size > self.max_file_size:
            raise MaxFileSizeExceededError(
                f"File size {file.file_size} exceeds maximum {self.max_file_size}"
            )

        if file.file_size == 0:
            raise EmptyContentError("File has no content")

        name_for_ext = file.sanitized_filename or file.original_filename

        ext = self.extract_extension(name_for_ext)
        if ext and ext not in self.allowed_extensions:
            raise UnsupportedFormatError(
                f"File extension '{ext}' is not allowed"
            )

        content = file.content
        if isinstance(content, str):
            content = content.encode("utf-8")
        if ext and not detect_magic(ext, content):
            raise FileValidationError(
                f"Content does not match expected format for '{ext}' "
                "(magic-byte mismatch or corrupt container)"
            )

        if not file.content_hash:
            raise FileValidationError("File content hash could not be computed")

        return len(errors) == 0, errors

    def validate_mime_type(self, mime_type: str) -> bool:
        """Check that a MIME type is allowed."""
        normalized = mime_type.split(";")[0].strip().lower()
        return normalized in self.allowed_mime_types

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize a filename for safe storage.

        Removes path separators, null bytes, traversal sequences and
        control characters. Never returns an empty or traversal-laden name.
        """
        name = filename.strip()

        # Control characters out
        name = "".join(ch for ch in name if ch.isprintable())

        # Null bytes out
        name = name.replace("\x00", "")

        # Path separators to underscores
        name = name.replace("/", "_").replace("\\", "_")

        # Remove parent-directory traversal segments
        segments = [seg for seg in name.split("_") if seg not in ("", ".", "..")]
        name = "_".join(segments).replace("..", "_")

        # Collapse underscores
        name = re.sub(r"_+", "_", name).strip("_")

        if not name:
            name = "upload"

        return name

    def validate_sanitized_filename(
        self, filename: str
    ) -> tuple[bool, str, list[str]]:
        """Sanitize and validate a filename.

        Returns (is_valid, sanitized_name, warnings).
        """
        warnings: list[str] = []
        sanitized = self.sanitize_filename(filename)

        if ".." in sanitized or sanitized.startswith(("/", "\\")):
            warnings.append(f"Path traversal detected in filename: {filename}")
            return False, sanitized, warnings

        if sanitized.startswith("."):
            warnings.append(f"Filename starts with '.': {filename}")

        return True, sanitized, warnings

    @staticmethod
    def extract_extension(filename: str) -> str:
        """Extract a lowercased file extension (including leading dot)."""
        if "." in filename:
            ext = filename.rsplit(".", 1)[-1]
            if ext:
                return f".{ext.lower()}"
        return ""

    def check_duplicate_content(
        self, file: UploadedFile, existing_files: dict[str, UploadedFile]
    ) -> tuple[bool, str | None]:
        """Check if file content hash matches an existing file.

        Returns (is_duplicate, existing_file_id).
        """
        for existing_id, existing in existing_files.items():
            if existing.content_hash == file.content_hash:
                return True, existing_id
        return False, None
