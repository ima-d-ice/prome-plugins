"""Shared loader exceptions."""

from __future__ import annotations


class LoaderError(Exception):
    """Base class for all loader-related errors."""


class FileValidationError(LoaderError):
    """Raised when a file fails validation."""


class LoaderValidationError(LoaderError):
    """Raised when loader validation fails."""


class LoaderStorageError(LoaderError):
    """Raised when loader storage operations fail."""


class LoaderDuplicateError(LoaderError):
    """Raised when a duplicate source is detected."""


class LoaderPermissionError(LoaderError):
    """Raised when loader operation violates permissions."""


class LoaderNotSupportedError(LoaderError):
    """Raised when a loader does not support a given source type."""


class LoaderConfigurationError(LoaderError):
    """Raised when loader configuration is invalid."""


class SSRFError(LoaderError):
    """Raised when a URL request is blocked by SSRF protection."""


class PathTraversalError(LoaderValidationError):
    """Raised when a path traversal attempt is detected."""


class SymlinkEscapeError(LoaderValidationError):
    """Raised when a symlink escape is detected."""


class DuplicateContentHashError(LoaderDuplicateError):
    """Raised when content hash matches an existing source."""


class EmptyContentError(LoaderValidationError):
    """Raised when uploaded content is empty."""


class UnsupportedFormatError(LoaderValidationError):
    """Raised when a file format is not supported."""


class MaxFileSizeExceededError(LoaderValidationError):
    """Raised when uploaded file exceeds maximum size."""


class MaxFilesPerBatchExceededError(LoaderValidationError):
    """Raised when files per batch exceeds maximum."""


class MaxBatchSizeExceededError(LoaderValidationError):
    """Raised when total batch size exceeds maximum."""