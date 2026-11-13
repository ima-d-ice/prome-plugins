"""Prome RAG Plugins — Loader ecosystem.

Modular loaders for fetching, receiving, and managing source content.
See individual submodules for detailed documentation.
"""

from loaders.contracts.capabilities import LoaderCapabilities
from loaders.contracts.exceptions import (
    LoaderDuplicateError,
    LoaderError,
    LoaderStorageError,
    LoaderValidationError,
)
from loaders.contracts.models import (
    ContentType,
    LoadedResource,
    LoadedResourceStatus,
    LoaderResult,
    SourceReference,
    SourceType,
)
from loaders.contracts.protocols import Loader
from loaders.uploads.manager import UploadManager
from loaders.uploads.models import (
    UploadedFile,
    UploadedFileStatus,
    UploadRequest,
    UploadResult,
    UploadScope,
)

__all__ = [
    "ContentType",
    "LoadedResource",
    "LoadedResourceStatus",
    "Loader",
    "LoaderCapabilities",
    "LoaderDuplicateError",
    "LoaderError",
    "LoaderResult",
    "LoaderStorageError",
    "LoaderValidationError",
    "SourceReference",
    "SourceType",
    "UploadManager",
    "UploadRequest",
    "UploadResult",
    "UploadScope",
    "UploadedFile",
    "UploadedFileStatus",
]
