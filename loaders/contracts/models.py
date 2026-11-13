"""Shared loader data models for Prome RAG Plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class SourceType(str, Enum):
    """Type of source origin."""

    LOCAL_FILE = "local_file"
    LOCAL_DIRECTORY = "local_directory"
    URL = "url"
    CRAWLED = "crawled"
    UPLOAD = "upload"


class ContentType(str, Enum):
    """Detected or inferred content type."""

    TEXT = "text"
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    MARKDOWN = "markdown"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    CODE = "code"
    UNKNOWN = "unknown"


class LoaderResultStatus(str, Enum):
    """Result status for a loader operation."""

    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


class LoadedResourceStatus(str, Enum):
    """Status of a loaded resource."""

    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class SourceReference:
    """Stable reference identifying a source.

    A SourceReference is immutable and uniquely identifies a source
    across loader operations. It is the primary key for tracking
    provenance and deduplication.
    """

    source_id: str
    source_type: SourceType
    original_uri_or_path: str
    filename: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tenant_id: str | None = None
    owner_id: str | None = None


@dataclass
class LastModified:
    """Last-modified timestamp information."""

    timestamp: datetime
    etag: str | None = None


@dataclass
class ETag:
    """Entity tag for conditional fetching."""

    tag: str
    weak: bool = False


@dataclass
class FetchInfo:
    """Information about a fetch operation."""

    fetch_timestamp: datetime
    status_code: int | None = None
    redirect_count: int = 0
    final_url: str | None = None


@dataclass
class SourceProvenance:
    """Provenance tracking for a source."""

    fetched_from: str | None = None
    fetched_at: datetime | None = None
    last_modified: LastModified | None = None
    content_hash: str | None = None
    etag: ETag | None = None


@dataclass
class LoadedResource:
    """A resource loaded from a source.

    Stops at providing a validated, retrievable source resource.
    Does not perform parsing, chunking, embedding or vector indexing.
    """

    reference: SourceReference
    content: bytes | str
    content_type: ContentType
    filename: str | None = None
    file_size: int = 0
    content_hash: str | None = None
    last_modified: LastModified | None = None
    fetch_info: FetchInfo | None = None
    provenance: SourceProvenance = field(default_factory=SourceProvenance)
    status: LoadedResourceStatus = LoadedResourceStatus.SUCCESS

    @property
    def is_text(self) -> bool:
        return self.content_type in (
            ContentType.TEXT,
            ContentType.MARKDOWN,
            ContentType.JSON,
            ContentType.HTML,
        )

    @property
    def is_binary(self) -> bool:
        return self.content_type in (
            ContentType.IMAGE,
            ContentType.AUDIO,
            ContentType.VIDEO,
        )


@dataclass
class LoaderFailure:
    """A failed load operation."""

    message: str
    reference: SourceReference | None = None
    error_code: str | None = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class LoaderSkip:
    """A skipped load operation."""

    message: str
    reference: SourceReference | None = None
    reason: str = "unspecified"
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class LoaderResult:
    """Result of a loader operation containing resources and diagnostics."""

    resources: list[LoadedResource] = field(default_factory=list)
    failures: list[LoaderFailure] = field(default_factory=list)
    skips: list[LoaderSkip] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return len(self.resources)

    @property
    def failure_count(self) -> int:
        return len(self.failures)

    @property
    def skip_count(self) -> int:
        return len(self.skips)

    def add_resource(self, resource: LoadedResource) -> LoaderResult:
        self.resources.append(resource)
        return self

    def add_failure(
        self,
        message: str,
        resource: SourceReference | None = None,
        error_code: str | None = None,
    ) -> LoaderResult:
        failure = LoaderFailure(
            message=message, reference=resource, error_code=error_code
        )
        self.failures.append(failure)
        return self

    def add_skip(
        self,
        message: str,
        resource: SourceReference | None = None,
        reason: str = "unspecified",
    ) -> LoaderResult:
        skip = LoaderSkip(message=message, reference=resource, reason=reason)
        self.skips.append(skip)
        return self
