"""Local filesystem file loader for Prome RAG Plugins."""

from __future__ import annotations

import hashlib
import mimetypes
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from loaders.contracts.exceptions import (
    PathTraversalError,
)
from loaders.contracts.models import (
    ContentType,
    FetchInfo,
    LastModified,
    LoadedResource,
    LoaderResult,
    SourceProvenance,
    SourceReference,
    SourceType,
)


class FileLoader:
    """Loader for explicit local file paths."""

    def __init__(
        self,
        *,
        permitted_roots: list[Path] | None = None,
        max_file_size: int = 100 * 1024 * 1024,  # 100 MiB
        supported_extensions: tuple[str, ...] | None = None,
    ):
        self.permitted_roots = [p.resolve() for p in (permitted_roots or [])]
        self.max_file_size = max_file_size
        self.supported_extensions = set(
            supported_extensions or ()
        )

    def load(self, source: str | Path) -> LoaderResult:
        """Load content from a local file path.

        Args:
            source: File path to load.

        Returns:
            A LoaderResult containing the loaded resource.
        """
        source_path = Path(source).resolve()

        # Check if path is within permitted roots
        if not self._is_path_permitted(source_path):
            raise PathTraversalError(
                f"Path {source_path} is not within permitted roots: {self.permitted_roots}"
            )

        # Check file exists and is readable
        if not source_path.exists():
            return LoaderResult().add_failure(
                f"File not found: {source_path}",
                SourceReference(
                    source_id=str(uuid.uuid4()),
                    source_type=SourceType.LOCAL_FILE,
                    original_uri_or_path=str(source_path),
                ),
            )

        if not os.access(source_path, os.R_OK):
            return LoaderResult().add_failure(
                f"File not readable: {source_path}",
                SourceReference(
                    source_id=str(uuid.uuid4()),
                    source_type=SourceType.LOCAL_FILE,
                    original_uri_or_path=str(source_path),
                ),
            )

        # Check file size
        try_file_size = source_path.stat().st_size
        if try_file_size > self.max_file_size:
            return LoaderResult().add_failure(
                f"File exceeds maximum size of {self.max_file_size} bytes",
                SourceReference(
                    source_id=str(uuid.uuid4()),
                    source_type=SourceType.LOCAL_FILE,
                    original_uri_or_path=str(source_path),
                ),
            )

        # Read content
        try:
            content = source_path.read_bytes()
        except OSError as e:
            return LoaderResult().add_failure(
                f"Failed to read file: {e}",
                SourceReference(
                    source_id=str(uuid.uuid4()),
                    source_type=SourceType.LOCAL_FILE,
                    original_uri_or_path=str(source_path),
                ),
            )

        # Detect content type
        content_type = self._detect_content_type(source_path, content)

        # Compute content hash
        content_hash = self._compute_hash(content)

        # Detect last modified
        stat = source_path.stat()
        last_modified = LastModified(
            timestamp=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            etag=None,
        )

        # Create source reference
        source_ref = SourceReference(
            source_id=str(uuid.uuid4()),
            source_type=SourceType.LOCAL_FILE,
            original_uri_or_path=str(source_path),
            filename=source_path.name,
            metadata={"permitted_roots": [str(r) for r in self.permitted_roots]},
        )

        # Create provenance
        provenance = SourceProvenance(
            fetched_from=str(source_path),
            fetched_at=datetime.now(timezone.utc),
            last_modified=last_modified,
            content_hash=content_hash,
        )

        # Create loaded resource
        resource = LoadedResource(
            reference=source_ref,
            content=content,
            content_type=content_type,
            filename=source_path.name,
            file_size=try_file_size,
            content_hash=content_hash,
            last_modified=last_modified,
            fetch_info=FetchInfo(fetch_timestamp=datetime.now(timezone.utc)),
            provenance=provenance,
        )

        return LoaderResult(resources=[resource])

    def _is_path_permitted(self, path: Path) -> bool:
        """Check if a path is within permitted roots."""
        if not self.permitted_roots:
            # If no roots configured, allow all (not recommended for production)
            return True

        for root in self.permitted_roots:
            try:
                path.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def _detect_content_type(self, path: Path, content: bytes) -> ContentType:
        """Detect the content type of a file."""

        # First check supported extensions
        if self.supported_extensions:
            ext = path.suffix.lower()
            if ext in self.supported_extensions:
                # Map extension to content type
                mime, _ = mimetypes.guess_type(str(path))
                if mime:
                    mime_type = mime.split("/")[0]
                    try:
                        return ContentType(mime_type)
                    except ValueError:
                        pass

        # Guess from MIME type
        mime, _ = mimetypes.guess_type(str(path))
        if mime:
            mime_type = mime.split("/")[0]
            try:
                return ContentType(mime_type)
            except ValueError:
                pass

        # Check content bytes for HTML/PDF/etc.
        if content.startswith(b"%PDF"):
            return ContentType.PDF
        if content.startswith((b"<!DOCTYPE html>", b"<html")):
            return ContentType.HTML
        if content.startswith((b"{", b"[")):
            return ContentType.JSON

        return ContentType.UNKNOWN

    @staticmethod
    def _compute_hash(content: bytes) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(content).hexdigest()
