"""Local filesystem directory loader for Prome RAG Plugins."""

from __future__ import annotations

import hashlib
import mimetypes
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

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
from loaders.filesystem.filters import (
    ExclusionFilter,
    ExtensionFilter,
    HiddenFilePolicy,
    SymlinkPolicy,
)

# Sensitive paths excluded by default (match on name).
DEFAULT_EXCLUDED_NAMES = frozenset({
    ".env",
    ".env.local",
    ".env.production",
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "build",
    "dist",
    ".idea",
    ".vscode",
    ".eggs",
    ".coverage",
    ".ssh",
    ".aws",
    "private_key",
    "id_rsa",
    "secrets.yaml",
    "credentials.json",
    "config.local.yaml",
})


def is_sensitive_name(name: str) -> bool:
    """Check if a file or directory name is sensitive."""
    lowered = name.lower()
    if lowered in DEFAULT_EXCLUDED_NAMES:
        return True
    # Generic private key patterns
    return lowered.startswith(".env") or lowered.endswith((".pem", ".key"))


class DirectoryLoader:
    """Loader for local directories with recursive discovery and filtering.

    Supports:
    - Recursive and non-recursive file discovery.
    - Extension allowlists and exclusion patterns.
    - Hidden-file policy.
    - Symlink policy.
    - Maximum files and maximum size limits.
    - Deterministic ordering.

    Sensitive paths (.env, private keys, .git, virtual environments,
    Python caches, build outputs) are excluded by default.
    """

    def __init__(
        self,
        *,
        permitted_roots: list[Path] | None = None,
        max_files: int = 500,
        max_total_size: int = 500 * 1024 * 1024,  # 500 MiB
        recursive: bool = True,
        extension_filter: ExtensionFilter | None = None,
        exclusion_filter: ExclusionFilter | None = None,
        hidden_file_policy: HiddenFilePolicy = HiddenFilePolicy.EXCLUDE,
        symlink_policy: SymlinkPolicy = SymlinkPolicy.REJECT,
        max_symlink_depth: int = 8,
        default_exclude_sensitive: bool = True,
    ):
        self.permitted_roots = [p.resolve() for p in (permitted_roots or [])]
        self.max_files = max_files
        self.max_total_size = max_total_size
        self.recursive = recursive
        self.extension_filter = extension_filter or ExtensionFilter(set())
        self.exclusion_filter = exclusion_filter or ExclusionFilter(set())
        self.hidden_file_policy = hidden_file_policy
        self.symlink_policy = symlink_policy
        self.max_symlink_depth = max_symlink_depth
        self.default_exclude_sensitive = default_exclude_sensitive

    def load(self, source: str | Path) -> LoaderResult:
        """Load all matching files from a directory."""
        dir_path = Path(source).resolve()
        result = LoaderResult()

        if not self._is_path_permitted(dir_path):
            result.add_failure(
                f"Directory {dir_path} is not within permitted roots: {[str(r) for r in self.permitted_roots] or '(none configured)'}",
                SourceReference(
                    source_id=str(uuid.uuid4()),
                    source_type=SourceType.LOCAL_DIRECTORY,
                    original_uri_or_path=str(dir_path),
                ),
            )
            return result

        if not dir_path.exists() or not dir_path.is_dir():
            result.add_failure(
                f"Directory not found: {dir_path}",
                SourceReference(
                    source_id=str(uuid.uuid4()),
                    source_type=SourceType.LOCAL_DIRECTORY,
                    original_uri_or_path=str(dir_path),
                ),
            )
            return result

        candidates = self._discover(dir_path)

        total_size = 0
        files_found = 0

        for entry_path in candidates:
            rel = self._relative(dir_path, entry_path)
            try:
                content = entry_path.read_bytes()
            except OSError as e:
                result.add_failure(f"Failed to read file {rel}: {e}")
                continue

            total_size += len(content)
            files_found += 1
            if files_found > self.max_files:
                result.add_failure(f"Maximum file count ({self.max_files}) exceeded")
                break
            if total_size > self.max_total_size:
                result.add_failure(f"Maximum total size ({self.max_total_size}) exceeded")
                break

            content_type = self._detect_content_type(entry_path, content)
            content_hash = hashlib.sha256(content).hexdigest()
            stat = entry_path.stat()
            last_modified = LastModified(
                timestamp=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            )

            source_ref = SourceReference(
                source_id=str(uuid.uuid4()),
                source_type=SourceType.LOCAL_DIRECTORY,
                original_uri_or_path=str(dir_path),
                filename=entry_path.name,
                metadata={
                    "recursive": self.recursive,
                    "relative_path": rel,
                    "permitted_roots": [str(r) for r in self.permitted_roots],
                },
            )
            provenance = SourceProvenance(
                fetched_from=str(entry_path),
                fetched_at=datetime.now(timezone.utc),
                last_modified=last_modified,
                content_hash=content_hash,
            )
            resource = LoadedResource(
                reference=source_ref,
                content=content,
                content_type=content_type,
                filename=entry_path.name,
                file_size=len(content),
                content_hash=content_hash,
                last_modified=last_modified,
                fetch_info=FetchInfo(fetch_timestamp=datetime.now(timezone.utc)),
                provenance=provenance,
            )
            result.resources.append(resource)

        return result

    def _discover(self, dir_path: Path) -> list[Path]:
        """Discover candidate files in deterministic order."""
        if self.recursive:
            candidates: list[Path] = []
            for root, dirs, files in os.walk(dir_path, followlinks=False):
                # Prune in-place for deterministic, safe traversal
                dirs.sort()
                dirs[:] = [
                    d
                    for d in dirs
                    if not self._is_excluded_dir(Path(root) / d)
                ]
                for name in sorted(files):
                    p = Path(root) / name
                    if self._accept(p):
                        candidates.append(p)
            return candidates

        candidates = []
        try:
            entries = sorted(os.listdir(dir_path))
        except OSError:
            return []
        for name in entries:
            p = dir_path / name
            if p.is_dir() and not p.is_symlink():
                continue
            if self._accept(p):
                candidates.append(p)
        return candidates

    def _relative(self, root: Path, path: Path) -> str:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return path.as_posix()

    def _is_excluded_dir(self, path: Path) -> bool:
        name = path.name
        if self.default_exclude_sensitive and is_sensitive_name(name):
            return True
        if self.hidden_file_policy == HiddenFilePolicy.EXCLUDE and name.startswith("."):
            return True
        if self.exclusion_filter.is_excluded(name):
            return True
        return not self._is_path_permitted(path)

    def _accept(self, path: Path) -> bool:
        name = path.name
        if self.default_exclude_sensitive and is_sensitive_name(name):
            return False
        if self.hidden_file_policy == HiddenFilePolicy.EXCLUDE and name.startswith("."):
            return False
        if self.exclusion_filter.is_excluded(name):
            return False

        # Symlink policy
        if path.is_symlink():
            if self.symlink_policy == SymlinkPolicy.REJECT:
                return False
            try:
                resolved = path.resolve()
            except OSError:
                return False
            if not self._is_path_permitted(resolved):
                return False
            if resolved.is_dir():
                return False
            path = resolved

        if not path.is_file():
            return False

        if not self._is_path_permitted(path):
            return False

        return self.extension_filter.accept(path.suffix)

    def _is_path_permitted(self, path: Path) -> bool:
        """Check if a path is within permitted roots."""
        if not self.permitted_roots:
            return True
        for root in self.permitted_roots:
            try:
                path.resolve().relative_to(root)
                return True
            except ValueError:
                continue
        return False

    @staticmethod
    def _detect_content_type(path: Path, content: bytes) -> ContentType:
        """Detect the content type of a file."""
        mime, _ = mimetypes.guess_type(str(path))
        if mime:
            mime_type = mime.split("/")[0]
            try:
                return ContentType(mime_type)
            except ValueError:
                pass

        if content.startswith(b"%PDF"):
            return ContentType.PDF
        if content.startswith((b"<!DOCTYPE html>", b"<html")):
            return ContentType.HTML
        if content.startswith((b"{", b"[")):
            return ContentType.JSON
        return ContentType.UNKNOWN
