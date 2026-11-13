"""Local file storage backend for Prome RAG Plugins uploads."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loaders.contracts.exceptions import LoaderStorageError
from loaders.uploads.models import UploadedFile


class StorageBackend:
    """Local file-based storage backend for uploaded files.

    Uses unique internal IDs. Never uses uploaded filenames directly as
    storage paths. Supports atomic writes, content hashes, and explicit
    storage root configuration.
    """

    def __init__(self, storage_root: str | Path, *, create_root: bool = True):
        self.storage_root = Path(storage_root).resolve()
        if create_root and not self.storage_root.exists():
            self.storage_root.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, dict[str, Any]] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load the storage index from disk if it exists."""
        index_path = self.storage_root / ".upload_index.json"
        if index_path.exists():
            try:
                with open(index_path) as f:
                    self._index = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._index = {}

    def _save_index(self) -> None:
        """Persist the storage index to disk."""
        index_path = self.storage_root / ".upload_index.json"
        try:
            with open(index_path, "w") as f:
                json.dump(self._index, f, indent=2)
        except OSError:
            pass  # Best-effort persistence

    def store(self, file: UploadedFile) -> str:
        """Store a file in the local backend.

        Returns the internal storage path used.
        """
        if file.file_id in self._index:
            # Already stored; return existing path
            return self._index[file.file_id]["storage_path"]

        # Generate safe storage path
        storage_path = self._generate_storage_path(file.file_id)

        # Write file atomically
        try:
            # Write content to a temp path first, then rename
            temp_path = storage_path.with_suffix(".tmp")
            with open(temp_path, "wb") as f:
                # We need the actual content - file.content is bytes or str
                content = file.content
                if isinstance(content, str):
                    content = content.encode("utf-8")
                f.write(content)

            # Atomic rename
            final_path = storage_path.resolve()
            temp_path.rename(final_path)

            # Record file content hash for integrity check
            content_hash = self._compute_file_hash(final_path)

            # Update index
            self._index[file.file_id] = {
                "storage_path": str(final_path),
                "content_hash": content_hash,
                "file_size": file.file_size,
                "sanitized_filename": file.sanitized_filename,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
                "status": file.status.value,
            }
            self._save_index()

            return str(final_path)

        except OSError as e:
            raise LoaderStorageError(f"Failed to store file: {e}") from e

    def retrieve(self, file_id: str) -> bytes | None:
        """Retrieve file content by internal file ID.

        Returns the file content as bytes, or None if not found.
        """
        info = self._index.get(file_id)
        if info is None:
            return None

        storage_path = Path(info["storage_path"])
        if not storage_path.exists():
            return None

        try:
            return storage_path.read_bytes()
        except OSError:
            return None

    def get_metadata(self, file_id: str) -> dict[str, Any] | None:
        """Get metadata for a stored file."""
        info = self._index.get(file_id)
        if info is None:
            return None
        return {
            "file_id": file_id,
            "storage_path": info.get("storage_path"),
            "content_hash": info.get("content_hash"),
            "file_size": info.get("file_size"),
            "sanitized_filename": info.get("sanitized_filename"),
            "status": info.get("status"),
            "uploaded_at": info.get("uploaded_at"),
        }

    def delete(self, file_id: str) -> bool:
        """Delete a stored file.

        Returns True if deleted, False if not found.
        """
        info = self._index.get(file_id)
        if info is None:
            return False

        storage_path = Path(info["storage_path"])
        try:
            if storage_path.exists():
                storage_path.unlink()
            # Remove from index
            del self._index[file_id]
            self._save_index()
            return True
        except OSError:
            return False

    def list_files(self) -> list[dict[str, Any]]:
        """List all stored file metadata."""
        return list(self._index.values())

    def _generate_storage_path(self, file_id: str) -> Path:
        """Generate a safe storage path for a file ID.

        Never uses the original filename as part of the path.
        """
        # Use first 2 chars of file_id for directory nesting,
        # rest as filename. This prevents collisions and keeps
        # directories manageable.
        safe_id = "".join(ch for ch in file_id if ch.isalnum() or ch in "-_")
        if len(safe_id) < 2:
            safe_id = (safe_id + "00")[:2]
        prefix = safe_id[:2]
        suffix = safe_id[2:] or "file"
        path = self.storage_root / prefix / suffix
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _compute_file_hash(self, path: Path) -> str:
        """Compute SHA-256 hash of a file for integrity checking."""
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
        except OSError:
            pass
        return h.hexdigest()

    def check_integrity(self, file_id: str) -> bool:
        """Check that a stored file's hash matches its content."""
        info = self._index.get(file_id)
        if info is None:
            return False

        storage_path = Path(info["storage_path"])
        if not storage_path.exists():
            return False

        current_hash = self._compute_file_hash(storage_path)
        return current_hash == info.get("content_hash")
