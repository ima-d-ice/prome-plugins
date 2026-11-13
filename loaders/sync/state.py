"""Synchronization state management for Prome RAG Plugins."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loaders.contracts.exceptions import LoaderError
from loaders.contracts.models import (
    SourceType,
)


def _loads_json(value, default):
    """Parse a JSON column value, tolerating NULLs and legacy raw values."""
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        parsed = json.loads(value)
        return parsed if parsed is not None else default
    except (json.JSONDecodeError, TypeError):
        return default


class SyncState:
    """Tracks synchronization state for a source.

    Uses stable source IDs, content hashes, ETag, and Last-Modified
    to detect changes between synchronizations.
    """

    def __init__(self, source_id: str, source_type: SourceType):
        self.source_id = source_id
        self.source_type = source_type
        self.last_sync: datetime | None = None
        self.operation: str = "initial"  # initial, sync, reconcile
        # Content identification
        self.content_hash: str | None = None
        self.etag: str | None = None
        self.last_modified: datetime | None = None
        # Page/URL tracking for directory/website sources
        self.tracked_urls: dict[str, dict[str, Any]] = {}
        # Change tracking
        self.new_files: set[str] = set()
        self.modified_files: set[str] = set()
        self.unchanged_files: set[str] = set()
        self.deleted_files: set[str] = set()
        # Metadata
        self.metadata: dict[str, Any] = {}

    def to_dict(self) -> dict[str, Any]:
        """Convert state to serializable dictionary."""
        return {
            "source_id": self.source_id,
            "source_type": self.source_type.value,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "operation": self.operation,
            "content_hash": self.content_hash,
            "etag": self.etag,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "tracked_urls": self.tracked_urls,
            "new_files": list(self.new_files),
            "modified_files": list(self.modified_files),
            "unchanged_files": list(self.unchanged_files),
            "deleted_files": list(self.deleted_files),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SyncState:
        """Create state from dictionary (reading SQLite row data)."""
        state = cls(source_id=data["source_id"], source_type=SourceType(data["source_type"]))
        state.last_sync = datetime.fromisoformat(data["last_sync"]) if data.get("last_sync") else None
        state.operation = data.get("operation", "initial")
        state.content_hash = data.get("content_hash")
        state.etag = data.get("etag")
        state.last_modified = datetime.fromisoformat(data["last_modified"]) if data.get("last_modified") else None
        state.tracked_urls = _loads_json(data.get("tracked_urls"), {})
        state.new_files = set(_loads_json(data.get("new_files"), []))
        state.modified_files = set(_loads_json(data.get("modified_files"), []))
        state.unchanged_files = set(_loads_json(data.get("unchanged_files"), []))
        state.deleted_files = set(_loads_json(data.get("deleted_files"), []))
        state.metadata = _loads_json(data.get("metadata"), {})
        return state


class SyncStatePersister:
    """Persists sync state using SQLite where suitable.

    Supports:
    - Atomic updates.
    - Safe reopening.
    - Change reports.
    - Explicit deletion events.
    - Reconciliation after interrupted synchronization.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database."""
        try:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sync_state (
                    source_id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    last_sync TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    content_hash TEXT,
                    etag TEXT,
                    last_modified TEXT,
                    tracked_urls TEXT,
                    new_files TEXT,
                    modified_files TEXT,
                    unchanged_files TEXT,
                    deleted_files TEXT,
                    metadata TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._conn.commit()
        except OSError as e:
            raise LoaderError(f"Failed to initialize sync state database: {e}") from e

    def get_state(self, source_id: str) -> SyncState | None:
        """Get sync state for a source ID."""
        try:
            cursor = self._conn.execute(
                "SELECT * FROM sync_state WHERE source_id = ?",
                (source_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            # We need to reconstruct from columns; use named approach
            cols = [desc[0] for desc in cursor.description]
            data = dict(zip(cols, row))
            return SyncState.from_dict(data)
        except sqlite3.Error as e:
            raise LoaderError(f"Failed to get sync state: {e}") from e

    def save_state(self, state: SyncState) -> None:
        """Save sync state atomically."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            last_sync = state.last_sync.isoformat() if state.last_sync else now
            self._conn.execute(
                """
                INSERT INTO sync_state (
                    source_id, source_type, last_sync, operation,
                    content_hash, etag, last_modified, tracked_urls,
                    new_files, modified_files, unchanged_files, deleted_files,
                    metadata, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    last_sync = excluded.last_sync,
                    operation = excluded.operation,
                    content_hash = excluded.content_hash,
                    etag = excluded.etag,
                    last_modified = excluded.last_modified,
                    tracked_urls = excluded.tracked_urls,
                    new_files = excluded.new_files,
                    modified_files = excluded.modified_files,
                    unchanged_files = excluded.unchanged_files,
                    deleted_files = excluded.deleted_files,
                    metadata = excluded.metadata,
                    updated_at = excluded.updated_at
                """,
                (
                    state.source_id,
                    state.source_type.value,
                    last_sync,
                    state.operation,
                    state.content_hash,
                    state.etag,
                    state.last_modified.isoformat() if state.last_modified else None,
                    json.dumps(state.tracked_urls) if state.tracked_urls else None,
                    json.dumps(list(state.new_files)) if state.new_files else None,
                    json.dumps(list(state.modified_files)) if state.modified_files else None,
                    json.dumps(list(state.unchanged_files)) if state.unchanged_files else None,
                    json.dumps(list(state.deleted_files)) if state.deleted_files else None,
                    json.dumps(state.metadata) if state.metadata else None,
                    now,
                ),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            self._conn.rollback()
            raise LoaderError(f"Failed to save sync state: {e}") from e

    def delete_state(self, source_id: str) -> None:
        """Delete sync state for a source."""
        try:
            self._conn.execute(
                "DELETE FROM sync_state WHERE source_id = ?",
                (source_id,),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            raise LoaderError(f"Failed to delete sync state: {e}") from e

    def list_sources(self) -> list[dict[str, Any]]:
        """List all tracked source IDs."""
        try:
            cursor = self._conn.execute(
                "SELECT source_id, source_type, last_sync, operation FROM sync_state"
            )
            return [dict(zip([desc[0] for desc in cursor], row)) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            raise LoaderError(f"Failed to list sources: {e}") from e