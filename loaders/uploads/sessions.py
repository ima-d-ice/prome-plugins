"""Upload session management for Prome RAG Plugins."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from loaders.uploads.models import UploadedFile, UploadScope


class UploadSession:
    """A single upload session tracking progress for a batch of files."""

    def __init__(self, session_id: str, scope: UploadScope, conversation_id: str | None = None):
        self.session_id = session_id
        self.scope = scope
        self.conversation_id = conversation_id
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at
        self.files: dict[str, UploadedFile] = {}  # file_id -> UploadedFile
        self.completed = False
        self.failed = False
        self.aborted = False
        self.error_message: str | None = None
        self.metadata: dict[str, Any] = {}

    def add_file(self, file: UploadedFile) -> None:
        """Add a file to the session."""
        self.files[file.file_id] = file
        self.updated_at = datetime.now(timezone.utc)

    def get_file(self, file_id: str) -> UploadedFile | None:
        """Get a file from the session."""
        return self.files.get(file_id)

    def remove_file(self, file_id: str) -> UploadedFile | None:
        """Remove a file from the session and return it."""
        return self.files.pop(file_id, None)

    def complete(self) -> None:
        """Mark the session as complete."""
        self.completed = True
        self.updated_at = datetime.now(timezone.utc)

    def fail(self, message: str) -> None:
        """Mark the session as failed."""
        self.failed = True
        self.aborted = True
        self.error_message = message
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the session to a dictionary."""
        return {
            "session_id": self.session_id,
            "scope": self.scope.value,
            "conversation_id": self.conversation_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed": self.completed,
            "failed": self.failed,
            "aborted": self.aborted,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "file_count": len(self.files),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UploadSession:
        """Deserialize a session from a dictionary."""
        session = cls(
            session_id=data["session_id"],
            scope=UploadScope(data["scope"]),
            conversation_id=data.get("conversation_id"),
        )
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.updated_at = datetime.fromisoformat(data["updated_at"])
        session.completed = data.get("completed", False)
        session.failed = data.get("failed", False)
        session.aborted = data.get("aborted", False)
        session.error_message = data.get("error_message")
        session.metadata = data.get("metadata", {})
        return session


class UploadSessionManager:
    """Manages upload sessions with persistence."""

    def __init__(self, storage_root: str | Path, *, session_ttl_hours: int = 168):
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, UploadSession] = {}
        self._session_ttl = timedelta(hours=session_ttl_hours)
        self._index_path = self.storage_root / ".upload_sessions.json"
        self._load_sessions()

    def _load_sessions(self) -> None:
        """Load sessions from disk."""
        if self._index_path.exists():
            try:
                with open(self._index_path, "r") as f:
                    data = json.load(f)
                for session_data in data.get("sessions", []):
                    session = UploadSession.from_dict(session_data)
                    self._sessions[session.session_id] = session
            except (json.JSONDecodeError, OSError):
                pass

    def _save_sessions(self) -> None:
        """Persist sessions to disk."""
        try:
            with open(self._index_path, "w") as f:
                json.dump(
                    {"sessions": [s.to_dict() for s in self._sessions.values()]},
                    f,
                    indent=2,
                )
        except OSError:
            pass  # Best-effort persistence

    def create_session(self, scope: UploadScope, conversation_id: str | None = None) -> UploadSession:
        """Create a new upload session."""
        session_id = str(uuid.uuid4())
        session = UploadSession(session_id=session_id, scope=scope, conversation_id=conversation_id)
        self._sessions[session_id] = session
        self._save_sessions()
        return session

    def get_session(self, session_id: str) -> UploadSession | None:
        """Get a session by ID, cleaning up expired ones."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        # Check TTL
        if datetime.now(timezone.utc) - session.created_at > self._session_ttl:
            self._remove_session(session_id)
            return None
        return session

    def remove_session(self, session_id: str) -> bool:
        """Remove a session."""
        if session_id in self._sessions:
            self._remove_session(session_id)
            return True
        return False

    def _remove_session(self, session_id: str) -> None:
        """Internal session removal."""
        self._sessions.pop(session_id, None)
        self._save_sessions()

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all active sessions."""
        return [s.to_dict() for s in self._sessions.values()]

    def cleanup_expired(self) -> int:
        """Remove expired sessions. Returns count removed."""
        now = datetime.now(timezone.utc)
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s.created_at > self._session_ttl
        ]
        for sid in expired:
            self._remove_session(sid)
        return len(expired)