"""Reconciliation for Prome RAG Plugins incremental sync."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from loaders.contracts.models import (
    LoadedResource,
    SourceType,
)
from loaders.sync.change_detector import ChangeDetector
from loaders.sync.state import SyncState, SyncStatePersister


class Reconciler:
    """Reconciles synchronization state after interrupted runs or configuration changes.

    Supports:
    - Atomic updates.
    - Safe reopening.
    - Change reports.
    - Explicit deletion events.
    - Full reconciliation after interrupted synchronization.
    """

    def __init__(self, persister: SyncStatePersister, change_detector: ChangeDetector | None = None):
        self.persister = persister
        self.change_detector = change_detector or ChangeDetector()

    def reconcile(
        self,
        source_id: str,
        source_type: SourceType,
        current_resources: list[LoadedResource],
        previous_state: SyncState | None = None,
    ) -> dict[str, Any]:
        """Full reconciliation for a source.

        Returns dict with change report and updated state.
        """
        # Get or create previous state
        if previous_state is None:
            previous_state = self.persister.get_state(source_id)

        # If no previous state, this is an initial sync
        if previous_state is None:
            return self._initial_reconcile(source_id, source_type, current_resources)

        # Detect changes
        detection = self.change_detector.detect_changes(previous_state, current_resources)

        # Build updated state
        updated_state = self.change_detector.build_updated_state(
            previous_state, detection, current_resources
        )

        # Persist the updated state
        self.persister.save_state(updated_state)

        return {
            "change_report": detection,
            "updated_state": updated_state,
            "reconciled": True,
            "note": "Reconciliation completed successfully",
        }

    def _initial_reconcile(
        self,
        source_id: str,
        source_type: SourceType,
        current_resources: list[LoadedResource],
    ) -> dict[str, Any]:
        """Handle initial synchronization (no previous state)."""
        updated = SyncState(source_id=source_id, source_type=source_type)
        updated.last_sync = datetime.now(timezone.utc)
        updated.operation = "initial"

        # Set content hash from first resource
        if current_resources:
            updated.content_hash = current_resources[0].content_hash
            updated.etag = current_resources[0].provenance.etag.tag if current_resources[0].provenance.etag else None
            updated.last_modified = current_resources[0].provenance.last_modified.timestamp if current_resources[0].provenance.last_modified else None

        # Track all current resources as "new"
        updated.new_files = {r.reference.source_id for r in current_resources}
        updated.tracked_urls = {}
        for r in current_resources:
            updated.tracked_urls[r.reference.source_id] = {
                "content_hash": r.content_hash,
                "etag": r.provenance.etag.tag if r.provenance.etag else None,
                "last_modified": r.provenance.last_modified.timestamp.isoformat() if r.provenance.last_modified else None,
            }

        # Persist
        self.persister.save_state(updated)

        return {
            "change_report": {
                "new": list(updated.new_files),
                "modified": [],
                "unchanged": [],
                "deleted": [],
                "changed_urls": [],
            },
            "updated_state": updated,
            "reconciled": True,
            "note": "Initial synchronization completed",
        }

    def handle_deletion(
        self,
        source_id: str,
        deleted_source_ids: set[str],
        reason: str = "explicit_deletion",
    ) -> dict[str, Any]:
        """Handle explicit deletion of sources.

        Does not directly delete vector-store or lexical records.
        The indexing coordinator/orchestration layer owns downstream updates.
        """
        # Get current state
        state = self.persister.get_state(source_id)
        if state is None:
            state = SyncState(source_id=source_id, source_type=SourceType.LOCAL_FILE)

        # Mark deleted files
        state.deleted_files.update(deleted_source_ids)
        state.operation = "reconcile"
        state.last_sync = datetime.now(timezone.utc)

        # Persist
        self.persister.save_state(state)

        return {
            "deleted_source_ids": list(deleted_source_ids),
            "reason": reason,
            "state": state,
        }
