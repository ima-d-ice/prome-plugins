"""Change detector for Prome RAG Plugins incremental sync."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from loaders.contracts.models import (
    LoadedResource,
)
from loaders.sync.state import SyncState


class ChangeDetector:
    """Detects changes between synchronization runs.

    Uses stable source IDs, content hashes, ETag, and Last-Modified
    to identify:
    - New files.
    - Modified files.
    - Unchanged files.
    - Deleted files.
    - Changed URLs.
    - Updated web resources.
    """

    @staticmethod
    def compute_content_hash(content: bytes) -> str:
        """Compute SHA-256 content hash."""
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def compute_resource_hash(resource: LoadedResource) -> str:
        """Compute hash for a LoadedResource."""
        h = hashlib.sha256()
        content = resource.content
        if isinstance(content, str):
            content = content.encode("utf-8")
        h.update(content)
        if resource.content_hash:
            h.update(resource.content_hash.encode())
        return h.hexdigest()

    def detect_changes(
        self,
        previous_state: SyncState,
        current_resources: list[LoadedResource],
    ) -> dict[str, Any]:
        """Detect changes between a previous sync state and current resources.

        Returns dict with:
        - new: source_ids of new resources
        - modified: source_ids of changed resources
        - unchanged: source_ids of unchanged resources
        - deleted: source_ids that were removed
        - changed_urls: source_ids with changed URLs
        """
        current_by_id: dict[str, LoadedResource] = {}
        for resource in current_resources:
            current_by_id[resource.reference.source_id] = resource

        # Track changes
        new: set[str] = set()
        modified: set[str] = set()
        unchanged: set[str] = set()
        deleted: set[str] = set()
        changed_urls: set[str] = set()

        # Get previous tracked URLs
        prev_tracked = previous_state.tracked_urls

        # Check each current resource against previously tracked state
        for resource in current_resources:
            sid = resource.reference.source_id
            content_hash = resource.content_hash or self.compute_content_hash(resource.content)

            prev_info = prev_tracked.get(sid)
            if prev_info is None:
                # New resource never tracked before
                new.add(sid)
                continue

            prev_hash = prev_info.get("content_hash")
            prev_etag = prev_info.get("etag")
            current_etag = resource.provenance.etag.tag if resource.provenance.etag else None

            if content_hash == prev_hash and current_etag == prev_etag:
                unchanged.add(sid)
            else:
                modified.add(sid)
                changed_urls.add(sid)

        # Deleted files: were tracked, no longer present
        prev_sids = set(prev_tracked.keys())
        current_sids = set(current_by_id.keys())
        deleted = prev_sids - current_sids

        return {
            "new": sorted(new),
            "modified": sorted(modified),
            "unchanged": sorted(unchanged),
            "deleted": sorted(deleted),
            "changed_urls": sorted(changed_urls),
            "previous_state": previous_state,
            "current_resources": current_by_id,
        }

    def build_updated_state(
        self,
        previous: SyncState,
        detection: dict[str, Any],
        current_resources: list[LoadedResource],
    ) -> SyncState:
        """Build an updated SyncState based on change detection results."""

        updated = SyncState(
            source_id=previous.source_id,
            source_type=previous.source_type,
        )

        updated.last_sync = datetime.now(timezone.utc)
        updated.operation = "reconcile"

        # Update content hash if we have resources
        if current_resources:
            # Use the first resource's hash as the overall content hash
            updated.content_hash = current_resources[0].content_hash
        else:
            updated.content_hash = previous.content_hash

        # Update ETag and Last-Modified from provenance
        if current_resources and current_resources[0].provenance.etag:
            updated.etag = current_resources[0].provenance.etag.tag
        if current_resources and current_resources[0].provenance.last_modified:
            updated.last_modified = current_resources[0].provenance.last_modified.timestamp

        # Track new files
        updated.new_files = set(detection["new"])
        updated.modified_files = set(detection["modified"])
        updated.unchanged_files = set(detection["unchanged"])
        updated.deleted_files = set(detection["deleted"])

        # Update tracked URLs
        updated.tracked_urls = {}
        for resource in current_resources:
            sid = resource.reference.source_id
            updated.tracked_urls[sid] = {
                "content_hash": resource.content_hash,
                "etag": resource.provenance.etag.tag if resource.provenance.etag else None,
                "last_modified": resource.provenance.last_modified.timestamp.isoformat() if resource.provenance.last_modified else None,
            }

        # Copy metadata
        updated.metadata = {**previous.metadata, **detection.get("metadata", {})}

        return updated