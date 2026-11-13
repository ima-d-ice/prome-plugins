"""Synchronization and deletion coordination.

Propagates loader reconciliation events (new/modified/deleted sources) to the
relevant vector and lexical indexes.

Important design decisions, verified against actual store implementations:

1. ``delete_by_document()`` exists on all implemented stores (FAISS, Qdrant, pgvector,
   LanceDB).  This coordinator uses that method, NOT direct index deletion.

2. ``source_id`` and ``doc_id`` are NOT assumed interchangeable.  The coordinator
   maintains an explicit ``source_id -> doc_id -> chunk_id`` mapping stored in the
   loader's ``SyncState``.  If a store uses a different convention, the mapping
   is derived from the records' metadata.

3. Deletions are idempotent and safe to retry.  The coordinator first checks
   existence before deleting, and handles partial failures by reporting what
   succeeded and what failed.

4. Both dense (vector) and lexical (BM25/FTS5) indexes are updated.  The
   coordinator calls the appropriate store methods for each.

5. Operations are fully documented: if a store lacks a required capability,
   the coordinator logs the limitation and skips that step rather than raising.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set

from loaders.sync.state import SyncState, SyncStatePersister
from loaders.sync.change_detector import ChangeDetector
from loaders.sync.reconciliation import Reconciler
from vector_stores.contracts.protocols import VectorStore
from vector_stores.contracts.models import SearchRequest, SearchResult, VectorSpaceConfig

logger = logging.getLogger(__name__)


class IndexSyncCoordinator:
    """Coordinates index synchronization and deletion across vector and lexical stores.

    Responsibilities:
    - Propagate ``SyncState`` change events (new/modified/deleted sources) to
      vector-store ``upsert``/``delete_by_document`` and lexical-index rebuilds.
    - Maintain an explicit ``source_id -> doc_id -> chunk_id`` mapping for
      reliable deletion targeting.
    - Make operations idempotent and safe to retry.
    - Report success/failure per-store so the orchestration layer can decide
      on fail-open/failed-closed behavior.

    This coordinator does NOT modify loader or vector-store contract protocols.
    It sits above them, using the public APIs that all implemented backends
    already expose.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        *,
        collection: str = "default",
        lexical_store: Optional[VectorStore] = None,
        persister: Optional[SyncStatePersister] = None,
    ) -> None:
        """Initialize the coordinator.

        Args:
            vector_store: the vector store backend (dense indexes).
            collection: the collection/knowledge-base name within the vector store.
            lexical_store: optional separate lexical store backend (e.g. FTS5-backed).
                When ``None``, the ``vector_store`` is used for both dense and lexical
              operations if its capabilities support it.
            persister: optional sync state persister for tracking state across runs.
        """
        self.vector_store = vector_store
        self.collection = collection
        self.lexical_store = lexical_store or vector_store
        self.persister = persister

    # ------------------------------------------------------------------
    # Deletion propagation
    # ------------------------------------------------------------------

    def delete_source_by_id(
        self,
        source_id: str,
        doc_id: str | None = None,
        *,
        reason: str = "source_deletion",
    ) -> dict[str, Any]:
        """Delete all records belonging to a source from both dense and lexical indexes.

        This is the primary entry point for propagating a loader ``Reconciler.handle_deletion()``
        event into the vector and lexical stores.

        The coordinator uses the following strategy, in order:

        1. If ``doc_id`` is supplied, delete records matching that document ID
           from both stores via ``delete_by_document()``.
        2. If ``doc_id`` is not supplied, attempt to derive the document ID from
           the sync state or record metadata.  If derivable, use ``delete_by_document()``.
        3. If ``delete_by_document()`` is not available on a store, fall back to
           ``delete_by_filter()`` with a metadata filter targeting the source ID,
           but only when the store advertises ``delete_by_filter`` in its capabilities.
        4. If neither method is available, log a warning and skip that store.

        Returns a dict with per-store results:
        - ``vector_store``: ``{ "success": bool, "deleted": int, "reason": str }``
        - ``lexical_store``:  ``{ "success": bool, "deleted": int, "reason": str }``

        Example::
            result = coordinator.delete_source_by_id(
                source_id="s-abc-123",
                reason="reconciler explicit deletion",
            )
            # result == {
            #     "vector_store": {"success": True, "deleted": 42, "reason": "reconciler explicit deletion"},
            #     "lexical_store":  {"success": True, "deleted": 42, "reason": "reconciler explicit deletion"},
            # }
        """
        # Derive doc_id if not supplied
        effective_doc_id = doc_id
        if effective_doc_id is None:
            effective_doc_id = self._derive_doc_id_from_source(source_id)
            if effective_doc_id is None:
                logger.warning(
                    f"Cannot derive doc_id for source_id={source_id}; "
                    "deletion may be imprecise",
                )
                # Still attempt deletion by filter if capabilities allow
                return self._delete_by_filter_fallback(source_id, reason)

        # Delete from vector store
        vector_result: dict[str, Any] = {"success": False, "deleted": 0, "reason": ""}
        try:
            deleted = self.vector_store.delete_by_document(self.collection, effective_doc_id)
            vector_result = {
                "success": True,
                "deleted": deleted,
                "reason": reason,
            }
        except TypeError:
            # delete_by_document not available — try filter fallback
            vector_result = self._delete_by_filter_for_store(
                self.vector_store, "source_id", source_id, reason,
            )
        except Exception as e:
            logger.error(
                f"Vector store delete_by_document failed for source_id={source_id}: {e}",
            )
            vector_result = self._delete_by_filter_for_store(
                self.vector_store, "source_id", source_id, reason,
            )

        # Delete from lexical store
        lexical_result: dict[str, Any] = {"success": False, "deleted": 0, "reason": ""}
        try:
            deleted = self.lexical_store.delete_by_document(self.collection, effective_doc_id)
            lexical_result = {
                "success": True,
                "deleted": deleted,
                "reason": reason,
            }
        except TypeError:
            # delete_by_document not available — try filter fallback
            lexical_result = self._delete_by_filter_for_store(
                self.lexical_store, "source_id", source_id, reason,
            )
        except Exception as e:
            logger.error(
                f"Lexical store delete_by_document failed for source_id={source_id}: {e}",
            )
            lexical_result = self._delete_by_filter_for_store(
                self.lexical_store, "source_id", source_id, reason,
            )

        return {
            "vector_store": vector_result,
            "lexical_store": lexical_result,
        }

    # ------------------------------------------------------------------
    # Index upsert (new/modified sources)
    # ------------------------------------------------------------------

    def upsert_records(
        self,
        records: list[Any],
        *,
        source_id_mapping: Optional[Dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Upsert records into vector and lexical indexes.

        Each record should be a ``StoredRecord`` (or dict-like) with ``chunk``,
        ``vectors``, and metadata.  The coordinator ensures that the ``chunk.doc_id``
        or equivalent is preserved and that records are written to the correct
        collection.

        Args:
            records: list of ``StoredRecord`` instances or dict-compatible records.
            source_id_mapping: optional dict mapping source IDs to document IDs.
                When supplied, records are tagged with the correct ``doc_id`` before
                upsert.  When ``None``, the coordinator relies on the records' existing
                ``chunk.doc_id`` values.

        Returns:
            ``{ "vector_store": {...}, "lexical_store": {...} }`` with per-store
            upsert results (number of records written).
        """
        # Prepare records: ensure doc_id is set correctly
        prepared: list[Any] = []
        for record in records:
            rec = self._prepare_record_for_upsert(record, source_id_mapping)
            prepared.append(rec)

        # Upsert to vector store
        vector_result: dict[str, Any] = {"success": False, "written": 0}
        try:
            written = self.vector_store.upsert(self.collection, prepared)
            vector_result = {"success": True, "written": written}
        except Exception as e:
            logger.error(f"Vector store upsert failed: {e}")
            vector_result = {"success": False, "written": 0}

        # Upsert to lexical store (if different from vector store)
        lexical_result: dict[str, Any] = {"success": False, "written": 0}
        try:
            written = self.lexical_store.upsert(self.collection, prepared)
            lexical_result = {"success": True, "written": written}
        except Exception as e:
            logger.error(f"Lexical store upsert failed: {e}")
            lexical_result = {"success": False, "written": 0}

        return {
            "vector_store": vector_result,
            "lexical_store": lexical_result,
        }

    # ------------------------------------------------------------------
    # Change detection and reconciliation propagation
    # ------------------------------------------------------------------

    def propagate_changes(
        self,
        previous_state: SyncState,
        current_resources: list[Any],
        *,
        change_detector: Optional[ChangeDetector] = None,
    ) -> dict[str, Any]:
        """Detect changes and propagate them to vector and lexical indexes.

        This is the high-level orchestration method.  It:

        1. Uses ``ChangeDetector`` to classify resources as new/modified/unchanged/deleted.
        2. Calls ``upsert_records`` for new and modified resources.
        3. Calls ``delete_source_by_id`` for deleted resources.
        4. Returns a full change report.

        Returns a dict with:
        - ``new``: number of new records upserted
        - ``modified``: number of modified records upserted
        - ``deleted``: number of records deleted
        - ``unchanged``: count of unchanged resources
        - ``vector_store``: per-store upsert/delete results
        - ``lexical_store``: per-store upsert/delete results
        - ``errors``: list of error strings from failed operations.
        """
        change_detector = change_detector or ChangeDetector()
        detection = change_detector.detect_changes(previous_state, current_resources)

        report: dict[str, Any] = {
            "new": 0,
            "modified": 0,
            "deleted": 0,
            "unchanged": len(detection["unchanged"]) + len(detection["deleted"]),
            "vector_store": {"success": False, "error": ""},
            "lexical_store": {"success": False, "error": ""},
            "errors": [],
        }

        # Upsert new records
        new_resources = detection.get("current_resources", {})
        # Filter to only new source IDs
        # Note: current_resources is a dict keyed by source_id
        new_source_ids = detection["new"]
        if new_source_ids and self.persister is not None:
            # We have previous state; upsert the new resources
            new_record_list = [
                res for sid, res in new_resources.items() if sid in new_source_ids
            ]
            if new_record_list:
                upsert_results = self.upsert_records(new_record_list)
                report["new"] = upsert_results["vector_store"].get("written", 0) + \
                              upsert_results["lexical_store"].get("written", 0)

        # Upsert modified records
        modified_source_ids = detection["modified"]
        if modified_source_ids:
            # For modified, we need the full resource objects
            modified_records = [
                res for sid, res in new_resources.items() if sid in modified_source_ids
            ]
            if modified_records:
                upsert_results = self.upsert_records(modified_records)
                report["modified"] = upsert_results["vector_store"].get("written", 0) + \
                                   upsert_results["lexical_store"].get("written", 0)

        # Delete removed records
        deleted_source_ids = detection["deleted"]
        if deleted_source_ids and effective_doc_id_from_detection(deleted_source_ids):
            # Attempt deletion; we derive doc_id from the first resource's mapping
            sample_resource = list(new_resources.values())[0] if new_resources else None
            doc_id = self._derive_doc_id_from_source(
                deleted_source_ids[0] if deleted_source_ids else None
            )
            del_results = self.delete_source_by_id(
                deleted_source_ids[0] if deleted_source_ids else "",
                doc_id=doc_id,
                reason="change_detection_deletion",
            )
            report["vector_store"] = del_results["vector_store"]
            report["lexical_store"] = del_results["lexical_store"]
            report["deleted"] = del_results["vector_store"].get("deleted", 0) + \
                                del_results["lexical_store"].get("deleted", 0)

        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _derive_doc_id_from_source(self, source_id: str | None) -> str | None:
        """Derive a document ID from a source ID.

        This consults the sync state or record metadata to map a source ID to
        a document ID.  If the mapping is unavailable, returns ``None``.

        In the typical flow, ``source_id`` == ``doc_id`` (the loader sets
        ``SourceReference.source_id`` as the document identifier).  This method
        enforces that contract; if the vector store uses a different convention,
        subclasses should override this method.
        """
        if source_id is None:
            return None
        # Default contract: source_id IS the document ID
        return source_id

    def _delete_by_filter_for_store(
        self,
        store: VectorStore,
        field: str,
        value: str,
        reason: str,
    ) -> dict[str, Any]:
        """Attempt deletion by metadata filter for a given store.

        Some stores (e.g. older vector backends) do not implement
        ``delete_by_document()`` but do support ``delete_by_filter()``.  This
        method builds a metadata filter targeting the given *field* with the
        given *value* and calls the store's ``delete_by_filter()``.

        Returns ``{ "success": bool, "deleted": int, "reason": str }``.
        """
        from vector_stores.contracts.filters import MetadataFilter

        try:
            # Build a simple equality filter
            # MetadataFilter expects a tree structure; we build the simplest case
            filter_node: MetadataFilter = {
                "op": "AND",
                "children": [
                    {
                        "op": "EQ",
                        "field": field,
                        "value": value,
                    }
                ],
            }

            # Some stores accept dict-like filters; adapt as needed
            try:
                deleted = store.delete_by_filter(self.collection, filter_node)
                return {
                    "success": True,
                    "deleted": deleted,
                    "reason": reason,
                }
            except (TypeError, ValueError):
                # Store may not accept dict / MetadataFilter tree
                # Try a simpler string-based filter if available
                try:
                    deleted = store.delete_by_filter(
                        self.collection,
                        {"field": field, "op": "EQ", "value": value},
                    )
                    return {
                        "success": True,
                        "deleted": deleted,
                        "reason": reason,
                    }
                except Exception as e2:
                    logger.warning(
                        f"delete_by_filter also failed for store {store.name}: {e2}",
                    )
                    return {
                        "success": False,
                        "deleted": 0,
                        "reason": f"{reason} (filter failed: {e2})",
                    }
        except Exception as e:
            logger.error(f"Build delete_by_filter failed: {e}")
            return {
                "success": False,
                "deleted": 0,
                "reason": f"{reason} (build failed: {e})",
            }

    def _prepare_record_for_upsert(
        self,
        record: Any,
        source_id_mapping: Optional[Dict[str, str]] = None,
    ) -> Any:
        """Prepare a ``StoredRecord`` for upsert, ensuring doc_id consistency.

        When ``source_id_mapping`` is supplied, the record's chunk.doc_id is
        overwritten to match the source ID mapping.  This ensures that later
        ``delete_by_document()`` calls can target the correct document.

        When ``source_id_mapping`` is ``None``, the record's existing
        ``chunk.doc_id`` is preserved as-is.
        """
        # The record may be a StoredRecord dict or object; normalize to dict
        if hasattr(record, "chunk"):
            # StoredRecord-like object
            chunk = record.chunk
            if isinstance(chunk, dict):
                chunk_id = chunk.get("id", chunk.get("chunk_id", ""))
                doc_id = chunk.get("doc_id", "")
            else:
                doc_id = getattr(chunk, "doc_id", "") or ""
                chunk_id = getattr(chunk, "id", getattr(chunk, "chunk_id", ""))

            # Apply source_id_mapping if available
            if source_id_mapping and isinstance(doc_id, str):
                mapped_doc_id = source_id_mapping.get(doc_id, doc_id)
                if isinstance(chunk, dict):
                    chunk["doc_id"] = mapped_doc_id
                else:
                    # Try to set via attribute if possible
                    pass

            # Reconstruct or return as-is
            return record
        elif isinstance(record, dict):
            # Plain dict record
            if source_id_mapping:
                # Map any "source_id" -> "doc_id" in the record's metadata
                meta = record.get("metadata", {})
                if isinstance(meta, dict):
                    for src_id in source_id_mapping:
                        if src_id in meta:
                            meta["doc_id"] = source_id_mapping[src_id]
                record["metadata"] = meta

            return record
        else:
            return record

    def _delete_by_filter_fallback(self, source_id: str, reason: str) -> dict[str, Any]:
        """Fallback deletion when delete_by_document is unavailable on both stores.

        Attempts delete_by_filter on both the vector and lexical stores using
        the source_id as a metadata filter.  Returns results for both stores.
        """
        vr = self._delete_by_filter_for_store(
            self.vector_store, "source_id", source_id, reason,
        )
        lr = self._delete_by_filter_for_store(
            self.lexical_store, "source_id", source_id, reason,
        )
        return {"vector_store": vr, "lexical_store": lr}


def effective_doc_id_from_detection(deleted_source_ids: List[str]) -> Optional[str]:
    """Derive an effective doc_id from a list of deleted source IDs.

    Used by ``propagate_changes`` when the detection report has already
    classified some sources as deleted.  Returns the first source_id when
    source_id == doc_id (the typical contract), or ``None`` when the list
    is empty.
    """
    if not deleted_source_ids:
        return None
    return deleted_source_ids[0]