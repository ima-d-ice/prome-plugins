"""Source trust and provenance for the Guardrails ecosystem.

Each source/context item retains: Document ID, Chunk ID, Original source reference,
Trust state, Prompt Guard status, Ingestion validation status, Optional policy decisions,
Timestamp and relevant diagnostics.

Trust states:
    UNCHECKED:   No security validation has been performed.
    TRUSTED:     Passed all security checks (prompt injection, access control, privacy).
    UNTRUSTED:   Failed a security check but is not blocked (e.g., warned).
    SUSPICIOUS:   Potential security concern detected (prompt injection, privacy leak).
    BLOCKED:      Explicitly blocked by policy or access control.

Do not automatically classify a document as trusted merely because it was
parsed successfully.

Source-level isolation: a malicious chunk should not automatically poison
every other chunk from the same document. Record suspicious locations and
allow the caller to: exclude the chunk, exclude the source, review it,
retain it with an explicit warning. Do not silently drop source evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any


class TrustState(Enum):
    """Trust state of a source or context item."""

    UNCHECKED = auto()
    """No security validation has been performed."""

    TRUSTED = auto()
    """Passed all security checks (prompt injection, access control, privacy)."""

    UNTRUSTED = auto()
    """Failed a security check but is not blocked (e.g., warned)."""

    SUSPICIOUS = auto()
    """Potential security concern detected (prompt injection, privacy leak)."""

    BLOCKED = auto()
    """Explicitly blocked by policy or access control."""


#: Human-readable trust state names
TRUST_STATE_NAMES: dict[TrustState, str] = {
    TrustState.UNCHECKED: "unchecked",
    TrustState.TRUSTED: "trusted",
    TrustState.UNTRUSTED: "untrusted",
    TrustState.SUSPICIOUS: "suspicious",
    TrustState.BLOCKED: "blocked",
}


@dataclass(frozen=True)
class SourceMetadata:
    """Metadata describing the source of a chunk or document.

    Attributes:
        doc_id: document identifier.
        chunk_id: chunk identifier (optional, None for whole-document sources).
        source_reference: original source reference (URI, path, ID).
        trust_state: current trust state of this source.
        prompt_guard_status: Prompt Guard classification result (BENIGN/MALICIOUS).
        ingestion_validation_status: result of ingestion validation.
        policy_decisions: optional policy decisions that affected this source.
        timestamp: when the trust state was last updated.
        diagnostics: additional diagnostics about the trust assessment.
    """

    doc_id: str
    chunk_id: str | None = None
    source_reference: str | None = None
    trust_state: TrustState = TrustState.UNCHECKED
    prompt_guard_status: str | None = None  # "BENIGN" or "MALICIOUS"
    ingestion_validation_status: str | None = None
    policy_decisions: tuple[str, ...] = field(default_factory=tuple)
    timestamp: float | None = None
    diagnostics: dict[str, Any] | None = None

    def with_updated_trust(self, new_state: TrustState) -> SourceMetadata:
        """Return new SourceMetadata with updated trust state.

        Args:
            new_state: the new trust state.

        Returns:
            New SourceMetadata instance with updated trust state and updated
            timestamp.
        """

        now = datetime.now(timezone.utc).timestamp()
        return SourceMetadata(
            doc_id=self.doc_id,
            chunk_id=self.chunk_id,
            source_reference=self.source_reference,
            trust_state=new_state,
            prompt_guard_status=self.prompt_guard_status,
            ingestion_validation_status=self.ingestion_validation_status,
            policy_decisions=self.policy_decisions,
            timestamp=now,
            diagnostics=self.diagnostics,
        )

    def with_updated_diagnostics(self, additional: dict[str, Any]) -> SourceMetadata:
        """Return new SourceMetadata with updated diagnostics.

        Args:
            additional: additional diagnostic key-value pairs to merge.

        Returns:
            New SourceMetadata instance with merged diagnostics.
        """

        existing = self.diagnostics or {}
        merged = {**existing, **additional}
        return SourceMetadata(
            doc_id=self.doc_id,
            chunk_id=self.chunk_id,
            source_reference=self.source_reference,
            trust_state=self.trust_state,
            prompt_guard_status=self.prompt_guard_status,
            ingestion_validation_status=self.ingestion_validation_status,
            policy_decisions=self.policy_decisions,
            timestamp=self.timestamp,
            diagnostics=merged,
        )


@dataclass(frozen=True)
class SourceTrustRecord:
    """A complete trust record for a source or chunk.

    Combines the source metadata with the full audit trail of security
    assessments that led to the current trust state.

    Attributes:
        source_metadata: the source metadata including trust state.
        prompt_guard_assessment: whether Prompt Guard was applied and result.
        ingestion_validation: whether ingestion validation was applied and result.
        policy_decisions: policy decisions that influenced the trust state.
        timestamp: when the trust assessment was performed.
        audit_trail: list of (policy_id, decision, timestamp) tuples describing
            each security assessment step.
    """

    source_metadata: SourceMetadata
    prompt_guard_assessment: dict[str, Any] | None = None
    ingestion_validation: dict[str, Any] | None = None
    policy_decisions: tuple[str, ...] = field(default_factory=tuple)
    timestamp: float | None = None
    audit_trail: tuple[tuple[str, str, float], ...] = field(default_factory=tuple)

    def with_new_audit_entry(
        self,
        policy_id: str,
        decision: str,
        timestamp: float | None = None,
    ) -> SourceTrustRecord:
        """Add a new audit entry to the trust record's trail.

        Args:
            policy_id: the policy that was evaluated.
            decision: the decision reached (allow/warn/block/review).
            timestamp: when the decision was made. When None, uses current time.

        Returns:
            New SourceTrustRecord with the added audit entry.
        """

        if timestamp is None:
            timestamp = datetime.now(timezone.utc).timestamp()

        new_entry = (policy_id, decision, timestamp)
        existing_trail = self.audit_trail
        new_trail = existing_trail + (new_entry,)

        return self._replace_with_trail(new_trail)

    def _replace_with_trail(self, audit_trail: tuple) -> SourceTrustRecord:
        """Internal helper to create a new record with updated audit trail."""

        # Since dataclasses are frozen, we create a new instance
        return SourceTrustRecord(
            source_metadata=self.source_metadata,
            prompt_guard_assessment=self.prompt_guard_assessment,
            ingestion_validation=self.ingestion_validation,
            policy_decisions=self.policy_decisions,
            timestamp=self.timestamp,
            audit_trail=audit_trail,
        )

    def get_effective_trust(self) -> TrustState:
        """Get the effective trust state considering all factors.

        The effective trust considers:
        1. The explicit trust_state in source_metadata
        2. Any SUSPICIOUS prompt guard findings
        3. Any BLOCKED ingestion validation findings
        4. Any policy denials

        Returns the most restrictive applicable state.
        """

        # Start with the explicit trust state
        effective = self.source_metadata.trust_state

        # Check prompt guard assessment - if malicious, at least SUSPICIOUS
        if self.prompt_guard_assessment:
            pg_status = self.prompt_guard_assessment.get("classification")
            if pg_status == "MALICIOUS":
                if effective.value < TrustState.SUSPICIOUS.value:
                    effective = TrustState.SUSPICIOUS
                # If already BLOCKED, keep BLOCKED

        # Check ingestion validation - if blocked, at least BLOCKED
        if self.ingestion_validation:
            inv_status = self.ingestion_validation.get("status")
            if inv_status == "blocked":
                if effective.value < TrustState.BLOCKED.value:
                    effective = TrustState.BLOCKED

        # Check policy decisions - if any block, at least BLOCKED
        for decision in self.policy_decisions:
            # Decisions are stored as "allow"/"warn"/"block"/"review"
            if decision == "block":
                if effective.value < TrustState.BLOCKED.value:
                    effective = TrustState.BLOCKED
                break  # block is the strongest

        return effective


@dataclass(frozen=True)
class ChunkProvenance:
    """Provenance information for a chunk, preserving original document text and origin.

    Attributes:
        doc_id: document identifier.
        chunk_id: chunk identifier.
        char_start: start character offset in the original document text.
        char_end: end character offset in the original document text.
        page_number: page number (None for pageless documents).
        block_type: type of block (e.g., "paragraph", "heading", "table").
        original_text_preserved: whether the original document text is preserved
            separate from any redaction or modification.
    """

    doc_id: str
    chunk_id: str
    char_start: int
    char_end: int
    page_number: int | None = None
    block_type: str = "paragraph"
    original_text_preserved: bool = True


@dataclass(frozen=True)
class SourceProvenance:
    """Full provenance tracking for a source, linking chunks to their
    original documents and tracking trust/validation state.

    Attributes:
        doc_id: document identifier.
        chunk_id: chunk identifier.
        original_text: original document text (preserved, not modified).
        char_start: start character offset in original document.
        char_end: end character offset in original document.
        trust_record: the SourceTrustRecord for this source.
        chunk_provenance: ChunkProvenance for positional tracking.
        ingestion_timestamp: when the source was ingested.
        last_scanned: when the source was last security-scanned.
    """

    doc_id: str
    chunk_id: str
    original_text: str
    char_start: int
    char_end: int
    trust_record: SourceTrustRecord
    chunk_provenance: ChunkProvenance
    ingestion_timestamp: float | None = None
    last_scanned: float | None = None
