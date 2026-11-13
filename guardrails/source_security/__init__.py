"""Source security and trust management for the Guardrails ecosystem.

Provides:
- Trust state management (UNCHECKED, TRUSTED, UNTRUSTED, SUSPICIOUS, BLOCKED)
- Source provenance tracking with original text preservation
- Deterministic access control and tenant isolation
- Source-level isolation preventing malicious chunks from poisoning other
  chunks from the same document

Key interfaces:
- SourceTrustRecord: complete trust assessment for a source/chunk
- SourceProvenance: full provenance linking chunks to original documents
- AccessControlEngine: deterministic access-control evaluation
- TenantPolicy: tenant-level access policies
- TrustState: trust state enumeration

Trust states:
    UNCHECKED:   No security validation performed
    TRUSTED:     Passed all security checks
    UNTRUSTED:   Failed check but not blocked
    SUSPICIOUS:   Potential security concern detected
    BLOCKED:      Explicitly blocked by policy or access control

Source-level isolation: a malicious chunk should not automatically poison
every other chunk from the same document. Record suspicious locations and
allow the caller to: exclude the chunk, exclude the source, review it,
retain it with an explicit warning. Do not silently drop source evidence.
"""

from guardrails.source_security.access_control import (
    AccessControlDecision,
    AccessControlEngine,
    AccessScope,
    TenantPolicy,
    UserAccessContext,
)
from guardrails.source_security.trust import (
    ChunkProvenance,
    SourceMetadata,
    SourceProvenance,
    SourceTrustRecord,
    TrustState,
)

__all__ = [
    "TrustState",
    "SourceMetadata",
    "SourceTrustRecord",
    "ChunkProvenance",
    "SourceProvenance",
    "AccessControlDecision",
    "TenantPolicy",
    "UserAccessContext",
    "AccessControlEngine",
    "AccessScope",
]
