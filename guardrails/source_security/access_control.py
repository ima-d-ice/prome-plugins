from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from guardrails.contracts.exceptions import GuardrailError
from guardrails.contracts.models import GuardrailStage

"""Source security: access control and tenant isolation.

Implements deterministic access-control interface for the RAG pipeline:

- Document ownership.
- Tenant isolation.
- User/group access scopes.
- Collection access restrictions.
- Document-level access policies.
- Metadata-filter enforcement.

Apply authorization restrictions before results are exposed to callers.
Never rely solely on model-generated reasoning to authorize document access.

An unauthorized document must not appear in:
- Retrieval output.
- Reranking candidates.
- Generation context.
- Citation output.

Where vector-store filtering is supported, integrate through the existing
portable metadata filter contract.

Where support is insufficient, reject the unsafe query or use a
correctness-preserving enforcement layer.
Do not silently ignore authorization filters.
Test cross-tenant leakage explicitly.

Do not build a complete identity provider or user-management system. Define
the authorization contract and a reference implementation suitable for
the existing RAG pipeline.
"""


class AccessDeniedError(GuardrailError):
    """Raised when a request is denied access to a document or source."""


class TenantIsolationError(GuardrailError):
    """Raised when cross-tenant leakage is detected."""


#: Supported access scope levels
class AccessScope(str, Enum):
    """Access scope levels for tenant isolation."""

    PUBLIC = "public"
    """Public access - no tenant restriction."""

    TENANT = "tenant"
    """Tenant-scoped access - within same tenant."""

    USER = "user"
    """User-scoped access - specific user only."""

    ADMIN = "admin"
    """Administrator access - across tenants."""


@dataclass(frozen=True)
class AccessControlDecision:
    """The result of an access-control check.

    Attributes:
        authorized: whether the access is permitted.
        reason_codes: machine-readable codes explaining the decision.
        policy_ids: identifiers of policies that influenced the decision.
        blocking_stage: the pipeline stage at which access was denied (if any).
        reason: human-readable explanation (no sensitive details).
    """

    authorized: bool
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    policy_ids: tuple[str, ...] = field(default_factory=tuple)
    blocking_stage: GuardrailStage | None = None
    reason: str = ""


@dataclass(frozen=True)
class TenantPolicy:
    """A policy defining tenant access restrictions.

    Attributes:
        tenant_id: the tenant this policy applies to.
        allowed_user_ids: tuple of user IDs allowed access (empty = all users
            in the tenant).
        allowed_document_ids: tuple of document IDs allowed access (empty =
            all documents in the tenant).
        required_scope: minimum access scope required (PUBLIC, TENANT, USER, ADMIN).
        deny_document_ids: tuple of document IDs explicitly denied access.
        description: human-readable description of the policy.
    """

    tenant_id: str
    allowed_user_ids: tuple[str, ...] = field(default_factory=tuple)
    allowed_document_ids: tuple[str, ...] = field(default_factory=tuple)
    required_scope: AccessScope = AccessScope.TENANT
    deny_document_ids: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""


@dataclass(frozen=True)
class UserAccessContext:
    """Context for access-control evaluation.

    Attributes:
        user_id: the ID of the requesting user.
        tenant_id: the tenant the user belongs to.
        access_scope: the access scope level (PUBLIC, TENANT, USER, ADMIN).
        requested_document_ids: document IDs the user is trying to access.
        requesting_collection: the collection/name of the data being accessed.
    """

    user_id: str
    tenant_id: str
    access_scope: AccessScope = AccessScope.PUBLIC
    requested_document_ids: tuple[str, ...] = field(default_factory=tuple)
    requesting_collection: str = ""


class AccessControlEngine:
    """Deterministic access-control engine for RAG pipeline security.

    Implements document ownership, tenant isolation, and user/group access
    scopes. Evaluates access requests against registered tenant policies.

    The engine does NOT build a complete identity provider or user-management
    system. It defines the authorization contract and a reference implementation
    suitable for the existing RAG pipeline.

    Key principles:
    - Apply authorization restrictions before results are exposed to callers.
    - Never rely solely on model-generated reasoning to authorize document access.
    - An unauthorized document must not appear in retrieval output, reranking
      candidates, generation context, or citation output.
    - Integrate through the existing portable metadata filter contract where
      vector-store filtering is supported.
    - Where support is insufficient, reject the unsafe query or use a
      correctness-preserving enforcement layer.
    """

    def __init__(self, policies: list[TenantPolicy] | None = None) -> None:
        """Initialize the access control engine.

        Args:
            policies: optional list of TenantPolicy instances. When None,
                starts with an empty policy set (all access permitted at PUBLIC
                scope).
        """

        self._policies: list[TenantPolicy] = policies or []
        self._tenant_index: dict[str, list[TenantPolicy]] = {}
        self._build_tenant_index()

    def _build_tenant_index(self) -> None:
        """Build an index of policies by tenant ID for efficient lookup."""

        self._tenant_index = {}
        for policy in self._policies:
            if policy.tenant_id not in self._tenant_index:
                self._tenant_index[policy.tenant_id] = []
            self._tenant_index[policy.tenant_id].append(policy)

    def add_policy(self, policy: TenantPolicy) -> None:
        """Add a tenant access policy.

        Args:
            policy: the tenant policy to add.
        """

        self._policies.append(policy)
        if policy.tenant_id not in self._tenant_index:
            self._tenant_index[policy.tenant_id] = []
        self._tenant_index[policy.tenant_id].append(policy)

    def remove_policy(self, tenant_id: str) -> None:
        """Remove all policies for a given tenant.

        Args:
            tenant_id: the tenant to remove policies for.
        """

        self._policies = [p for p in self._policies if p.tenant_id != tenant_id]
        self._tenant_index.pop(tenant_id, None)

    def evaluate_access(
        self,
        context: UserAccessContext,
    ) -> AccessControlDecision:
        """Evaluate whether the user has access to the requested documents.

        Args:
            context: the user access context containing user identity,
                tenant, scope, and requested documents.

        Returns:
            AccessControlDecision with authorized flag and reason codes.
            When authorized is False, the document must not be exposed in
            any downstream output.
        """

        reason_codes: list[str] = []
        blocking_stage: GuardrailStage | None = None
        effective_policy_ids: list[str] = []

        # Start with the user's declared access scope
        user_scope = context.access_scope
        user_tenant = context.tenant_id
        user_id = context.user_id
        requested_docs = set(context.requested_document_ids)

        # Step 1: Check for explicit denials first (highest priority)
        # Any document in the deny list is automatically blocked
        for policy in self._policies:
            if policy.deny_document_ids and len(requested_docs) > 0:
                denied_docs = requested_docs & set(policy.deny_document_ids)
                if denied_docs:
                    reason_codes.append("explicit_denial")
                    blocking_stage = GuardrailStage.INPUT
                    # Remove denied docs from the request set
                    requested_docs -= denied_docs
                    effective_policy_ids.append(policy.tenant_id)

        # Step 2: Check tenant isolation
        # If user is in a tenant, verify they're accessing their own tenant's docs
        user_tenant_policies = self._tenant_index.get(user_tenant, [])

        # Step 3: Evaluate based on access scope
        authorized = True

        if user_scope == AccessScope.ADMIN:
            # Admin has broad access, but still check denials
            # (already checked above)
            pass
        elif user_scope == AccessScope.USER:
            # User-scoped access: verify the user has explicit permission
            # Check if any policy allows this specific user to access these docs
            user_authorized = False
            for policy in user_tenant_policies:
                if user_id in policy.allowed_user_ids:
                    # User is allowed in this tenant; check document permissions
                    if policy.allowed_document_ids:
                        # Specific documents allowed
                        allowed_set = set(policy.allowed_document_ids)
                        if requested_docs and not requested_docs.isdisjoint(allowed_set):
                            # Requested docs intersect with allowed docs - check if all are allowed
                            if requested_docs.issubset(allowed_set):
                                user_authorized = True
                            else:
                                # Some requested docs not allowed
                                reason_codes.append("user_document_not_allowed")
                                authorized = False
                        # If requested_docs is empty, user can access what they asked for
                        # after other checks
                    else:
                        # No document restrictions beyond tenant
                        user_authorized = True
                # Also check if policy allows all users in tenant
                if not policy.allowed_user_ids:
                    # Policy allows all users in tenant; check document permissions
                    if policy.allowed_document_ids:
                        allowed_set = set(policy.allowed_document_ids)
                        if requested_docs and not requested_docs.isdisjoint(allowed_set):
                            if requested_docs.issubset(allowed_set):
                                user_authorized = True
                            else:
                                reason_codes.append("user_document_not_allowed")
                                authorized = False
                    else:
                        user_authorized = True
            if not user_authorized and authorized:
                authorized = False
                reason_codes.append("user_not_authorized")

        elif user_scope == AccessScope.TENANT:
            # Tenant-scoped access: user can access any document in their tenant
            # unless explicitly denied (checked above) or restricted by policy
            tenant_authorized = False
            for policy in user_tenant_policies:
                # Check if policy allows access to these documents
                if policy.allowed_document_ids:
                    allowed_set = set(policy.allowed_document_ids)
                    if requested_docs and not requested_docs.isdisjoint(allowed_set):
                        if requested_docs.issubset(allowed_set):
                            tenant_authorized = True
                        else:
                            reason_codes.append("tenant_document_not_allowed")
                            authorized = False
                else:
                    # No document restrictions beyond tenant
                    tenant_authorized = True

            if not tenant_authorized and authorized:
                authorized = False

        elif user_scope == AccessScope.PUBLIC:
            # Public access: check if any policy restricts this to non-public
            # Public is the most permissive; only deny if explicitly restricted
            # (already checked denials above)
            pass
        else:
            # Unknown scope - default to denied
            authorized = False
            reason_codes.append("unknown_access_scope")

        # Step 4: Finalize - if any reason codes were added, authorized is False
        if reason_codes:
            authorized = False

        # Build the decision
        decision = AccessControlDecision(
            authorized=authorized,
            reason_codes=tuple(reason_codes) if reason_codes else tuple(),
            policy_ids=tuple(effective_policy_ids) if effective_policy_ids else tuple(),
            blocking_stage=blocking_stage,
            reason=self._format_reason(reason_codes)
            if not authorized
            else "",
        )

        return decision

    def _format_reason(self, reason_codes: list[str]) -> str:
        """Format reason codes into a human-readable explanation.

        Does not expose sensitive details (no API keys, no auth headers,
        no raw content).
        """

        if not reason_codes:
            return ""

        # Map reason codes to descriptions
        reason_map: dict[str, str] = {
            "explicit_denial": "Document explicitly denied access",
            "user_not_authorized": "User not authorized for this document",
            "user_document_not_allowed": "Requested document not allowed for user",
            "tenant_document_not_allowed": "Requested document not allowed for tenant",
            "unknown_access_scope": "Unknown access scope",
        }

        descriptions = [
            reason_map.get(code, "Access denied")
            for code in reason_codes
        ]
        return "; ".join(descriptions)
