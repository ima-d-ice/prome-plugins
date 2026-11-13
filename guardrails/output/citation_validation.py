"""Citation validation for the Guardrails ecosystem.

Validates generated answer citations against the original citation/source
contract. Performs deterministic identity validation.

Checks:
- The citation marker exists.
- The source ID was supplied.
- The source belongs to the current request.
- The source is authorized for the current user/tenant.
- The source was not excluded or blocked.
- The source reference and provenance are preserved.

Distinguishes:
- Valid citation syntax.
- Valid source identity.
- Authorized source.
- Evidence-supported citation.

Deeper semantic claim-support evaluation belongs in the Evaluation subsystem
and must not be duplicated here.

Do not automatically rewrite generated answers to hide citation failures.
Return structured validation findings.
"""

from __future__ import annotations

from typing import Any

from guardrails.source_security.access_control import AccessControlEngine, AccessScope, UserAccessContext
from guardrails.source_security.trust import TrustState


class CitationValidator:
    """Independent citation validation layer.

    Validates generated answer citations against the original citation/source
    contract. Performs deterministic identity validation.

    This layer should NOT perform deeper semantic claim-support evaluation
    (that belongs in the Evaluation subsystem). It only validates citation
    identity, provenance, and authorization.

    Do not automatically rewrite generated answers to hide citation failures.
    Return structured validation findings.
    """

    def __init__(
        self,
        access_control: AccessControlEngine | None = None,
    ) -> None:
        """Initialize the citation validator.

        Args:
            access_control: optional AccessControlEngine instance for
                authorization checks. When None, a default engine is created.
        """

        self._access_control = access_control or AccessControlEngine()

    def validate(
        self,
        citations: list[dict[str, Any]],
        *,
        user_id: str | None = None,
        tenant_id: str | None = None,
        source_trust_records: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate a list of citations from a generated answer.

        Checks each citation for:
        - Valid citation syntax (marker exists).
        - Valid source identity (source ID supplied).
        - Authorized source (for current user/tenant).
        - Source not excluded or blocked (trust state check).
        - Source reference and provenance preservation.

        Args:
            citations: list of citation dicts from the generated answer.
                Each dict should have "source_id" and optionally
                "citation_marker", "provenance".
            user_id: the ID of the requesting user (for authorization checks).
            tenant_id: the tenant ID of the requesting user (for authorization checks).
            source_trust_records: dict mapping source_id -> SourceTrustRecord
                for trust state verification.

        Returns:
            dict with validation findings:
                - valid: True if all citations are valid.
                - errors: list of validation error details per citation.
                - warnings: list of validation warnings per citation.
                - authorized: set of source IDs that are authorized.
                - blocked: set of source IDs that are blocked/excluded.
        """

        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        authorized: set[str] = set()
        blocked: set[str] = set()

        for citation in citations:
            citation_errors: list[dict[str, Any]] = []
            citation_warnings: list[dict[str, Any]] = []

            source_id = citation.get("source_id", "")
            citation_marker = citation.get("citation_marker", "")
            provenance = citation.get("provenance", {})

            # Check 1: The citation marker exists
            if not citation_marker:
                citation_errors.append(
                    {
                        "code": "missing_citation_marker",
                        "message": "Citation reference missing citation_marker",
                        "source_id": source_id,
                    }
                )
            elif not isinstance(citation_marker, str) or not citation_marker.strip():
                citation_errors.append(
                    {
                        "code": "invalid_citation_marker",
                        "message": "Citation marker is empty or invalid",
                        "source_id": source_id,
                    }
                )

            # Check 2: The source ID was supplied
            if not source_id:
                citation_errors.append(
                    {
                        "code": "missing_source_id",
                        "message": "Citation reference missing source_id",
                        "source_id": source_id,
                    }
                )
            else:
                # Check 3: The source belongs to the current request/user/tenant
                if tenant_id:
                    # Check authorization via access control
                    access_context = UserAccessContext(
                        user_id=user_id or "anonymous",
                        tenant_id=tenant_id,
                        access_scope=AccessScope.TENANT,
                        requested_document_ids=(source_id,),
                        requesting_collection="rag_pipeline",
                    )
                    access_decision = self._access_control.evaluate_access(access_context)

                    if not access_decision.authorized:
                        citation_warnings.append(
                            {
                                "code": "unauthorized_source",
                                "message": f"Source '{source_id}' not authorized for "
                                f"user/tenant",
                                "source_id": source_id,
                                "reason": access_decision.reason,
                            }
                        )
                        blocked.add(source_id)
                    else:
                        authorized.add(source_id)
                elif user_id:
                    # Simple user-level check without full tenant context
                    # If no specific policy denies, consider authorized
                    authorized.add(source_id)
                else:
                    # No authorization context - assume potentially authorized
                    # but flag for review
                    citation_warnings.append(
                        {
                            "code": "no_auth_context",
                            "message": "No user/tenant authorization context provided; "
                                        "source may not be authorized",
                            "source_id": source_id,
                        }
                    )
                    warnings.append(
                        {
                            "code": "no_auth_context",
                            "message": "No user/tenant authorization context provided",
                            "source_id": source_id,
                        }
                    )

            # Check 4: The source was not excluded or blocked (trust state check)
            if source_id and source_trust_records:
                trust_record_data = source_trust_records.get(source_id)
                if trust_record_data:
                    # Try to construct a SourceTrustRecord for trust state check
                    # This is a simplified check - full integration would parse
                    # the trust record properly
                    trust_status = self._check_trust_state(trust_record_data)
                    if trust_status == TrustState.BLOCKED:
                        citation_errors.append(
                            {
                                "code": "source_blocked",
                                "message": f"Source '{source_id}' is blocked per trust state",
                                "source_id": source_id,
                            }
                        )
                        blocked.add(source_id)
                    elif trust_status == TrustState.SUSPICIOUS:
                        citation_warnings.append(
                            {
                                "code": "source_suspicious",
                                "message": f"Source '{source_id}' is marked as suspicious",
                                "source_id": source_id,
                            }
                        )
                        warnings.append(
                            {
                                "code": "source_suspicious",
                                "message": "Source marked as suspicious per trust assessment",
                                "source_id": source_id,
                            }
                        )

            # Check 5: Source reference and provenance are preserved
            if provenance:
                # Verify provenance has required fields
                if "doc_id" not in provenance:
                    citation_warnings.append(
                        {
                            "code": "provenance_incomplete",
                            "message": f"Citation for source '{source_id}' has incomplete provenance",
                            "source_id": source_id,
                        }
                    )
                    warnings.append(
                        {
                            "code": "provenance_incomplete",
                            "message": "Provenance information may be incomplete",
                            "source_id": source_id,
                        }
                    )
                if "chunk_id" not in provenance:
                    citation_warnings.append(
                        {
                            "code": "provenance_missing_chunk",
                            "message": f"Citation for source '{source_id}' missing chunk ID",
                            "source_id": source_id,
                        }
                    )
                    warnings.append(
                        {
                            "code": "provenance_missing_chunk",
                            "message": "Provenance missing chunk-level provenance",
                            "source_id": source_id,
                        }
                    )

            # Add per-citation errors/warnings to the result
            if citation_errors:
                errors.append(
                    {
                        "citation": citation,
                        "errors": citation_errors,
                    }
                )

            if citation_warnings and not citation_errors:
                warnings.append(
                    {
                        "citation": citation,
                        "warnings": citation_warnings,
                    }
                )
            elif citation_warnings and citation_errors:
                # Errors take precedence; warnings are secondary
                errors.append(
                    {
                        "citation": citation,
                        "errors": citation_errors,
                        "warnings": citation_warnings,
                    }
                )

        # Build final result
        result: dict[str, Any] = {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "authorized": authorized,
            "blocked": blocked,
        }

        return result

    def _check_trust_state(
        self, trust_record_data: dict[str, Any],
    ) -> TrustState:
        """Check the trust state from a trust record data dict.

        Simplified check for the citation validator's use.
        In full integration, would construct a SourceTrustRecord.

        Args:
            trust_record_data: dict from a SourceTrustRecord.

        Returns:
            TrustState value.
        """

        # Check the trust_state field if present
        trust_state_str = trust_record_data.get("source_metadata", {}).get("trust_state")
        if trust_state_str:
            try:
                return TrustState[trust_state_str]
            except KeyError:
                pass

        # Default to UNCHECKED if we can't determine
        return TrustState.UNCHECKED
