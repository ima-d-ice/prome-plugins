"""Security audit events for the Guardrails ecosystem.

Implements structured, minimal audit events that record security-relevant
information across the pipeline. Designed for privacy-preserving logging.

Never log:
- API keys.
- Authorization headers.
- Full secrets.
- Raw private content by default.
- Unnecessary complete prompts.

Default to privacy-preserving logs.

Support configurable retention in the audit storage interface.

Do not build a separate observability platform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


class AuditEvent:
    """Structured security audit event.

    Records security-relevant information across the RAG pipeline in a
    privacy-preserving way. Each event contains enough information to be
    auditable without exposing sensitive data.

    Attributes:
        request_id: unique identifier for the request (no API keys or headers).
        timestamp: when the event occurred (ISO-8601 format).
        pipeline_stage: the guardrail stage where the event was recorded.
        policy_id: optional policy ID that influenced the decision.
        policy_version: optional policy version.
        detector_model: the model or classifier used, if applicable.
        decision: the security decision (allow/warn/block/review).
        reason_codes: machine-readable codes explaining the decision.
        source_reference: reference to source or chunk (chunk_id, doc_id).
        latency: how long the check took in milliseconds.
        error_status: whether the check resulted in an error.
        redacted: whether content was redacted or excluded.
        diagnostics: additional structured diagnostic information (no full
            secrets, no auth headers).
    """

    def __init__(
        self,
        *,
        request_id: str,
        pipeline_stage: GuardrailStage,
        policy_id: str | None = None,
        policy_version: str | None = None,
        detector_model: str | None = None,
        decision: str | None = None,
        reason_codes: list[str] | None = None,
        source_reference: str | None = None,
        latency: float | None = None,
        error_status: bool = False,
        redacted: bool = False,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        self.request_id = request_id
        self.pipeline_stage = pipeline_stage
        self.policy_id = policy_id
        self.policy_version = policy_version
        self.detector_model = detector_model
        self.decision = decision
        self.reason_codes = reason_codes or []
        self.source_reference = source_reference
        self.latency = latency
        self.error_status = error_status
        self.redacted = redacted
        self.diagnostics = diagnostics or {}
        self._timestamp = datetime.now(timezone.utc)

    @property
    def timestamp(self) -> str:
        """Return the timestamp as an ISO-8601 string."""
        return self._timestamp.isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert the audit event to a dictionary for logging/storage.

        Excludes sensitive data: no API keys, no authorization headers,
        no full secrets, no raw private content.

        Returns:
            A dict suitable for JSON serialization and structured logging.
        """

        # Build the event dict, carefully excluding sensitive fields
        event: dict[str, Any] = {
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "pipeline_stage": self.pipeline_stage.value if hasattr(self.pipeline_stage, "value") else self.pipeline_stage,
            "decision": self.decision,
            "reason_codes": self.reason_codes,
            "source_reference": self.source_reference,
            "latency_ms": self.latency,
            "error_status": self.error_status,
            "redacted": self.redacted,
        }

        # Add optional fields only when they have values
        if self.policy_id:
            event["policy_id"] = self.policy_id
        if self.policy_version:
            event["policy_version"] = self.policy_version
        if self.detector_model:
            event["detector_model"] = self.detector_model
        if self.diagnostics:
            # Sanitize diagnostics: remove any fields that look like secrets
            sanitized = self._sanitize_diagnostics(self.diagnostics)
            event["diagnostics"] = sanitized

        return event

    def _sanitize_diagnostics(self, diagnostics: dict[str, Any]) -> dict[str, Any]:
        """Sanitize diagnostics to exclude sensitive information.

        Removes or redacts:
        - API key values
        - Authorization headers
        - Full secret values
        - Raw private content

        Args:
            diagnostics: the diagnostics dict to sanitize.

        Returns:
            A sanitized copy of the diagnostics dict.
        """

        sanitized: dict[str, Any] = {}

        for key, value in diagnostics.items():
            # Skip known sensitive field names
            sensitive_keys = {
                "api_key",
                "authorization",
                "secret",
                "token",
                "credential",
                "password",
            }
            key_lower = key.lower()
            if any(sk in key_lower for sk in sensitive_keys):
                # Redact: keep the key but mask the value
                if isinstance(value, str):
                    sanitized[key] = "***REDACTED***"
                elif isinstance(value, dict):
                    sanitized[key] = self._sanitize_diagnostics(value)
                elif isinstance(value, list):
                    sanitized[key] = [
                        self._sanitize_diagnostics(item) if isinstance(item, dict) else item
                        for item in value
                    ]
                else:
                    sanitized[key] = value
            else:
                # Non-sensitive field - pass through
                if isinstance(value, dict):
                    sanitized[key] = self._sanitize_diagnostics(value)
                elif isinstance(value, list):
                    sanitized[key] = [
                        self._sanitize_diagnostics(item) if isinstance(item, dict) else item
                        for item in value
                    ]
                else:
                    sanitized[key] = value

        return sanitized


@dataclass
class AuditLogger:
    """Simple audit logger that stores events in a list.

    In production, this would write to a configurable storage backend
    (file, database, observability platform). For now, it maintains an
    in-memory list with privacy-preserving serialization.

    Does not build a separate observability platform.
    """

    _events: list[AuditEvent] = field(default_factory=list, init=False)

    def log(self, event: AuditEvent) -> None:
        """Log an audit event.

        Args:
            event: the AuditEvent to record.
        """

        self._events.append(event)

    def get_events(
        self,
        *,
        pipeline_stage: GuardrailStage | None = None,
        error_only: bool = False,
        request_id: str | None = None,
    ) -> list[AuditEvent]:
        """Retrieve logged events matching filter criteria.

        Args:
            pipeline_stage: filter by pipeline stage.
            error_only: when True, return only events with error_status=True.
            request_id: filter by request ID.

        Returns:
            list of matching AuditEvent instances.
        """

        events = self._events

        if pipeline_stage is not None:
            events = [e for e in events if e.pipeline_stage == pipeline_stage]

        if error_only:
            events = [e for e in events if e.error_status]

        if request_id is not None:
            events = [e for e in events if e.request_id == request_id]

        return events

    def get_event_count(self) -> int:
        """Return the total number of logged events."""
        return len(self._events)

    def clear(self) -> None:
        """Clear all logged events."""
        self._events.clear()