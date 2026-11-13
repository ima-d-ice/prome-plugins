"""Audit storage interface for the Guardrails ecosystem.

Defines the interface for audit event storage and retention configuration.

Do not build a separate observability platform. This interface is designed
to be implemented by whatever storage backend the deployment chooses
(file, database, observability platform), keeping the guardrails package
independently usable.
"""

from __future__ import annotations

from typing import Any

from guardrails.audit.events import AuditEvent


class AuditStorageInterface:
    """Interface for audit event storage.

    Implementations should provide durable storage of AuditEvent instances
    with configurable retention policies. The interface keeps the guardrails
    package independently usable without coupling to a specific storage
    technology.

    Do not build a separate observability platform. This is a minimal
    interface for persisting audit events.
    """

    def store(self, event: AuditEvent) -> bool:
        """Store a single audit event.

        Args:
            event: the AuditEvent to store.

        Returns:
            True if the event was successfully stored, False otherwise.
        """

        raise NotImplementedError(
        "Subclasses must implement the store method"
        )

    def store_batch(self, events: list[AuditEvent]) -> list[bool]:
        """Store a batch of audit events.

        Args:
            events: list of AuditEvent instances to store.

        Returns:
            list of booleans indicating success for each event.
        """

        raise NotImplementedError(
        "Subclasses must implement the store_batch method"
        )

    def set_retention(self, **kwargs: Any) -> None:
        """Configure retention policy for stored events.

        Args:
            **kwargs: retention configuration parameters. Supported keys
                depend on the implementation (e.g., max_age_days, max_events,
                keep_days, etc.).
        """

        raise NotImplementedError(
        "Subclasses must implement the set_retention method"
        )

    def get_retention(self) -> dict[str, Any]:
        """Get the current retention policy configuration.

        Returns:
            dict with the current retention settings.
        """

        raise NotImplementedError(
        "Subclasses must implement the get_retention method"
        )
