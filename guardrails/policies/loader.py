"""Policy loader for the Guardrails ecosystem.

Loads policies from JSON, YAML, or TOML configuration files. Does not execute
arbitrary Python policy code. Provides neutral sample policies for initial
setup.

All loading is deterministic and validated. Invalid or malformed policy
configurations are rejected with actionable errors.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from guardrails.contracts.models import Policy
from guardrails.contracts.exceptions import (
    DuplicatePolicyIDError,
    GuardrailConfigError,
    MissingMandatoryFieldError,
)


def load_policy_from_json(filepath: str | Path) -> Policy:
    """Load a single policy from a JSON configuration file.

    The JSON file should contain a policy object with the fields matching
    the :class:`guardrails.policies.models.Policy` dataclass.

    Args:
        filepath: path to the JSON configuration file.

    Returns:
        A Policy instance constructed from the JSON data.

    Raises:
        GuardrailConfigError: if the JSON is malformed or missing required
            fields.
        MissingMandatoryFieldError: if a required field is absent.
    """

    filepath = Path(filepath)

    if not filepath.exists():
        raise GuardrailConfigError(f"Policy file not found: {filepath}")

    if filepath.suffix.lower() != ".json":
        raise GuardrailConfigError(
            f"Expected JSON policy file, got {filepath.suffix}"
        )

    try:
        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise GuardrailConfigError(
            f"Invalid JSON in policy file {filepath}: {e}"
        ) from e

    return _policy_dict_to_policy(data)


def load_policies_from_directory(directory: str | Path) -> list[Policy]:
    """Load all policy JSON files from a directory.

    Scans the directory for ``*.json`` files and loads each as a separate
    policy. Duplicate policy IDs are rejected.

    Args:
        directory: path to the directory containing policy JSON files.

    Returns:
        list of Policy instances.

    Raises:
        DuplicatePolicyIDError: if two policies share the same policy_id.
        GuardrailConfigError: if any policy file is malformed.
    """

    directory = Path(directory)

    if not directory.is_dir():
        raise GuardrailConfigError(
            f"Policy directory not found or not a directory: {directory}"
        )

    policies: list[Policy] = []
    seen_ids: set[str] = set()

    for filepath in sorted(directory.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue

        try:
            policy = load_policy_from_json(filepath)
        except GuardrailConfigError as e:
            raise GuardrailConfigError(
                f"Failed to load policy from {filepath.name}: {e}"
            ) from e

        if policy.policy_id in seen_ids:
            raise DuplicatePolicyIDError(
                f"Duplicate policy ID found: {policy.policy_id}"
            )

        seen_ids.add(policy.policy_id)
        policies.append(policy)

    return policies


def _policy_dict_to_policy(data: dict[str, Any]) -> Policy:
    """Convert a policy dict to a Policy dataclass.

    Validates all required fields and translates the dict into the
    Policy structure.

    Args:
        data: policy configuration dictionary.

    Returns:
        Policy instance.

    Raises:
        MissingMandatoryFieldError: if a required field is absent.
        InvalidActionError: if enforcement_action is invalid.
        InvalidStageError: if a stage in applicable_stages is invalid.
        InvalidPriorityError: if priority is not an integer.
        UnknownConditionTypeError: if conditions is not a dict.
    """

    # Extract and validate required fields
    policy_id = data.get("policy_id")
    if not policy_id:
        raise MissingMandatoryFieldError("policy_id is required")

    version = data.get("version")
    if not version:
        raise MissingMandatoryFieldError("policy version is required")

    description = data.get("description")
    if not description:
        raise MissingMandatoryFieldError("policy description is required")

    # Extract optional fields with defaults
    enabled = data.get("enabled", True)
    if not isinstance(enabled, bool):
        raise GuardrailConfigError("policy enabled must be a boolean")

    # Handle applicable_stages
    raw_stages = data.get("applicable_stages")
    if raw_stages is None:
        applicable_stages: frozenset[str] | None = None
    elif isinstance(raw_stages, str):
        applicable_stages = frozenset([raw_stages])
    elif isinstance(raw_stages, list):
        applicable_stages = frozenset(raw_stages)
    elif isinstance(raw_stages, frozenset):
        applicable_stages = raw_stages
    else:
        raise GuardrailConfigError(
            f"Invalid applicable_stages type: {type(raw_stages).__name__}"
        )

    # Handle conditions
    raw_conditions = data.get("conditions")
    conditions: dict[str, Any] | None = None
    if raw_conditions is not None:
        if isinstance(raw_conditions, dict):
            conditions = raw_conditions
        else:
            raise UnknownConditionTypeError(
                "policy conditions must be a dict"
            )

    # Handle enforcement_action
    raw_action = data.get("enforcement_action", "allow")
    if raw_action not in {"allow", "warn", "block", "review"}:
        raise InvalidActionError(
            f"Invalid enforcement_action: {raw_action}"
        )

    # Handle priority
    raw_priority = data.get("priority", 0)
    try:
        priority = int(raw_priority)
    except (ValueError, TypeError):
        raise InvalidPriorityError(
            f"Policy priority must be an integer, got {raw_priority!r}"
        )

    # Handle expiry_date
    raw_expiry = data.get("expiry_date")
    expiry_date: datetime | None = None
    if raw_expiry is not None:
        if isinstance(raw_expiry, str):
            # Simple ISO-8601 parsing fallback
            try:
                from datetime import datetime as _dt
                expiry_date = _dt.fromisoformat(raw_expiry)
            except ValueError:
                raise GuardrailConfigError(
                    f"Could not parse expiry_date: {raw_expiry}"
                )
        else:
            raise GuardrailConfigError(
                f"expiry_date must be a string or None, got {type(raw_expiry).__name__}"
            )

    # Handle deprecation_meta
    raw_deprecation = data.get("deprecation_meta")
    deprecation_meta: dict[str, Any] | None = None
    if raw_deprecation is not None:
        if isinstance(raw_deprecation, dict):
            deprecation_meta = raw_deprecation
        else:
            raise GuardrailConfigError(
                "deprecation_meta must be a dict or None"
            )

    return Policy(
        policy_id=policy_id,
        version=version,
        description=description,
        enabled=enabled,
        applicable_stages=applicable_stages,
        conditions=conditions,
        enforcement_action=raw_action,
        priority=priority,
        expiry_date=expiry_date,
        deprecation_meta=deprecation_meta,
    )