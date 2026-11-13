"""Privacy rules and configuration for the Guardrails ecosystem.

Defines the detection categories, actions, and rule evaluations for the
privacy detection layer.

Initial detection categories:
- email: email addresses
- phone: phone numbers
- api_key: common API key formats (sk-..., ghp_...)
- bearer_token: Bearer token patterns
- access_token: access token patterns
- cloud_credential: common cloud credential patterns (AKIA..., GCE-, aws_)
- private_key: private key patterns (-----BEGIN PRIVATE KEY-----)
- password_field: password-like credential fields

Actions:
- DETECT: identify sensitive patterns (default).
- WARN: flag but allow through with warning.
- MASK: produce derived copy with sensitive values masked.
- BLOCK: block the request/review.
- REVIEW: send for human review.

Rules are evaluated in order; the first matching action is applied.
Per-category overrides are supported.
"""

from __future__ import annotations

from typing import Any

from guardrails.privacy.detectors import PrivacyDetector, PrivacyRules, DetectorResult


#: Default privacy rules with default action of "detect"
DEFAULT_PRIVACY_RULES = PrivacyRules(default_action="detect")

#: Default privacy detector with all categories enabled
DEFAULT_PRIVACY_DETECTOR = PrivacyDetector(enabled_categories=None)

#: Category-friendly display names
CATEGORY_NAMES: dict[str, str] = {
    "email": "email addresses",
    "phone": "phone numbers",
    "api_key": "API keys",
    "bearer_token": "bearer tokens",
    "access_token": "access tokens",
    "cloud_credential": "cloud credentials",
    "private_key": "private keys",
    "password_field": "password fields",
}


def get_category_name(category: str) -> str:
    """Get a human-readable name for a detection category.

    Args:
        category: the category key.

    Returns:
        Human-readable name, or the category key if not found.
    """

    return CATEGORY_NAMES.get(category, category)


def default_rules() -> PrivacyRules:
    """Provide default privacy rules for testing.

    Returns a PrivacyRules instance with the default action of "detect"
    and no per-category overrides.
    """

    return PrivacyRules(default_action="detect")


def strict_rules() -> PrivacyRules:
    """Provide strict privacy rules for sensitive environments.

    Returns a PrivacyRules instance with default action of "block",
    blocking detection of all sensitive categories by default.
    """

    return PrivacyRules(default_action="block")


def relaxed_rules() -> PrivacyRules:
    """Provide relaxed privacy rules for non-sensitive environments.

    Returns a PrivacyRules instance with default action of "warn",
    allowing content through with a warning when sensitive patterns are
    detected.
    """

    return PrivacyRules(default_action="warn")