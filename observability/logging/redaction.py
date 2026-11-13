"""Configurable redaction for sensitive data in logs, spans, and errors.

Redaction is on by default and is deliberately aggressive: credentials,
tokens, authorization headers, emails, and high-entropy key-like values
never leave the process unmasked. Forbidden content such as prompts or
document payloads is dropped entirely by the callers; anything that slips
through is still pattern-redacted here.

Opt-out is possible per-redactor (``enabled=False``) for tests, but is
not exposed as a runtime configuration flag: production telemetry should
not have a switch that disables secret scrubbing.

Diagnostic mode (``PROMERAG_DIAGNOSTIC_MODE=true``) only controls
verbosity/content opt-ins elsewhere; it does not weaken this module.
"""
from __future__ import annotations

import logging
import re
import threading
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ..contracts.models import RedactionRule

logger = logging.getLogger("prome_rag.logging.redaction")

_REDACTED = "***REDACTED***"
_EMAIL_REPLACEMENT = "***REDACTED-EMAIL***"

DEFAULT_REPLACEMENT = _REDACTED

_DEFAULT_RULE_SPECS = [
    # PEM private keys (multi-line)
    (r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----.*?-----END (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----", _REDACTED, "private-key"),
    # JWT tokens
    (r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b", _REDACTED, "jwt"),
    # Authorization header schemes
    (r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._\-+/=]{8,}", r"\1 ***REDACTED***", "authorization-header"),
    # AWS access key ids
    (r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b", _REDACTED, "aws-access-key-id"),
    (r"(?i)\baws[_-]?secret[_-]?access[_-]?key\b\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{20,}", _REDACTED, "aws-secret"),
    # GitHub tokens
    (r"\bgh[pousr]_[A-Za-z0-9]{20,}\b", _REDACTED, "github-token"),
    (r"\bgithub_pat_[A-Za-z0-9_]{20,}\b", _REDACTED, "github-fine-grained-pat"),
    # Slack tokens
    (r"\bxox[baprs]-[A-Za-z0-9\-]{8,}\b", _REDACTED, "slack-token"),
    # Google API keys
    (r"\bAIza[0-9A-Za-z\-_]{20,}\b", _REDACTED, "google-api-key"),
    # OpenAI / Anthropic style keys
    (r"\bsk-(?:ant-|proj-|live-)?[A-Za-z0-9_\-]{8,}\b", _REDACTED, "provider-api-key"),
    # Generic key=value secrets (api_key=..., password: ..., secret="...", token:'...')
    # Note: bare "authorization" is intentionally absent — the bearer/basic rule above handles it.
    (r"(?i)\b(api[_-]?key|apikey|token|access[_-]?token|auth[_-]?token|password|passwd|pwd|secret|credential|client[_-]?secret|private[_-]?key)\b(\s*[:=]\s*)['\"]?[^\s,'\"}&]{4,}", r"\1\2***REDACTED***", "key-value-secret"),
    # Email addresses (PII)
    (r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", _EMAIL_REPLACEMENT, "email"),
]

_SENSITIVE_KEY_RE = re.compile(
    r"(?i)(password|passwd|pwd|secret|token|api[_-]?key|apikey|authorization|credential|private[_-]?key|access[_-]?key|client[_-]?secret)"
)


def _compile(rule_specs: Iterable) -> list[tuple[re.Pattern, str]]:
    compiled: list[tuple[re.Pattern, str]] = []
    for spec in rule_specs:
        pattern, replacement = spec[0], spec[1]
        compiled.append((re.compile(pattern, re.DOTALL), replacement))
    return compiled


def _default_specs() -> list[tuple[str, str, str]]:
    return list(_DEFAULT_RULE_SPECS)


class Redactor:
    """Pattern- and key-based redactor for arbitrary Python structures."""

    def __init__(
        self,
        rules: Sequence[RedactionRule] | None = None,
        *,
        extra_rules: Sequence[RedactionRule] | None = None,
        enabled: bool = True,
    ) -> None:
        specs = _default_specs()
        if rules is not None:  # explicit replacement of the default rule set
            specs = [(rule.pattern, rule.replacement, rule.description) for rule in rules]
        elif extra_rules:
            specs = specs + [(rule.pattern, rule.replacement, rule.description) for rule in extra_rules]
        self._enabled = enabled
        self._patterns = _compile(specs)

    # -- text --------------------------------------------------------------

    def redact_text(self, text: str) -> str:
        if not self._enabled or not text:
            return text
        result = text
        for pattern, replacement in self._patterns:
            result = pattern.sub(replacement, result)
        return result

    # -- structured values ---------------------------------------------------

    def redact_value(self, value: Any, *, max_depth: int = 6) -> Any:
        if not self._enabled:
            return value
        return self._redact(value, depth=0, max_depth=max_depth)

    def redact_mapping(self, mapping: Mapping[str, Any]) -> dict[str, Any]:
        if not self._enabled:
            return dict(mapping)
        return self._redact_mapping(mapping, depth=0, max_depth=6)

    # -- internals -----------------------------------------------------------

    def _redact(self, value: Any, *, depth: int, max_depth: int) -> Any:
        if depth >= max_depth:
            return "***"
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, Mapping):
            return self._redact_mapping(value, depth=depth, max_depth=max_depth)
        if isinstance(value, (list, tuple)):
            return [self._redact(item, depth=depth + 1, max_depth=max_depth) for item in value]
        if isinstance(value, (set, frozenset)):
            return [self._redact(item, depth=depth + 1, max_depth=max_depth) for item in value]
        if isinstance(value, bytes):
            return _REDACTED
        return value

    def _redact_mapping(self, mapping: Mapping[str, Any], *, depth: int, max_depth: int) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in mapping.items():
            key_str = str(key)
            if _SENSITIVE_KEY_RE.search(key_str):
                result[key_str] = _REDACTED
                continue
            result[key_str] = self._redact(value, depth=depth + 1, max_depth=max_depth)
        return result


_default_lock = threading.Lock()
_default_redactor: Redactor | None = None


def get_default_redactor() -> Redactor:
    global _default_redactor
    with _default_lock:
        if _default_redactor is None:
            _default_redactor = Redactor()
        return _default_redactor


def configure_default_redactor(
    *,
    extra_rules: Sequence[RedactionRule] | None = None,
    rules: Sequence[RedactionRule] | None = None,
    enabled: bool | None = None,
) -> Redactor:
    global _default_redactor
    with _default_lock:
        _default_redactor = Redactor(
            rules=rules,
            extra_rules=extra_rules,
            enabled=True if enabled is None else enabled,
        )
        return _default_redactor


def reset_default_redactor() -> None:
    global _default_redactor
    with _default_lock:
        _default_redactor = None


def redact_text(text: str) -> str:
    return get_default_redactor().redact_text(text)


def redact_value(value: Any) -> Any:
    return get_default_redactor().redact_value(value)


def redact_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return get_default_redactor().redact_mapping(mapping)
