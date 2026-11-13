"""Redaction of secrets and PII."""
from __future__ import annotations

from observability.logging.redaction import (
    Redactor,
    get_default_redactor,
    redact_mapping,
    redact_text,
    redact_value,
    reset_default_redactor,
)


def test_openai_style_key_redacted():
    assert redact_text("key=sk-abc1234567890def") == "key=***REDACTED***"
    assert "abc1234567890def" not in redact_text("sk-proj-abcdefghijklmnop")


def test_bearer_and_basic_headers_redacted():
    assert "Bearer ***REDACTED***" in redact_text("Authorization: Bearer abcdefgh12345678")
    assert "Basic ***REDACTED***" in redact_text("Authorization: Basic dXNlcjpwYXNzd29yZA==")


def test_aws_and_github_keys_redacted():
    assert "AKIA" not in redact_text("aws key AKIAIOSFODNN7EXAMPLE here")
    assert "ghp_" not in redact_text("token ghp_a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9")
    assert "github_pat_" not in redact_text("github_pat_abcdefghijklmnopqrstuvwxyzabcd")


def test_key_value_secrets_redacted():
    assert "hunter2" not in redact_text("password=hunter2")
    assert "s3cr3t!!" not in redact_text("secret: s3cr3t!!")
    assert "token=abcdef123456" not in redact_text("token=abcdef123456")


def test_email_addresses_redacted():
    assert "user@example.com" not in redact_text("contact user@example.com now")
    assert "***REDACTED-EMAIL***" in redact_text("contact user@example.com now")


def test_private_key_block_redacted():
    blob = (
        "-----BEGIN PRIVATE KEY-----\n"
        "MIIBogIBAAJBAKsecretmaterialthatshouldneverlog\n"
        "-----END PRIVATE KEY-----"
    )
    redacted = redact_text(blob)
    assert "secretmaterialthatshouldneverlog" not in redacted


def test_redact_mapping_masks_sensitive_keys():
    mapping = {
        "password": "hunter2",
        "api_key": "sk-abc1234567890def",
        "authorization": "Bearer abcdefgh12345678",
        "note": "public sk-abc1234567890def value",
        "nested": {"client_secret": "xyz123456789", "safe": "ok"},
        "items": ["email me at person@example.com", "fine"],
    }
    result = redact_mapping(mapping)
    assert result["password"] == "***REDACTED***"
    assert result["api_key"] == "***REDACTED***"
    assert result["authorization"] == "***REDACTED***"
    assert "sk-abc1234567890def" not in str(result)
    assert result["nested"]["client_secret"] == "***REDACTED***"
    assert result["nested"]["safe"] == "ok"
    assert "person@example.com" not in str(result)


def test_redact_value_handles_nested_structures():
    value = {"outer": ["sk-abc1234567890def", {"inner": "password=hunter2"}]}
    result = redact_value(value)
    assert "abc1234567890def" not in str(result)
    assert "hunter2" not in str(result)


def test_disable_redactor_passthrough():
    redactor = Redactor(enabled=False)
    assert redactor.redact_text("sk-abc1234567890def") == "sk-abc1234567890def"


def test_extra_rules_extend_defaults():
    from observability.contracts.models import RedactionRule

    redactor = Redactor(extra_rules=[RedactionRule(pattern=r"CONFID-\d+", replacement="CONFID-***")])
    assert redactor.redact_text("ref CONFID-12345 token sk-abc1234567890def") == "ref CONFID-*** token ***REDACTED***"


def test_default_redactor_is_shared_and_resettable():
    first = get_default_redactor()
    assert get_default_redactor() is first
    reset_default_redactor()
    second = get_default_redactor()
    assert second is not first


def test_encrypted_env_style_secret_values():
    assert "s3cr3t" not in redact_text("API_KEY: s3cr3t-value-here\npassword: s3cr3t")
    assert "s3cr3t" not in redact_text("token: 's3cr3t-value-here'")
