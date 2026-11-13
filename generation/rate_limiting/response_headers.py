"""Response header parser for Groq API rate-limit headers.

Verifies header semantics against current Groq documentation to update
the active key's internal ledger accurately.

Supports parsing of:
- RateLimit-Remaining, RateLimit-Retry-After
- Token consumption headers if provided by Groq
"""

from __future__ import annotations

from typing import Any


class ResponseHeaderParser:
    """Parse Groq API response headers for rate-limit metadata."""

    #: Header name patterns and their expected semantics
    HEADER_PATTERNS: dict[str, str] = {
        "rate_limit_remaining": [
            "rateLimit-remaining",
            "x-ratelimit-remaining",
            "rate-limit-remaining",
        ],
        "rate_limit_reset": [
            "rateLimit-reset",
            "x-ratelimit-reset",
            "rate-limit-reset",
        ],
        "rate_limit_total": [
            "rateLimit-total",
            "x-ratelimit-total",
            "rate-limit-total",
        ],
        "retry_after": [
            "retry-after",
            "retry-after-seconds",
        ],
    }

    @classmethod
    def parse_rate_limit_headers(
        cls, headers: dict[str, str] | None,
    ) -> dict[str, Any]:
        """Parse rate-limit relevant headers from an API response.

        Args:
            headers: the HTTP response headers dict.

        Returns:
            dict with parsed rate-limit metadata:
            - remaining: remaining requests/tokens
            - total: total quota
            - reset_time: unix timestamp when the window resets
            - retry_after: suggested wait time in seconds
        """
        if not headers:
            return {
                "remaining": None,
                "total": None,
                "reset_time": None,
                "retry_after": None,
            }

        result: dict[str, Any] = {
            "remaining": None,
            "total": None,
            "reset_time": None,
            "retry_after": None,
        }

        # Normalize header keys for lookup (case-insensitive)
        normalized = {k.lower(): v for k, v in headers.items()}

        # Parse remaining
        for pattern in cls.HEADER_PATTERNS["rate_limit_remaining"]:
            for key, val in normalized.items():
                if any(pattern in key for pattern in [pattern]):
                    try:
                        result["remaining"] = int(val)
                    except (ValueError, TypeError):
                        pass
                    break

        # Parse total
        for pattern in cls.HEADER_PATTERNS["rate_limit_total"]:
            for key, val in normalized.items():
                if any(pattern in key for pattern in [pattern]):
                    try:
                        result["total"] = int(val)
                    except (ValueError, TypeError):
                        pass
                    break

        # Parse reset time
        for pattern in cls.HEADER_PATTERNS["rate_limit_reset"]:
            for key, val in normalized.items():
                if any(pattern in key for pattern in [pattern]):
                    try:
                        result["reset_time"] = int(val)
                    except (ValueError, TypeError):
                        pass
                    break

        # Parse retry-after
        for pattern in cls.HEADER_PATTERNS["retry_after"]:
            for key, val in normalized.items():
                if any(pattern in key for pattern in [pattern]):
                    try:
                        result["retry_after"] = int(val)
                    except (ValueError, TypeError):
                        # Try float
                        try:
                            result["retry_after"] = float(val)
                        except (ValueError, TypeError):
                            pass
                    break

        return result