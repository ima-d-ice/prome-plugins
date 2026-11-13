"""Rate limiting and token accounting for the Prome RAG Plugins generation ecosystem.

Provides model-aware, per-key rate limiting and token accounting across
a 12-key rotation pool. Tracks RPM, RPD, TPM, and TPD independently.

The contracts are dependency-free and independent of any HTTP SDK or
provider implementation. Concrete implementations (Groq-specific) use
httpx or requests for header semantics.
"""

from __future__ import annotations

from generation.rate_limiting.key_rotation import KeyRotationPool, KeyRotationRecord
from generation.rate_limiting.limiter import RateLimiter, RateLimitRecord
from generation.rate_limiting.response_headers import ResponseHeaderParser
from generation.rate_limiting.token_budget import TokenBudget

__all__ = [
    "KeyRotationPool",
    "KeyRotationRecord",
    "RateLimitRecord",
    "RateLimiter",
    "ResponseHeaderParser",
    "TokenBudget",
]