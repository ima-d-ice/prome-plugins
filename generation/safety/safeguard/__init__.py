"""Safeguard safety sub-package.

Contains the SafeguardEvaluator and SafeguardVerdict for optional
policy-based content safety evaluation using openai/gpt-oss-safeguard-20b.
"""

from __future__ import annotations

from generation.safety.safeguard.evaluator import (
    DEFAULT_MODE,
    SafeguardEvaluator,
    SafeguardVerdict,
)

__all__ = ["SafeguardEvaluator", "SafeguardVerdict"]