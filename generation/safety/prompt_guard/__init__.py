"""Prompt Guard safety sub-package.

Contains the PromptGuardClassifier and related types for adaptive prompt
injection and jailbreak detection.
"""

from __future__ import annotations

from generation.safety.prompt_guard.classifier import (
    GuardResult,
    PromptGuardClassifier,
)

__all__ = ["GuardResult", "PromptGuardClassifier"]