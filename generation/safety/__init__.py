"""Safety ecosystem for the Prome RAG Plugins generation pipeline.

Includes adaptive prompt injection/jailbreak detection (Prompt Guard)
and optional policy-based safeguard evaluation (GPT-OSS Safeguard).

The contracts are independent of any specific model or HTTP SDK.
"""

from __future__ import annotations

from generation.safety.prompt_guard.classifier import PromptGuardClassifier
from generation.safety.safeguard.evaluator import SafeguardEvaluator

__all__ = ["PromptGuardClassifier", "SafeguardEvaluator"]