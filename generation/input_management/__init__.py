"""Input management for the Prome RAG Plugins generation ecosystem.

Provides input and token capping, token counting, and context budgeting
independently from provider implementations.

Supports configurable:
- Maximum request body size.
- Maximum user query tokens.
- Maximum conversation history tokens.
- Maximum retrieval-context tokens.
- Maximum model input tokens.
- Maximum completion tokens.
- Maximum context window.
- Safety classifier input limits.
"""

from __future__ import annotations

from generation.input_management.caps import Caps
from generation.input_management.context_budget import ContextBudget
from generation.input_management.token_counter import TokenCounter, get_token_counter

__all__ = ["Caps", "ContextBudget", "TokenCounter", "get_token_counter"]