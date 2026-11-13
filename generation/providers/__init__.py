"""Providers package for the Prome RAG Plugins generation ecosystem.

Contains independent generation provider implementations for Ollama and Groq.
Each provider is independently testable and follows the shared generation contracts.
"""

from __future__ import annotations

from generation.providers.groq.provider import GroqProvider
from generation.providers.ollama.provider import OllamaProvider

__all__ = ["GroqProvider", "OllamaProvider"]