"""Ollama provider for local Qwen model generation.

Supports the two manually selectable Qwen models (8B and 4B, 4-bit quantized).
Base URL configurable to localhost:11434 by default.
Lazy connection and model initialization.
Manual model selection, not automatic fallback to Groq.
"""

from __future__ import annotations

from generation.providers.ollama.provider import OllamaProvider
from generation.providers.ollama.qwen_4b import qwen_4b_configuration
from generation.providers.ollama.qwen_8b import qwen_8b_configuration

__all__ = ["OllamaProvider", "qwen_4b_configuration", "qwen_8b_configuration"]