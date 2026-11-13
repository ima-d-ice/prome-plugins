"""Routing package for the Prome RAG Plugins generation ecosystem.

Contains the generation router, fallback logic, and model selection
components that direct requests to the appropriate provider and model.
"""

from __future__ import annotations

from generation.routing.fallback import FallbackManager
from generation.routing.generation_router import GenerationRouter
from generation.routing.model_selection import ModelSelector

__all__ = ["FallbackManager", "GenerationRouter", "ModelSelector"]