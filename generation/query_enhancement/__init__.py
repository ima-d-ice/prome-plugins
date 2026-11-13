"""Query enhancement for the Prome RAG Plugins generation ecosystem.

Optional query enhancement as an independent component.
Default: QUERY_ENHANCEMENT_ENABLED=false

When enabled:
- Uses an injected generation provider.
- Requires explicit provider/model selection.
- Does NOT call Prompt Guard as the enhancer.
- Preserves the original query.
- Returns the enhanced query separately.
- Never mutates the original user message.
- Makes the enhanced query available to retrieval without requiring
  retrieval code in this package.
"""

from __future__ import annotations

from generation.query_enhancement.enhancer import QueryEnhancer

__all__ = ["QueryEnhancer"]