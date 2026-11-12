"""Post-processing of retrieval results.

Post-processing never changes *what* was retrieved, only how much context
travels with it.
"""

from retrieval.postprocessing.context_expansion import (
    ContextExpansionConfig,
    ContextExpander,
    ExpandedContext,
)

__all__ = ["ContextExpansionConfig", "ContextExpander", "ExpandedContext"]
