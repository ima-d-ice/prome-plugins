"""Context expansion for retrieval results.

Context expansion attaches the material that explains a match without
replacing it: the section a passage belongs to, and optionally its
neighbours.
"""

from retrieval.postprocessing.context_expansion.expander import (
    ContextExpansionConfig,
    ContextExpander,
    ExpandedContext,
)

__all__ = ["ContextExpansionConfig", "ContextExpander", "ExpandedContext"]
