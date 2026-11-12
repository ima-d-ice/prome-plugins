"""Merging of retrieved items with their adjacent chunks.

Only consecutive chunks of the same document are merged, and the merged
text never exceeds the configured size limit.  The original evidence
references stay intact.
"""

from retrieval.postprocessing.adjacent_merge.merger import (
    AdjacentMergeConfig,
    AdjacentMerger,
    AdjacentMergeReport,
    build_context_chunk,
)

__all__ = [
    "AdjacentMergeConfig",
    "AdjacentMerger",
    "AdjacentMergeReport",
    "build_context_chunk",
]
