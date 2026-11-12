"""Coordination of the dense and the lexical index.

The two indexes hold the same chunks.  There is no distributed transaction
between them, so the coordinator reports exactly which backend was updated
and offers a safe retry of the one that was not.
"""

from retrieval.indexing.coordinator import IndexingCoordinator, IndexingReport

__all__ = ["IndexingCoordinator", "IndexingReport"]
