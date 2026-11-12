"""Deduplication of retrieved items.

Two channels can return the same chunk.  The deduplication keeps one entry
per chunk id -- the best ranked one -- and records the channels that also
returned it.
"""

from retrieval.postprocessing.deduplication.deduplicator import deduplicate

__all__ = ["deduplicate"]
