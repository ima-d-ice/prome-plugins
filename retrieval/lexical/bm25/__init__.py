"""BM25 lexical retrieval.

The initial backend is SQLite FTS5, which provides the BM25 ranking as a
built-in function.  It is local, persistent and needs no model.
"""

from retrieval.lexical.bm25.sqlite_fts5 import (
    BM25Hit,
    BM25Retriever,
    LexicalRecord,
    SQLiteFTS5Index,
    fts5_available,
)

__all__ = [
    "BM25Hit",
    "BM25Retriever",
    "LexicalRecord",
    "SQLiteFTS5Index",
    "fts5_available",
]
