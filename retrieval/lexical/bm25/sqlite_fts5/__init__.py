"""BM25 over a SQLite FTS5 index.

The index is a relational content table plus an FTS5 virtual table, kept in
sync by SQLite triggers.  FTS5 provides the BM25 ranking; the raw score is
negative and the lower the value the better the match.
"""

from retrieval.lexical.bm25.sqlite_fts5.index import (
    CHUNKS_TABLE,
    DEFAULT_TOKENIZER,
    FTS_TABLE,
    LexicalRecord,
    SQLiteFTS5Index,
    fts5_available,
)
from retrieval.lexical.bm25.sqlite_fts5.retriever import (
    BM25Hit,
    BM25Retriever,
    to_sql_filter,
)
from retrieval.lexical.bm25.sqlite_fts5.tokenizer import (
    build_match_query,
    build_prefix_match_query,
    quote_term,
    tokenize_query,
)

__all__ = [
    "BM25Hit",
    "BM25Retriever",
    "CHUNKS_TABLE",
    "DEFAULT_TOKENIZER",
    "FTS_TABLE",
    "LexicalRecord",
    "SQLiteFTS5Index",
    "build_match_query",
    "build_prefix_match_query",
    "fts5_available",
    "quote_term",
    "to_sql_filter",
    "tokenize_query",
]
