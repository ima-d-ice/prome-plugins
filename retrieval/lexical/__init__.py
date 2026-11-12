"""Lexical retrieval: retrieval without embeddings.

The lexical channel matches the *words* of a query against the words of a
document, which makes it precise for exact terms, error messages and API
names -- the cases a dense model tends to blur.
"""

from retrieval.lexical.bm25 import BM25Retriever, SQLiteFTS5Index

__all__ = ["BM25Retriever", "SQLiteFTS5Index"]
