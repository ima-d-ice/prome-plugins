"""The lexical retrieval channel: BM25 over a SQLite FTS5 index.

FTS5 provides the BM25 ranking as the ``bm25()`` function.  Its score has
the opposite sign convention of a cosine similarity: it is **negative** and
the *lower* the value, the better the match.  The raw score is therefore
kept as it is and the direction is recorded explicitly, so a BM25 score is
never compared with a dense score.

The channel applies the *same* metadata filter AST as the dense channel
(:class:`~vector_stores.contracts.filters.MetadataFilter`), translated to
SQL.  A filter the lexical backend cannot express is reported, never
ignored.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from retrieval.contracts.capabilities import RetrievalCapabilities
from retrieval.contracts.exceptions import (
    LexicalIndexError,
    RetrievalConfigError,
    UnsupportedRetrievalCapabilityError,
)
from retrieval.contracts.models import (
    RetrievedItem,
    RetrievalQuery,
    RetrievalResponse,
)
from retrieval.dense.retriever import filter_fields, required_filter_ops
from retrieval.lexical.bm25.sqlite_fts5.index import (
    CHUNKS_TABLE,
    FTS_TABLE,
    SQLiteFTS5Index,
)
from retrieval.lexical.bm25.sqlite_fts5.tokenizer import build_match_query
from vector_stores.contracts.filters import (
    FIELD_OPS,
    FilterOp,
    MetadataFilter,
    eq,
    is_in,
)
from vector_stores.contracts.payload import decode_block_refs
from vector_stores.contracts.models import ScoreDirection


@dataclass(frozen=True)
class BM25Hit:
    """One hit of the lexical search."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any]
    provenance: Any
    rank: int


def _metadata_expression(field: str) -> str:
    """Return the SQL expression that reads the metadata field ``field``."""
    if not re_match_identifier(field):
        raise UnsupportedRetrievalCapabilityError(
            f"the lexical index can only filter on simple field names, got {field!r}"
        )
    return f"json_extract({CHUNKS_TABLE}.metadata, '$.{field}')"


def re_match_identifier(field: str) -> bool:
    """Return whether ``field`` is a plain SQL identifier."""
    return bool(field) and field.replace("_", "a").isalnum() and not field[0].isdigit()


def to_sql_filter(node: MetadataFilter) -> tuple[str, list[Any]]:
    """Translate a portable filter into SQL and its parameters.

    The values are always bound as parameters; only validated identifiers and
    the fixed ``json_extract`` calls end up in the SQL text.

    Raises:
        UnsupportedRetrievalCapabilityError: when the filter uses an
            operation or a field the lexical index cannot express.
    """
    op = node.op
    if op is FilterOp.AND:
        parts, params = [], []
        for child in node.children:
            sql, child_params = to_sql_filter(child)
            parts.append(sql)
            params.extend(child_params)
        return "(" + " AND ".join(parts) + ")", params
    if op is FilterOp.OR:
        parts, params = [], []
        for child in node.children:
            sql, child_params = to_sql_filter(child)
            parts.append(sql)
            params.extend(child_params)
        return "(" + " OR ".join(parts) + ")", params
    if op is FilterOp.NOT:
        sql, params = to_sql_filter(node.children[0])
        return f"NOT ({sql})", params

    if node.field == "doc_id":
        column = f"{CHUNKS_TABLE}.doc_id"
    elif node.field == "chunk_id":
        column = f"{CHUNKS_TABLE}.chunk_id"
    else:
        column = _metadata_expression(node.field)

    if op is FilterOp.EQ:
        return f"{column} = ?", [node.value]
    if op is FilterOp.NE:
        return f"{column} <> ?", [node.value]
    if op is FilterOp.IN:
        values = list(node.value)
        return f"{column} IN ({', '.join('?' * len(values))})", values
    if op is FilterOp.NOT_IN:
        values = list(node.value)
        return f"{column} NOT IN ({', '.join('?' * len(values))})", values
    if op is FilterOp.GT:
        return f"{column} > ?", [node.value]
    if op is FilterOp.GTE:
        return f"{column} >= ?", [node.value]
    if op is FilterOp.LT:
        return f"{column} < ?", [node.value]
    if op is FilterOp.LTE:
        return f"{column} <= ?", [node.value]
    if op is FilterOp.EXISTS:
        return (
            f"{column} IS NOT NULL" if node.value else f"{column} IS NULL"
        ), []
    raise UnsupportedRetrievalCapabilityError(
        f"the lexical index does not support the filter operation {op.value!r}"
    )


class BM25Retriever:
    """Lexical retrieval with BM25 over a SQLite FTS5 index.

    Args:
        index: the lexical index to search.
        name: the identifier of the channel.
    """

    name = "bm25"

    def __init__(self, index: SQLiteFTS5Index, *, name: str = "bm25") -> None:
        self.index = index
        self.name = name

    # ------------------------------------------------------------------
    @property
    def capabilities(self) -> RetrievalCapabilities:
        """Return the capabilities of the lexical channel."""
        return RetrievalCapabilities(
            dense=False,
            lexical=True,
            fusion=False,
            metadata_filtering=True,
            filter_ops=frozenset(FIELD_OPS),
            details={
                "backend": "sqlite-fts5",
                "tokenizer": self.index.tokenizer,
                "score": "bm25 (lower is better)",
            },
        )

    # ------------------------------------------------------------------
    def search(
        self,
        text: str,
        *,
        limit: int,
        filter: MetadataFilter | None = None,
        doc_ids: Sequence[str] | None = None,
    ) -> list[BM25Hit]:
        """Return the chunks that match ``text``, best first.

        Args:
            text: the query text.
            limit: the maximum number of hits.
            filter: the metadata filter every candidate must satisfy.
            doc_ids: an optional restriction to a set of documents.
        """
        if limit < 1:
            raise RetrievalConfigError(f"the search limit must be >= 1, got {limit}")
        match = build_match_query(text)
        if not match:
            # A query without a searchable term matches nothing.  It must
            # never degrade into "match everything".
            return []

        clauses = [f"{FTS_TABLE} MATCH ?"]
        params: list[Any] = [match]
        if filter is not None:
            sql, filter_params = to_sql_filter(filter)
            clauses.append(sql)
            params.extend(filter_params)
        if doc_ids:
            clauses.append(
                f"{CHUNKS_TABLE}.doc_id IN ({', '.join('?' * len(doc_ids))})"
            )
            params.extend(list(doc_ids))

        sql = (
            f"SELECT {CHUNKS_TABLE}.chunk_id AS chunk_id, "
            f"{CHUNKS_TABLE}.doc_id AS doc_id, "
            f"{CHUNKS_TABLE}.text AS text, "
            f"{CHUNKS_TABLE}.metadata AS metadata, "
            f"{CHUNKS_TABLE}.provenance AS provenance, "
            f"bm25({FTS_TABLE}) AS bm25_score "
            f"FROM {FTS_TABLE} JOIN {CHUNKS_TABLE} "
            f"ON {CHUNKS_TABLE}.rowid = {FTS_TABLE}.rowid "
            f"WHERE {' AND '.join(clauses)} "
            f"ORDER BY bm25_score ASC LIMIT ?"
        )
        params.append(limit)

        try:
            rows = self.index.connection.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise LexicalIndexError(f"the lexical search failed: {exc}") from exc

        hits: list[BM25Hit] = []
        for rank, row in enumerate(rows, start=1):
            hits.append(
                BM25Hit(
                    chunk_id=str(row["chunk_id"]),
                    doc_id=str(row["doc_id"]),
                    text=str(row["text"]),
                    score=float(row["bm25_score"]),
                    metadata=_decode_metadata(row["metadata"]),
                    provenance=_decode_provenance(row["provenance"]),
                    rank=rank,
                )
            )
        return hits

    # ------------------------------------------------------------------
    def retrieve(self, query: RetrievalQuery) -> RetrievalResponse:
        """Retrieve the passages that answer ``query``."""
        if not isinstance(query, RetrievalQuery):
            raise RetrievalConfigError(
                f"the query must be a RetrievalQuery, got {type(query).__name__}"
            )
        if not query.text.strip():
            raise RetrievalConfigError("the query text must not be empty")
        hits = self.search(
            query.text,
            limit=query.effective_lexical_limit,
            filter=query.filters,
            doc_ids=query.doc_ids,
        )
        return RetrievalResponse(
            results=[self.to_item(hit) for hit in hits],
            query=query,
            strategy=self.name,
        )

    def to_item(self, hit: BM25Hit) -> RetrievedItem:
        """Convert a lexical hit into a standardized item.

        The BM25 score is negative and the lower the value the better the
        match, so the direction is recorded as
        :attr:`~vector_stores.contracts.models.ScoreDirection.LOWER_IS_BETTER`.
        """
        return RetrievedItem(
            chunk_id=hit.chunk_id,
            doc_id=hit.doc_id,
            text=hit.text,
            score=hit.score,
            score_direction=ScoreDirection.LOWER_IS_BETTER,
            score_interpretation="bm25",
            rank=hit.rank,
            strategy=self.name,
            metadata=dict(hit.metadata),
            provenance=hit.provenance,
            methods=(self.name,),
            channel_ranks={self.name: hit.rank},
            channel_scores={self.name: hit.score},
        )


def _decode_metadata(raw: Any) -> dict[str, Any]:
    """Decode the metadata JSON of a content row."""
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _decode_provenance(raw: Any) -> Any:
    """Decode the provenance JSON of a content row."""
    if not raw:
        return None
    from vector_stores.contracts.models import ChunkProvenance

    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(decoded, dict):
        return None
    return ChunkProvenance(
        source=decoded.get("source"),
        pages=tuple(int(page) for page in decoded.get("pages") or ()),
        block_refs=tuple(decode_block_refs(decoded.get("block_refs"))),
        char_start=decoded.get("char_start"),
        char_end=decoded.get("char_end"),
    )


def build_document_filter(doc_ids: Sequence[str]) -> MetadataFilter | None:
    """Return the filter that restricts a search to ``doc_ids``."""
    if not doc_ids:
        return None
    return eq("doc_id", doc_ids[0]) if len(doc_ids) == 1 else is_in("doc_id", list(doc_ids))
