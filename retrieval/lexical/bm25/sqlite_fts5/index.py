"""The lexical index: a relational content table plus an FTS5 index.

The index has two parts:

* a **relational content table** (``chunks``) that holds the authoritative
  row of every chunk -- its id, its document, its metadata and its provenance;
* an **FTS5 virtual table** (``chunks_fts``) that holds the searchable text
  and provides the BM25 ranking.

The two are kept in sync by SQLite triggers, so the FTS index cannot drift
away from the content table: an insert, an update and a delete of a content
row are mirrored into the FTS index by the database itself.  The triggers use
the FTS5 ``external content`` mode, which means the text is stored **once**,
in the content table.

The FTS5 tokenizer is configurable.  The default keeps the underscore inside
a token, so a technical identifier such as ``get_user_id`` stays one token
instead of being split into three.  See :func:`retrieval.lexical.bm25.sqlite_fts5.tokenizer.build_match_query`
for the exact behaviour and for how a query is turned into a safe FTS5
``MATCH`` expression.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retrieval.contracts.exceptions import (
    FTS5UnavailableError,
    LexicalIndexError,
)
from vector_stores.contracts.payload import decode_block_refs
from vector_stores.contracts.models import ChunkProvenance

#: The name of the relational content table.
CHUNKS_TABLE = "chunks"

#: The name of the FTS5 virtual table.
FTS_TABLE = "chunks_fts"

#: The default FTS5 tokenizer.
#:
#: ``unicode61`` is the built-in Unicode tokenizer; ``tokenchars '_'`` keeps
#: the underscore inside a token so that ``snake_case`` identifiers survive
#: as one token.
DEFAULT_TOKENIZER = "unicode61 tokenchars '_'"

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {CHUNKS_TABLE} (
    chunk_id   TEXT PRIMARY KEY,
    doc_id     TEXT NOT NULL,
    text       TEXT NOT NULL,
    metadata   TEXT NOT NULL DEFAULT '{{}}',
    provenance TEXT,
    source     TEXT,
    chunk_index INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON {CHUNKS_TABLE}(doc_id);
"""


def fts5_available(connection: sqlite3.Connection | None = None) -> bool:
    """Return whether the SQLite build provides FTS5."""
    own = connection is None
    connection = connection or sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE VIRTUAL TABLE temp.__fts5_probe USING fts5(x)")
        connection.execute("DROP TABLE temp.__fts5_probe")
        return True
    except sqlite3.Error:
        return False
    finally:
        if own:
            connection.close()


@dataclass(frozen=True)
class LexicalRecord:
    """One row of the lexical index."""

    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any]
    provenance: ChunkProvenance | None = None
    source: str | None = None
    chunk_index: int | None = None

    @classmethod
    def from_chunk(
        cls,
        chunk: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> LexicalRecord:
        """Build a record from a :class:`chunkers.contracts.models.Chunk`."""
        return cls(
            chunk_id=chunk.id,
            doc_id=chunk.doc_id,
            text=chunk.text,
            metadata=dict(metadata if metadata is not None else chunk.metadata),
            provenance=ChunkProvenance.from_chunk(chunk),
            source=chunk.metadata.get("source", chunk.doc_id),
            chunk_index=chunk.index,
        )


class SQLiteFTS5Index:
    """A persistent lexical index over chunks.

    Args:
        path: the file of the index.  ``":memory:"`` creates an ephemeral one.
        tokenizer: the FTS5 tokenizer specification.
    """

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        tokenizer: str = DEFAULT_TOKENIZER,
    ) -> None:
        self.path = str(path)
        self.tokenizer = tokenizer
        self._connection: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def open(self) -> SQLiteFTS5Index:
        """Open (and create when needed) the index."""
        if self._connection is not None:
            return self
        try:
            connection = sqlite3.connect(self.path)
        except sqlite3.Error as exc:
            raise LexicalIndexError(
                f"could not open the lexical index {self.path!r}: {exc}"
            ) from exc
        connection.row_factory = sqlite3.Row
        if not fts5_available(connection):
            connection.close()
            raise FTS5UnavailableError(
                "the SQLite build of this interpreter does not provide FTS5; the "
                "lexical channel needs it for its BM25 ranking. The capability is "
                "reported instead of being replaced by another implementation."
            )
        self._connection = connection
        self._create_schema()
        return self

    def close(self) -> None:
        """Close the index; the data stays on disk."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> SQLiteFTS5Index:
        return self.open()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def connection(self) -> sqlite3.Connection:
        """Return the open connection."""
        if self._connection is None:
            raise LexicalIndexError("the lexical index is not open")
        return self._connection

    def _create_schema(self) -> None:
        """Create the content table, the FTS index and the sync triggers."""
        connection = self.connection
        try:
            connection.executescript(_SCHEMA)
            # The tokenizer specification is embedded in the DDL of the FTS5
            # table, so the single quotes it contains must be doubled the
            # same way any SQLite string literal quotes them.
            tokenizer = self.tokenizer.replace("'", "''")
            connection.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE} USING fts5("
                "text, content='chunks', content_rowid='rowid', "
                f"tokenize='{tokenizer}')"
            )
            connection.executescript(_TRIGGERS)
            connection.commit()
        except sqlite3.Error as exc:
            raise LexicalIndexError(
                f"could not create the lexical index {self.path!r}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # writing
    # ------------------------------------------------------------------
    def upsert(self, records: Iterable[LexicalRecord]) -> int:
        """Insert or replace ``records``; return how many were written."""
        rows = [
            (
                record.chunk_id,
                record.doc_id,
                record.text,
                json.dumps(record.metadata, ensure_ascii=False, default=str),
                (
                    json.dumps(record.provenance.as_dict(), ensure_ascii=False, default=str)
                    if record.provenance is not None
                    else None
                ),
                record.source,
                record.chunk_index,
            )
            for record in records
        ]
        if not rows:
            return 0
        connection = self.connection
        try:
            connection.executemany(
                f"INSERT INTO {CHUNKS_TABLE} "
                "(chunk_id, doc_id, text, metadata, provenance, source, chunk_index) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(chunk_id) DO UPDATE SET "
                "doc_id=excluded.doc_id, text=excluded.text, "
                "metadata=excluded.metadata, provenance=excluded.provenance, "
                "source=excluded.source, chunk_index=excluded.chunk_index",
                rows,
            )
            connection.commit()
        except sqlite3.Error as exc:
            raise LexicalIndexError(
                f"could not write to the lexical index {self.path!r}: {exc}"
            ) from exc
        return len(rows)

    def delete(self, chunk_ids: Sequence[str]) -> int:
        """Delete the given chunks; return how many rows were removed."""
        if not chunk_ids:
            return 0
        connection = self.connection
        try:
            cursor = connection.execute(
                f"DELETE FROM {CHUNKS_TABLE} WHERE chunk_id IN "
                f"({', '.join('?' * len(chunk_ids))})",
                list(chunk_ids),
            )
            connection.commit()
        except sqlite3.Error as exc:
            raise LexicalIndexError(
                f"could not delete from the lexical index {self.path!r}: {exc}"
            ) from exc
        return cursor.rowcount

    def delete_documents(self, doc_ids: Sequence[str]) -> int:
        """Delete every chunk of the given documents."""
        if not doc_ids:
            return 0
        connection = self.connection
        try:
            cursor = connection.execute(
                f"DELETE FROM {CHUNKS_TABLE} WHERE doc_id IN "
                f"({', '.join('?' * len(doc_ids))})",
                list(doc_ids),
            )
            connection.commit()
        except sqlite3.Error as exc:
            raise LexicalIndexError(
                f"could not delete from the lexical index {self.path!r}: {exc}"
            ) from exc
        return cursor.rowcount

    # ------------------------------------------------------------------
    # reading
    # ------------------------------------------------------------------
    def count(self) -> int:
        """Return the number of indexed chunks."""
        connection = self.connection
        try:
            row = connection.execute(f"SELECT COUNT(*) AS n FROM {CHUNKS_TABLE}").fetchone()
        except sqlite3.Error as exc:
            raise LexicalIndexError(
                f"could not count the lexical index {self.path!r}: {exc}"
            ) from exc
        return int(row["n"])

    def get(self, chunk_ids: Sequence[str]) -> list[LexicalRecord]:
        """Return the records of the given chunk ids."""
        if not chunk_ids:
            return []
        connection = self.connection
        placeholders = ", ".join("?" * len(chunk_ids))
        try:
            rows = connection.execute(
                f"SELECT * FROM {CHUNKS_TABLE} WHERE chunk_id IN ({placeholders})",
                list(chunk_ids),
            ).fetchall()
        except sqlite3.Error as exc:
            raise LexicalIndexError(
                f"could not read the lexical index {self.path!r}: {exc}"
            ) from exc
        return [_row_to_record(row) for row in rows]

    def documents(self) -> list[str]:
        """Return the ids of the indexed documents."""
        connection = self.connection
        try:
            rows = connection.execute(
                f"SELECT DISTINCT doc_id FROM {CHUNKS_TABLE} ORDER BY doc_id"
            ).fetchall()
        except sqlite3.Error as exc:
            raise LexicalIndexError(
                f"could not read the lexical index {self.path!r}: {exc}"
            ) from exc
        return [str(row["doc_id"]) for row in rows]

    # ------------------------------------------------------------------
    # integrity
    # ------------------------------------------------------------------
    def integrity_check(self) -> bool:
        """Return whether every content row is findable in the FTS index.

        The check is a coverage check: for every row of the content table it
        asks the FTS index for the rows matching the first searchable term of
        that row, and expects the row itself to be among them.  If the FTS
        index was corrupted or modified outside of this class, a row that is
        no longer findable is detected.

        The FTS5 built-in ``integrity-check`` command only verifies the
        internal consistency of the index tables, not that the index agrees
        with the content table, so it is not used here.
        """
        from retrieval.lexical.bm25.sqlite_fts5.tokenizer import tokenize_query

        connection = self.connection
        try:
            rows = connection.execute(
                f"SELECT rowid, text FROM {CHUNKS_TABLE}"
            ).fetchall()
            for row in rows:
                terms = tokenize_query(str(row["text"]))
                if not terms:
                    continue
                hits = connection.execute(
                    f"SELECT rowid FROM {FTS_TABLE} WHERE "
                    f"{FTS_TABLE} MATCH ?",
                    (terms[0],),
                ).fetchall()
                if int(row["rowid"]) not in {int(hit[0]) for hit in hits}:
                    return False
        except sqlite3.Error as exc:
            raise LexicalIndexError(
                f"the integrity check of {self.path!r} failed: {exc}"
            ) from exc
        return True

    def rebuild(self) -> None:
        """Rebuild the FTS index from the content table.

        The triggers keep the index in sync, so a rebuild is only needed
        after an external modification of the database.
        """
        connection = self.connection
        try:
            connection.execute(f"INSERT INTO {FTS_TABLE}({FTS_TABLE}) VALUES('rebuild')")
            connection.commit()
        except sqlite3.Error as exc:
            raise LexicalIndexError(
                f"could not rebuild the lexical index {self.path!r}: {exc}"
            ) from exc


#: The triggers that keep the FTS index in sync with the content table.
_TRIGGERS = f"""
CREATE TRIGGER IF NOT EXISTS {CHUNKS_TABLE}_ai AFTER INSERT ON {CHUNKS_TABLE} BEGIN
    INSERT INTO {FTS_TABLE}(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS {CHUNKS_TABLE}_ad AFTER DELETE ON {CHUNKS_TABLE} BEGIN
    INSERT INTO {FTS_TABLE}({FTS_TABLE}, rowid, text)
    VALUES ('delete', old.rowid, old.text);
END;
CREATE TRIGGER IF NOT EXISTS {CHUNKS_TABLE}_au AFTER UPDATE ON {CHUNKS_TABLE} BEGIN
    INSERT INTO {FTS_TABLE}({FTS_TABLE}, rowid, text)
    VALUES ('delete', old.rowid, old.text);
    INSERT INTO {FTS_TABLE}(rowid, text) VALUES (new.rowid, new.text);
END;
"""


def _row_to_record(row: sqlite3.Row) -> LexicalRecord:
    """Turn a content row into a :class:`LexicalRecord`."""
    metadata: dict[str, Any] = {}
    raw_metadata = row["metadata"]
    if raw_metadata:
        try:
            decoded = json.loads(raw_metadata)
        except (TypeError, ValueError):
            decoded = {}
        if isinstance(decoded, dict):
            metadata = decoded
    provenance = None
    raw_provenance = row["provenance"]
    if raw_provenance:
        try:
            decoded = json.loads(raw_provenance)
        except (TypeError, ValueError):
            decoded = None
        if isinstance(decoded, dict):
            provenance = ChunkProvenance(
                source=decoded.get("source"),
                pages=tuple(int(page) for page in decoded.get("pages") or ()),
                block_refs=tuple(decode_block_refs(decoded.get("block_refs"))),
                char_start=decoded.get("char_start"),
                char_end=decoded.get("char_end"),
            )
    return LexicalRecord(
        chunk_id=str(row["chunk_id"]),
        doc_id=str(row["doc_id"]),
        text=str(row["text"]),
        metadata=metadata,
        provenance=provenance,
        source=row["source"],
        chunk_index=row["chunk_index"],
    )
