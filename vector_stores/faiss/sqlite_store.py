"""The SQLite metadata store of a FAISS collection.

FAISS only stores vectors; the records, their metadata, their provenance and
the collection configuration live in SQLite.  SQLite is the **authoritative**
store: the FAISS index is a derived artifact that can be rebuilt from the
vectors kept here.

The metadata is stored as a JSON document and read back with
``json_extract``; the declared payload fields get a SQLite expression index
on that expression.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from vector_stores.contracts.exceptions import (
    InvalidRecordError,
)
from vector_stores.contracts.filters import MetadataFilter
from vector_stores.contracts.models import CollectionConfig
from vector_stores.faiss.mapping import (
    CHUNK_ID_COLUMN,
    FAISS_ID_COLUMN,
    METADATA_COLUMN,
    PAYLOAD_COLUMN,
    VECTOR_COLUMN,
    metadata_expression,
    payload_index_name,
    to_sql_filter,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    faiss_id INTEGER NOT NULL,
    payload TEXT NOT NULL,
    metadata TEXT NOT NULL,
    vector BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_records_faiss_id ON records(faiss_id);
CREATE INDEX IF NOT EXISTS idx_records_doc_id ON records(doc_id);
CREATE TABLE IF NOT EXISTS collection_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class SQLiteRecordStore:
    """The authoritative record store of one collection."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self._path))
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(_SCHEMA)
        self._connection.commit()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close the database connection."""
        self._connection.commit()
        self._connection.close()

    # ------------------------------------------------------------------
    # collection configuration
    # ------------------------------------------------------------------
    def set_config(self, config: CollectionConfig) -> None:
        """Store the declared configuration of the collection."""
        self._connection.execute(
            "INSERT INTO collection_meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            ("config", json.dumps(config.as_dict(), default=str)),
        )
        self._connection.commit()

    def get_config(self) -> CollectionConfig | None:
        """Return the declared configuration of the collection."""
        row = self._connection.execute(
            "SELECT value FROM collection_meta WHERE key = ?", ("config",)
        ).fetchone()
        if row is None:
            return None
        try:
            return CollectionConfig.from_dict(json.loads(row["value"]))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None

    def delete_config(self) -> None:
        """Remove the declared configuration."""
        self._connection.execute("DELETE FROM collection_meta WHERE key = ?", ("config",))
        self._connection.commit()

    # ------------------------------------------------------------------
    # records
    # ------------------------------------------------------------------
    def upsert_record(
        self,
        *,
        chunk_id: str,
        doc_id: str,
        faiss_id: int,
        payload: str,
        metadata: str,
        vector: bytes,
    ) -> int | None:
        """Insert or replace a record.

        Returns:
            The previous FAISS id of the record, or ``None`` when the record
            was new.  The caller uses it to know which index slot became
            stale.
        """
        row = self._connection.execute(
            f"SELECT {FAISS_ID_COLUMN} FROM records WHERE {CHUNK_ID_COLUMN} = ?",
            (chunk_id,),
        ).fetchone()
        previous = int(row[FAISS_ID_COLUMN]) if row is not None else None
        self._connection.execute(
            "INSERT INTO records(chunk_id, doc_id, faiss_id, payload, metadata, vector) "
            "VALUES(?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(chunk_id) DO UPDATE SET doc_id = excluded.doc_id, "
            f"{FAISS_ID_COLUMN} = excluded.{FAISS_ID_COLUMN}, "
            f"{PAYLOAD_COLUMN} = excluded.{PAYLOAD_COLUMN}, "
            f"{METADATA_COLUMN} = excluded.{METADATA_COLUMN}, "
            f"{VECTOR_COLUMN} = excluded.{VECTOR_COLUMN}",
            (chunk_id, doc_id, faiss_id, payload, metadata, vector),
        )
        self._connection.commit()
        return previous

    def get_records(self, chunk_ids: Sequence[str]) -> list[sqlite3.Row]:
        """Return the records of the given chunk ids."""
        if not chunk_ids:
            return []
        placeholders = ", ".join("?" for _ in chunk_ids)
        return list(
            self._connection.execute(
                f"SELECT * FROM records WHERE {CHUNK_ID_COLUMN} IN ({placeholders})",
                list(chunk_ids),
            )
        )

    def delete_records(self, chunk_ids: Sequence[str]) -> int:
        """Delete the records of the given chunk ids."""
        if not chunk_ids:
            return 0
        placeholders = ", ".join("?" for _ in chunk_ids)
        cursor = self._connection.execute(
            f"DELETE FROM records WHERE {CHUNK_ID_COLUMN} IN ({placeholders})",
            list(chunk_ids),
        )
        self._connection.commit()
        return cursor.rowcount

    def delete_by_filter(self, node: MetadataFilter) -> int:
        """Delete every record matching ``node``."""
        sql, params = to_sql_filter(node)
        cursor = self._connection.execute(f"DELETE FROM records WHERE {sql}", params)
        self._connection.commit()
        return cursor.rowcount

    def count(self, node: MetadataFilter | None = None) -> int:
        """Count the records, optionally filtered."""
        if node is None:
            row = self._connection.execute("SELECT COUNT(*) FROM records").fetchone()
            return int(row[0])
        sql, params = to_sql_filter(node)
        row = self._connection.execute(
            f"SELECT COUNT(*) FROM records WHERE {sql}", params
        ).fetchone()
        return int(row[0])

    def scroll(
        self,
        *,
        limit: int,
        offset: int = 0,
        node: MetadataFilter | None = None,
    ) -> list[sqlite3.Row]:
        """Return one page of records, ordered by chunk id."""
        if node is None:
            return list(
                self._connection.execute(
                    "SELECT * FROM records ORDER BY chunk_id LIMIT ? OFFSET ?",
                    (limit, offset),
                )
            )
        sql, params = to_sql_filter(node)
        return list(
            self._connection.execute(
                f"SELECT * FROM records WHERE {sql} ORDER BY chunk_id LIMIT ? OFFSET ?",
                [*params, limit, offset],
            )
        )

    def iter_records(self, node: MetadataFilter | None = None) -> Iterator[sqlite3.Row]:
        """Iterate over every record."""
        if node is None:
            cursor = self._connection.execute("SELECT * FROM records ORDER BY chunk_id")
        else:
            sql, params = to_sql_filter(node)
            cursor = self._connection.execute(
                f"SELECT * FROM records WHERE {sql} ORDER BY chunk_id", params
            )
        yield from cursor

    def live_faiss_ids(self, node: MetadataFilter | None = None) -> list[int]:
        """Return the FAISS ids of the live records."""
        if node is None:
            rows = self._connection.execute(
                f"SELECT {FAISS_ID_COLUMN} FROM records"
            ).fetchall()
        else:
            sql, params = to_sql_filter(node)
            rows = self._connection.execute(
                f"SELECT {FAISS_ID_COLUMN} FROM records WHERE {sql}", params
            ).fetchall()
        return [int(row[0]) for row in rows]

    def row_by_faiss_id(self, faiss_id: int) -> sqlite3.Row | None:
        """Return the record stored under a FAISS id."""
        return self._connection.execute(
            f"SELECT * FROM records WHERE {FAISS_ID_COLUMN} = ?", (faiss_id,)
        ).fetchone()

    def max_faiss_id(self) -> int:
        """Return the highest FAISS id in use (``-1`` when there is none)."""
        row = self._connection.execute(
            f"SELECT MAX({FAISS_ID_COLUMN}) FROM records"
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else -1

    def set_faiss_id(self, chunk_id: str, faiss_id: int) -> None:
        """Renumber a record (used when the index is rebuilt)."""
        self._connection.execute(
            f"UPDATE records SET {FAISS_ID_COLUMN} = ? WHERE {CHUNK_ID_COLUMN} = ?",
            (faiss_id, chunk_id),
        )
        self._connection.commit()

    # ------------------------------------------------------------------
    # payload indexes
    # ------------------------------------------------------------------
    def create_payload_index(self, field: str) -> str:
        """Create a SQLite index on a metadata field."""
        expression = metadata_expression(field)
        name = payload_index_name(field)
        self._connection.execute(
            f"CREATE INDEX IF NOT EXISTS {name} ON records({expression})"
        )
        self._connection.commit()
        return name

    def drop_payload_index(self, field: str) -> bool:
        """Drop the SQLite index of a metadata field."""
        name = payload_index_name(field)
        row = self._connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name = ?",
            (name,),
        ).fetchone()
        if row is None:
            return False
        self._connection.execute(f"DROP INDEX {name}")
        self._connection.commit()
        return True

    def list_payload_indexes(self) -> list[str]:
        """Return the metadata fields that have a SQLite index."""
        rows = self._connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' "
            "AND name LIKE 'idx_meta_%' ORDER BY name"
        ).fetchall()
        return [str(row["name"])[len("idx_meta_") :] for row in rows]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def row_payload(row: Mapping[str, Any]) -> dict[str, Any]:
        """Return the contract payload of a row."""
        raw = row[PAYLOAD_COLUMN]
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def row_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
        """Return the metadata of a row."""
        raw = row[METADATA_COLUMN]
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def check_chunk_id(chunk_id: str) -> str:
        """Validate a chunk id."""
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise InvalidRecordError("chunk id must be a non-empty string")
        return chunk_id

    @staticmethod
    def require_doc_id(doc_id: str) -> str:
        """Validate a document id."""
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise InvalidRecordError("chunk document id must be a non-empty string")
        return doc_id
