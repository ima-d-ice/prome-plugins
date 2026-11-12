"""Backend independent encoding of the contract fields of a stored record.

Every backend stores the payload in its own way (an Arrow struct, a JSONB
document, a SQLite column, ...), but the *fields* the contract defines are
always the same:

- the chunk text, its index/count and its length,
- the original chunk metadata,
- the provenance (source, pages, block references, character offsets),
- the embedding-space descriptor of every named vector space.

Encoding them once keeps the adapters small and makes a record portable
between backends: a record written by one adapter can be read by another
without losing information.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from chunkers.contracts.models import BlockReference, Chunk
from vector_stores.contracts.models import (
    ChunkProvenance,
    SpaceVectors,
    StoredRecord,
    descriptor_from_dict,
    descriptor_to_dict,
)

#: Key under which the contract fields are stored in a payload.
CONTRACT_PAYLOAD_KEY = "_prome"

#: Top level payload keys owned by the contract.
RESERVED_PAYLOAD_KEYS = ("chunk_id", "doc_id", CONTRACT_PAYLOAD_KEY)


def encode_contract_payload(record: StoredRecord) -> dict[str, Any]:
    """Return the contract fields of ``record`` (the ``_prome`` content)."""
    chunk = record.chunk
    return {
        "text": chunk.text,
        "index": chunk.index,
        "count": chunk.count,
        "length": chunk.length,
        "length_unit": chunk.length_unit,
        "heading": chunk.heading,
        "block_refs": [ref.as_dict() for ref in chunk.block_refs],
        "chunk_metadata": dict(chunk.metadata),
        "provenance": ChunkProvenance.from_chunk(chunk).as_dict(),
        "embedding_spaces": {
            name: descriptor_to_dict(vectors.embedding_space)
            for name, vectors in record.vectors.items()
        },
    }


def decode_block_refs(raw: Any) -> list[BlockReference]:
    """Rebuild the block references of a chunk."""
    refs: list[BlockReference] = []
    for item in raw or []:
        if isinstance(item, BlockReference):
            refs.append(item)
        elif isinstance(item, Mapping):
            refs.append(
                BlockReference(
                    block_index=item["block_index"],
                    page_number=item.get("page_number"),
                    block_type=item.get("block_type", "other"),
                    char_start=item.get("char_start"),
                    char_end=item.get("char_end"),
                )
            )
    return refs


def decode_contract_payload(payload: Mapping[str, Any]) -> StoredRecord:
    """Rebuild a record from a payload.

    A payload that does not carry the contract fields (``_prome``) still
    yields the chunk id and the document id, so a result never loses its
    identity.
    """
    payload = payload or {}
    contract = payload.get(CONTRACT_PAYLOAD_KEY) or {}
    chunk = Chunk(
        id=str(payload.get("chunk_id", "")),
        doc_id=str(payload.get("doc_id", "")),
        text=str(contract.get("text", "")),
        index=int(contract.get("index", 0)),
        count=int(contract.get("count", 0)),
        length=int(contract.get("length", 0)),
        length_unit=str(contract.get("length_unit", "characters")),
        block_refs=decode_block_refs(contract.get("block_refs")),
        heading=contract.get("heading"),
        metadata=dict(contract.get("chunk_metadata") or {}),
    )
    metadata = {
        key: value for key, value in payload.items() if key not in RESERVED_PAYLOAD_KEYS
    }
    return StoredRecord(chunk=chunk, vectors={}, metadata=metadata)


def contract_vectors(
    payload: Mapping[str, Any],
) -> dict[str, SpaceVectors]:
    """Rebuild the (empty) vector slots of a record from its payload.

    The descriptors are restored; the vectors themselves are added by the
    backend, which knows where they are stored.
    """
    contract = payload.get(CONTRACT_PAYLOAD_KEY) or {}
    descriptors = contract.get("embedding_spaces") or {}
    vectors: dict[str, SpaceVectors] = {}
    for name, raw in descriptors.items():
        vectors[name] = SpaceVectors(
            space=name,
            embedding_space=descriptor_from_dict(raw),
        )
    return vectors


def payload_to_json(payload: Mapping[str, Any]) -> str:
    """Serialize a payload to JSON (for backends with a text column)."""
    return json.dumps(payload, ensure_ascii=False, default=str)


def payload_from_json(raw: str | None) -> dict[str, Any]:
    """Parse a payload from JSON."""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}
