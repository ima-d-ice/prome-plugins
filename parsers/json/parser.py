"""Standard-library JSON and NDJSON/JSONL parser with JSON-path hierarchy preservation.

Preserves JSON paths, key-value relationships, array indices, nested hierarchy,
and primitive data types without lossy flattening.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Generator

from parsers.contracts import (
    BlockType,
    CorruptDocumentError,
    EmptyDocumentError,
    ParsedBlock,
    ParsedDocument,
)
from parsers.contracts.validation import validate_source

logger = logging.getLogger(__name__)


def _walk_leaves(obj: Any, path: str = "$") -> Generator[tuple[str, str, Any, str], None, None]:
    """Yield (path, key, value, data_type) for leaf nodes."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            sub = f"{path}.{k}"
            yield from _walk_leaves(v, sub)
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            sub = f"{path}[{idx}]"
            yield from _walk_leaves(item, sub)
    else:
        key = path.rsplit(".", 1)[-1].split("[")[0] if "." in path else path
        val_type = "null" if obj is None else type(obj).__name__
        yield path, key, obj, val_type


class JsonParser:
    """Extract structured JSON and NDJSON content into path-preserved blocks."""

    name = "json_stdlib"

    def __init__(self, mode: str = "auto") -> None:
        """Mode can be 'auto', 'json', or 'ndjson'."""
        self.mode = mode

    def parse(self, source: str | Path) -> ParsedDocument:
        path = validate_source(source)

        if path.stat().st_size == 0:
            raise EmptyDocumentError(f"JSON file '{path}' is empty (0 bytes).")

        try:
            content_text = path.read_text(encoding="utf-8")
        except Exception as exc:
            raise CorruptDocumentError(f"Could not read JSON file '{path}': {exc}") from exc

        trimmed = content_text.strip()
        if not trimmed:
            raise EmptyDocumentError(f"JSON file '{path}' contains only whitespace.")

        is_ndjson = False
        parsed_data: Any = None

        if self.mode == "ndjson":
            is_ndjson = True
        elif self.mode == "json":
            is_ndjson = False
        else:
            # Auto-detect: if trimmed doesn't start with { or [ but lines do
            if not trimmed.startswith(("{", "[")) and "\n" in trimmed:
                is_ndjson = True

        blocks: list[ParsedBlock] = []

        if is_ndjson:
            lines = [ln.strip() for ln in trimmed.splitlines() if ln.strip()]
            if not lines:
                raise EmptyDocumentError(f"NDJSON file '{path}' contains no records.")
            records = []
            for line_idx, line in enumerate(lines, start=1):
                try:
                    item = json.loads(line)
                    records.append(item)
                except Exception as exc:
                    raise CorruptDocumentError(f"Error parsing NDJSON line {line_idx} in '{path}': {exc}") from exc

                # Emit RECORD block for this line
                blocks.append(
                    ParsedBlock(
                        block_type=BlockType.RECORD,
                        content=json.dumps(item, ensure_ascii=False),
                        metadata={"ndjson_line": line_idx, "json_path": f"$[{line_idx - 1}]"},
                    )
                )
                # Emit leaf KEY_VALUE blocks
                for leaf_path, key, val, val_type in _walk_leaves(item, path=f"$[{line_idx - 1}]"):
                    blocks.append(
                        ParsedBlock(
                            block_type=BlockType.KEY_VALUE,
                            content=f"{leaf_path} = {val}",
                            metadata={
                                "json_path": leaf_path,
                                "key": key,
                                "value": val,
                                "data_type": val_type,
                                "ndjson_line": line_idx,
                            },
                        )
                    )
            root_type = "ndjson"
            total_records = len(records)
        else:
            try:
                parsed_data = json.loads(trimmed)
            except Exception as exc:
                # If auto-mode failed, maybe it is ndjson?
                if self.mode == "auto" and "\n" in trimmed:
                    try:
                        lines = [ln.strip() for ln in trimmed.splitlines() if ln.strip()]
                        records = [json.loads(ln) for ln in lines]
                        return JsonParser(mode="ndjson").parse(source)
                    except Exception:
                        pass
                raise CorruptDocumentError(f"Could not parse '{path}' as valid JSON: {exc}") from exc

            if parsed_data == {} or parsed_data == []:
                raise EmptyDocumentError(f"JSON file '{path}' contains an empty container ({type(parsed_data).__name__}).")

            # Emit top-level RECORD block
            blocks.append(
                ParsedBlock(
                    block_type=BlockType.RECORD,
                    content=json.dumps(parsed_data, indent=2, ensure_ascii=False),
                    metadata={"json_path": "$", "data_type": type(parsed_data).__name__},
                )
            )

            # Emit leaf KEY_VALUE blocks
            for leaf_path, key, val, val_type in _walk_leaves(parsed_data):
                blocks.append(
                    ParsedBlock(
                        block_type=BlockType.KEY_VALUE,
                        content=f"{leaf_path} = {val}",
                        metadata={
                            "json_path": leaf_path,
                            "key": key,
                            "value": val,
                            "data_type": val_type,
                        },
                    )
                )

            root_type = type(parsed_data).__name__
            total_records = len(parsed_data) if isinstance(parsed_data, (list, dict)) else 1

        metadata = {
            "root_type": root_type,
            "record_count": total_records,
            "leaf_count": sum(1 for b in blocks if b.block_type == BlockType.KEY_VALUE),
        }

        return ParsedDocument(
            source=str(path),
            parser_name=self.name,
            parser_version="stdlib",
            metadata=metadata,
            pages=[],
            blocks=blocks,
            warnings=[],
        )
