"""MBOX email parser: multi-message mailbox extraction using Python's mailbox module."""

from __future__ import annotations

import email
from email import policy
import mailbox
import logging
from pathlib import Path
from typing import Any

from parsers.contracts import (
    CorruptDocumentError,
    EmptyDocumentError,
    ParsedBlock,
    ParsedDocument,
    ParsedPage,
)
from parsers.contracts.validation import validate_source
from parsers.email.eml.parser import _extract_parts, _parse_headers

logger = logging.getLogger(__name__)


class MboxParser:
    """Extract structured multi-message mailbox archives (.mbox)."""

    name = "mbox_stdlib"

    def parse(self, source: str | Path) -> ParsedDocument:
        path = validate_source(source)

        if path.stat().st_size == 0:
            raise EmptyDocumentError(f"MBOX file '{path}' is empty (0 bytes).")

        try:
            box = mailbox.mbox(str(path))
        except Exception as exc:
            raise CorruptDocumentError(f"Could not open '{path}' as an MBOX file: {exc}") from exc

        pages: list[ParsedPage] = []
        messages_meta: list[dict[str, Any]] = []

        try:
            for msg_idx, item in enumerate(box, start=1):
                raw = item.as_bytes()
                msg = email.message_from_bytes(raw, policy=policy.default)
                headers = _parse_headers(msg)
                blocks, mime_struct = _extract_parts(msg)

                # Set page_number on blocks
                for b in blocks:
                    b.page_number = msg_idx

                headers["message_index"] = msg_idx
                headers["mime_structure"] = mime_struct
                messages_meta.append(headers)

                pages.append(
                    ParsedPage(
                        page_number=msg_idx,
                        blocks=blocks,
                    )
                )
        except Exception as exc:
            raise CorruptDocumentError(f"Error parsing messages in MBOX file '{path}': {exc}") from exc

        if not pages:
            raise EmptyDocumentError(f"MBOX file '{path}' contains no messages.")

        return ParsedDocument(
            source=str(path),
            parser_name=self.name,
            parser_version="stdlib",
            metadata={
                "message_count": len(pages),
                "messages": messages_meta,
            },
            pages=pages,
            blocks=[],
            warnings=[],
        )
