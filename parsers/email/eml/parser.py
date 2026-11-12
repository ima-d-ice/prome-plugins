"""EML email parser: RFC 822 / MIME message extraction with attachments and thread relationships."""

from __future__ import annotations

import email
from email import policy
import logging
from pathlib import Path
from typing import Any

from parsers.contracts import (
    BlockType,
    CorruptDocumentError,
    EmptyDocumentError,
    ParsedBlock,
    ParsedDocument,
)
from parsers.contracts.validation import validate_source

logger = logging.getLogger(__name__)


def _parse_headers(msg) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    for key in ("from", "to", "cc", "bcc", "subject", "date", "message-id", "in-reply-to", "references"):
        val = msg[key]
        if val is not None:
            headers[key.replace("-", "_")] = str(val).strip()

    # Parse references into list if present
    refs_val = headers.get("references")
    if refs_val:
        headers["references_list"] = [r.strip() for r in refs_val.split() if r.strip()]
    else:
        headers["references_list"] = []

    headers["is_thread_reply"] = bool(headers.get("in_reply_to") or headers["references_list"])
    return headers


def _extract_parts(msg) -> tuple[list[ParsedBlock], list[dict[str, Any]]]:
    blocks: list[ParsedBlock] = []
    mime_structure: list[dict[str, Any]] = []

    text_bodies: list[str] = []
    html_bodies: list[str] = []

    for part_idx, part in enumerate(msg.walk()):
        content_type = part.get_content_type()
        is_attachment = part.is_attachment()
        filename = part.get_filename()
        payload = part.get_payload(decode=True)
        size_bytes = len(payload) if payload is not None else 0

        mime_structure.append({
            "part_index": part_idx,
            "content_type": content_type,
            "is_attachment": is_attachment,
            "filename": filename,
            "size_bytes": size_bytes,
        })

        if is_attachment or filename:
            meta = {
                "filename": filename or f"attachment_{part_idx}",
                "content_type": content_type,
                "size_bytes": size_bytes,
                "content_disposition": str(part.get("content-disposition", "")),
            }
            blocks.append(
                ParsedBlock(
                    block_type=BlockType.ATTACHMENT,
                    content=f"[Attachment: {meta['filename']} ({content_type}, {size_bytes} bytes)]",
                    page_number=None,
                    metadata=meta,
                )
            )
            continue

        if content_type == "text/plain":
            try:
                body = part.get_content()
                if body and isinstance(body, str) and body.strip():
                    text_bodies.append(body.strip())
            except Exception:
                if payload:
                    try:
                        text_bodies.append(payload.decode("utf-8", errors="replace").strip())
                    except Exception:
                        pass
        elif content_type == "text/html":
            try:
                body = part.get_content()
                if body and isinstance(body, str) and body.strip():
                    html_bodies.append(body.strip())
            except Exception:
                if payload:
                    try:
                        html_bodies.append(payload.decode("utf-8", errors="replace").strip())
                    except Exception:
                        pass

    # Add text bodies first
    for text in text_bodies:
        blocks.append(
            ParsedBlock(
                block_type=BlockType.TEXT,
                content=text,
                page_number=None,
                metadata={"content_type": "text/plain", "region": "body"},
            )
        )

    # If no text body, or in addition to text body, add html body
    for html in html_bodies:
        blocks.append(
            ParsedBlock(
                block_type=BlockType.TEXT,
                content=html,
                page_number=None,
                metadata={"content_type": "text/html", "region": "body_html"},
            )
        )

    return blocks, mime_structure


class EmlParser:
    """Extract structured email content from RFC 822 / .eml files."""

    name = "eml_stdlib"

    def parse(self, source: str | Path) -> ParsedDocument:
        path = validate_source(source)

        if path.stat().st_size == 0:
            raise EmptyDocumentError(f"EML file '{path}' is empty (0 bytes).")

        try:
            raw_bytes = path.read_bytes()
            msg = email.message_from_bytes(raw_bytes, policy=policy.default)
        except Exception as exc:
            raise CorruptDocumentError(f"Could not parse '{path}' as an EML message: {exc}") from exc

        headers = _parse_headers(msg)
        blocks, mime_struct = _extract_parts(msg)

        if not blocks and not any(headers.values()):
            raise EmptyDocumentError(f"EML file '{path}' contains no message content or headers.")

        metadata = {
            **headers,
            "mime_structure": mime_struct,
            "attachment_count": sum(1 for b in blocks if b.block_type == BlockType.ATTACHMENT),
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
