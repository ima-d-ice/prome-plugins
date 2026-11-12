"""ElementTree-based XML parser: safe standard-library extraction.

Preserves XML paths, namespaces, attributes, element hierarchy, and text content.
External entity resolution and network access are disabled.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from parsers.contracts import (
    BlockType,
    CorruptDocumentError,
    EmptyDocumentError,
    ParsedBlock,
    ParsedDocument,
)
from parsers.contracts.validation import validate_source

logger = logging.getLogger(__name__)


def _split_tag(tag: str) -> tuple[str, str | None]:
    if tag.startswith("{") and "}" in tag:
        ns, local = tag[1:].split("}", 1)
        return local, ns
    return tag, None


def _walk_et(elem: ET.Element, path: str = "") -> list[tuple[str, ET.Element]]:
    local, _ = _split_tag(elem.tag)
    current_path = f"{path}/{local}" if path else f"/{local}"
    items = [(current_path, elem)]

    # Count occurrences of child tags for indexed xpath
    child_counts: dict[str, int] = {}
    for child in elem:
        c_local, _ = _split_tag(child.tag)
        child_counts[c_local] = child_counts.get(c_local, 0) + 1

    child_indices: dict[str, int] = {}
    for child in elem:
        c_local, _ = _split_tag(child.tag)
        if child_counts[c_local] > 1:
            child_indices[c_local] = child_indices.get(c_local, 0) + 1
            idx_suffix = f"[{child_indices[c_local]}]"
        else:
            idx_suffix = ""
        items.extend(_walk_et(child, f"{current_path}/{c_local}{idx_suffix}"))
    return items


class ElementTreeXmlParser:
    """Safe, lightweight XML parser using Python's xml.etree.ElementTree."""

    name = "elementtree_xml"

    def parse(self, source: str | Path) -> ParsedDocument:
        path = validate_source(source)

        if path.stat().st_size == 0:
            raise EmptyDocumentError(f"XML file '{path}' is empty (0 bytes).")

        try:
            content_bytes = path.read_bytes()
        except Exception as exc:
            raise CorruptDocumentError(f"Could not read XML file '{path}': {exc}") from exc

        # Security check: disallow <!ENTITY with SYSTEM/PUBLIC external references
        content_lower = content_bytes.lower()
        if b"<!entity" in content_lower and (b"system" in content_lower or b"public" in content_lower):
            raise CorruptDocumentError(f"XML file '{path}' contains forbidden external entity declarations.")

        try:
            root = ET.fromstring(content_bytes)
        except Exception as exc:
            raise CorruptDocumentError(f"Could not parse '{path}' as valid XML: {exc}") from exc

        if root is None or (len(root) == 0 and not (root.text and root.text.strip()) and not root.attrib):
            raise EmptyDocumentError(f"XML file '{path}' contains an empty root element.")

        blocks: list[ParsedBlock] = []
        elem_list = _walk_et(root)

        for xpath, elem in elem_list:
            local, ns = _split_tag(elem.tag)
            text = (elem.text or "").strip()
            attribs = dict(elem.attrib)

            meta: dict[str, Any] = {
                "xpath": xpath,
                "tag": local,
                "namespace": ns,
                "attributes": attribs,
                "has_children": len(elem) > 0,
            }

            if len(elem) == 0 and text:
                # Leaf element with text
                blocks.append(
                    ParsedBlock(
                        block_type=BlockType.KEY_VALUE,
                        content=f"{xpath} = {text}",
                        metadata={**meta, "value": text},
                    )
                )
            elif attribs or text:
                # Element with attributes or mixed text
                desc = f"{xpath}"
                if attribs:
                    desc += f" {attribs}"
                if text:
                    desc += f": {text}"
                blocks.append(
                    ParsedBlock(
                        block_type=BlockType.RECORD,
                        content=desc,
                        metadata={**meta, "value": text},
                    )
                )

        if not blocks:
            raise EmptyDocumentError(f"XML file '{path}' contains no text or attribute content.")

        root_local, root_ns = _split_tag(root.tag)
        metadata: dict[str, Any] = {
            "root_tag": root_local,
            "root_namespace": root_ns,
            "element_count": len(elem_list),
            "block_count": len(blocks),
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
