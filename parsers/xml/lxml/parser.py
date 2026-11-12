"""lxml-based XML parser: high-performance, XPath-rich, namespace-aware XML extraction.

Security configuration:
- resolve_entities=False (XXE prevention)
- load_dtd=False (DTD loading disabled)
- no_network=True (network access prohibited)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from parsers.contracts import (
    BlockType,
    CorruptDocumentError,
    EmptyDocumentError,
    ParsedBlock,
    ParsedDocument,
    ParserDependencyError,
)
from parsers.contracts.validation import validate_source

logger = logging.getLogger(__name__)


def _load_lxml():
    try:
        import lxml.etree as etree

        return etree
    except ImportError as exc:
        raise ParserDependencyError(
            "lxml is not installed. Install it with:\n"
            "  pip install 'prome-rag-plugins[xml]'\n"
            "or:\n"
            "  pip install lxml"
        ) from exc


def _split_tag(tag: str) -> tuple[str, str | None]:
    if isinstance(tag, str) and tag.startswith("{") and "}" in tag:
        ns, local = tag[1:].split("}", 1)
        return local, ns
    return str(tag), None


class LxmlXmlParser:
    """Extract XML content with native XPath support and XXE protection using lxml."""

    name = "lxml_xml"

    def parse(self, source: str | Path) -> ParsedDocument:
        etree = _load_lxml()
        path = validate_source(source)

        if path.stat().st_size == 0:
            raise EmptyDocumentError(f"XML file '{path}' is empty (0 bytes).")

        try:
            content_bytes = path.read_bytes()
        except Exception as exc:
            raise CorruptDocumentError(f"Could not read XML file '{path}': {exc}") from exc

        # Security-hardened parser: XXE prevention and no network
        xml_parser = etree.XMLParser(
            resolve_entities=False,
            load_dtd=False,
            no_network=True,
            remove_comments=True,
        )

        try:
            root = etree.fromstring(content_bytes, parser=xml_parser)
        except Exception as exc:
            raise CorruptDocumentError(f"Could not parse '{path}' as valid XML: {exc}") from exc

        if root is None or (len(root) == 0 and not (root.text and root.text.strip()) and not root.attrib):
            raise EmptyDocumentError(f"XML file '{path}' contains an empty root element.")

        tree = etree.ElementTree(root)
        blocks: list[ParsedBlock] = []
        element_count = 0

        for elem in root.iter():
            if not isinstance(elem.tag, str):  # e.g. comments or processing instructions
                continue
            element_count += 1
            xpath = tree.getpath(elem)
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
                blocks.append(
                    ParsedBlock(
                        block_type=BlockType.KEY_VALUE,
                        content=f"{xpath} = {text}",
                        metadata={**meta, "value": text},
                    )
                )
            elif attribs or text:
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
        metadata = {
            "root_tag": root_local,
            "root_namespace": root_ns,
            "element_count": element_count,
            "block_count": len(blocks),
            "lxml_version": str(getattr(etree, "__version__", "unknown")),
        }

        return ParsedDocument(
            source=str(path),
            parser_name=self.name,
            parser_version=str(getattr(etree, "__version__", "unknown")),
            metadata=metadata,
            pages=[],
            blocks=blocks,
            warnings=[],
        )
