"""Shared contracts for Prome RAG parsers."""

from parsers.contracts.exceptions import (
    CorruptDocumentError,
    DocumentParseError,
    EmptyDocumentError,
    EncryptedDocumentError,
    ParserDependencyError,
    ParserError,
)
from parsers.contracts.models import (
    BlockType,
    BoundingBox,
    ParsedBlock,
    ParsedDocument,
    ParsedPage,
)
from parsers.contracts.protocols import Parser

__all__ = [
    "BlockType",
    "BoundingBox",
    "CorruptDocumentError",
    "DocumentParseError",
    "EmptyDocumentError",
    "EncryptedDocumentError",
    "ParsedBlock",
    "ParsedDocument",
    "ParsedPage",
    "Parser",
    "ParserDependencyError",
    "ParserError",
]
