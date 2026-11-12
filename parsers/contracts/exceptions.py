"""Common exceptions for Prome RAG parsers."""


class ParserError(Exception):
    """Base class for all parser-related errors."""


class ParserDependencyError(ParserError):
    """A required optional dependency for a parser is not installed."""


class DocumentParseError(ParserError):
    """Base class for failures while parsing a document."""


class CorruptDocumentError(DocumentParseError):
    """The document is unreadable, truncated or not a valid file."""


class EmptyDocumentError(DocumentParseError):
    """The document contains no pages or no extractable content."""


class EncryptedDocumentError(DocumentParseError):
    """The document is password-protected and cannot be opened."""
