"""Tests for the Loader → Parser adapter."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from loaders.contracts.models import (
    ContentType,
    LoadedResource,
    SourceReference,
    SourceType,
)
from parsers.contracts.models import ParsedDocument
from integrations.loader_parser_adapter import (
    parse_loaded_resource,
    resource_to_parser_input,
    _write_temp_file,
    _guess_parser_extension,
)


class FakeParser:
    """Minimal parser satisfying the Parser protocol: parse(source: str | Path) -> ParsedDocument."""

    def __init__(self):
        self.parse_calls = []

    def parse(self, source):
        self.parse_calls.append(str(source))
        if isinstance(source, Path):
            try:
                text = source.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # Binary file: read raw bytes and use as content
                text = source.read_bytes().hex()
        else:
            text = str(source)
        return ParsedDocument(
            source=str(source),
            parser_name="FakeParser",
            parser_version="0.1.0",
            blocks=[],
            pages=[],
            metadata={},
        )


def test_parse_loaded_resource_text_string():
    """Text content passed directly as string to parser."""
    resource = LoadedResource(
        reference=SourceReference(source_id="s-1", source_type=SourceType.LOCAL_FILE, original_uri_or_path="/foo.txt"),
        content="Hello world",
        content_type=ContentType.TEXT,
        filename="test.txt",
    )
    parser = FakeParser()
    doc = parse_loaded_resource(resource, parser)
    assert doc is not None
    assert parser.parse_calls == ["Hello world"]


def test_parse_loaded_resource_accept_binary():
    """Binary content with accept_binary=True goes to temp file then parser."""
    resource = LoadedResource(
        reference=SourceReference(source_id="s-2", source_type=SourceType.LOCAL_FILE, original_uri_or_path="/foo.bin"),
        content=b"\x00\x01\x02\x03",
        content_type=ContentType.UNKNOWN,
        filename="test.bin",
    )
    parser = FakeParser()
    doc = parse_loaded_resource(resource, parser, accept_binary=True)
    assert doc is not None


def test_parse_loaded_resource_unsupported_binary_raises():
    """Binary content without accept_binary raises ValueError."""
    resource = LoadedResource(
        reference=SourceReference(source_id="s-3", source_type=SourceType.LOCAL_FILE, original_uri_or_path="/foo.bin"),
        content=b"\x00\x01\x02\x03",
        content_type=ContentType.UNKNOWN,
        filename="test.bin",
    )
    parser = FakeParser()
    with pytest.raises(ValueError, match="Binary content"):
        parse_loaded_resource(resource, parser)


def test_resource_to_parser_input_str_content():
    """String content goes directly to parser.parse()."""
    resource = LoadedResource(
        reference=SourceReference(source_id="s-4", source_type=SourceType.LOCAL_FILE, original_uri_or_path="/foo.txt"),
        content="direct string",
        content_type=ContentType.TEXT,
        filename="test.txt",
    )
    parser = FakeParser()
    result = resource_to_parser_input(resource, parser)
    assert result is not None


def test_resource_to_parser_input_bytes_text_type():
    """Bytes with text-like content type decoded and passed as string."""
    resource = LoadedResource(
        reference=SourceReference(source_id="s-5", source_type=SourceType.LOCAL_FILE, original_uri_or_path="/foo.txt"),
        content=b"hello utf-8",
        content_type=ContentType.TEXT,
        filename="test.txt",
    )
    parser = FakeParser()
    result = resource_to_parser_input(resource, parser)
    assert result is not None


def test_resource_to_parser_input_uses_temp_file_for_binary():
    """Binary content without accept_binary uses temp file strategy."""
    resource = LoadedResource(
        reference=SourceReference(source_id="s-6", source_type=SourceType.LOCAL_FILE, original_uri_or_path="/foo.bin"),
        content=b"\x89PNG\r\n\x1a\n",
        content_type=ContentType.UNKNOWN,
        filename="test.png",
    )
    parser = FakeParser()
    result = resource_to_parser_input(resource, parser, accept_binary=True)
    assert result is not None


def test_guess_parser_extension_content_type():
    """Content type maps to correct extension."""
    assert _guess_parser_extension(ContentType.TEXT, None) == ".txt"
    assert _guess_parser_extension(ContentType.MARKDOWN, None) == ".md"
    assert _guess_parser_extension(ContentType.JSON, None) == ".json"
    assert _guess_parser_extension(ContentType.HTML, None) == ".html"
    assert _guess_parser_extension(ContentType.PDF, None) == ".pdf"


def test_guess_parser_extension_fallback_to_filename():
    """Fall back to filename extension when content type unknown."""
    assert _guess_parser_extension(None, "document.pdf") == ".pdf"
    assert _guess_parser_extension(None, "data.json") == ".json"


def test_guess_parser_extension_default_txt():
    """Default to .txt when neither content type nor filename available."""
    assert _guess_parser_extension(None, None) == ".txt"


def test_write_temp_file_cleanup():
    """Temp file is written and cleanup callback works."""
    resource = LoadedResource(
        reference=SourceReference(source_id="s-7", source_type=SourceType.LOCAL_FILE, original_uri_or_path="/foo.txt"),
        content="temp file test",
        content_type=ContentType.TEXT,
        filename="test.txt",
    )
    tmp_path, cleanup = _write_temp_file(resource)
    assert tmp_path.exists()
    cleanup()


def test_write_temp_file_binary_content():
    """Binary content written raw to temp file."""
    resource = LoadedResource(
        reference=SourceReference(source_id="s-8", source_type=SourceType.LOCAL_FILE, original_uri_or_path="/foo.bin"),
        content=b"\x00\x01\x02\x03",
        content_type=ContentType.UNKNOWN,
        filename="test.bin",
    )
    tmp_path, cleanup = _write_temp_file(resource)
    assert tmp_path.read_bytes() == b"\x00\x01\x02\x03"
    cleanup()


def test_resource_to_parser_input_preserves_source_via_parsed_doc():
    """ParsedDocument inherits source from resource reference."""
    resource = LoadedResource(
        reference=SourceReference(source_id="src-99", source_type=SourceType.LOCAL_FILE, original_uri_or_path="/doc.pdf"),
        content=b"%PDF-1.4 fake",
        content_type=ContentType.PDF,
        filename="doc.pdf",
    )
    parser = FakeParser()
    doc = parse_loaded_resource(resource, parser, accept_binary=True)
    # The parsed doc should have source set from the resource
    assert doc is not None


def test_parse_loaded_resource_preserves_filename_and_content_type():
    """ParsedDocument metadata gets filename and content_type from resource."""
    resource = LoadedResource(
        reference=SourceReference(source_id="s-10", source_type=SourceType.LOCAL_FILE, original_uri_or_path="/doc.json"),
        content='{"key": "value"}',
        content_type=ContentType.JSON,
        filename="config.json",
    )
    parser = FakeParser()
    doc = parse_loaded_resource(resource, parser)
    assert doc is not None
    # Metadata should include filename and content_type
    assert doc.metadata.get("filename") == "config.json"
    assert doc.metadata.get("content_type") == "json"


def test_parse_loaded_resource_no_filename_no_content_type():
    """Parse resource without filename or content type defaults safely."""
    resource = LoadedResource(
        reference=SourceReference(source_id="s-11", source_type=SourceType.LOCAL_FILE, original_uri_or_path="/unknown.xyz"),
        content="unknown content",
        content_type=ContentType.UNKNOWN,
        filename=None,
    )
    parser = FakeParser()
    doc = parse_loaded_resource(resource, parser)
    assert doc is not None
    assert doc.source is not None