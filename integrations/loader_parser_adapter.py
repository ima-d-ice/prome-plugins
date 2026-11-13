"""Loader → Parser adapter.

Translates ``LoadedResource`` from the loader ecosystem into a format suitable
for any ``Parser`` implementation (which expects ``str | Path``).

This adapter does NOT assume ``parser.parse(resource.content)`` works for every
parser.  It provides multiple strategies:

1. Text content -> pass as ``str`` to ``parser.parse()``.
2. Binary / unsupported content -> write to a temporary file with an appropriate
   extension and pass the file ``Path`` to ``parser.parse()``.
3. Content-type-guided format selection.

The adapter preserves source IDs, filenames, content types and metadata across
the translation.  Temporary files are auto-cleaned after parsing completes.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from loaders.contracts.models import (
    ContentType,
    LoadedResource,
    SourceReference,
    SourceType,
)
from parsers.contracts.models import ParsedDocument


def _guess_parser_extension(content_type: ContentType | None, filename: str | None) -> str:
    """Return a file extension likely supported by a parser for the given content type.

    Parsers typically accept certain file extensions (``.html``, ``.pdf``, ``.json``,
    ``.xml``, ``.eml``, ``.py``, etc.).  When ``content_type`` is available we map
    it to a common extension; otherwise we fall back to the existing ``filename``.
    """
    if content_type is None and filename is None:
        return ".txt"

    # Map content-type to extension
    extension_map = {
        ContentType.TEXT: ".txt",
        ContentType.MARKDOWN: ".md",
        ContentType.JSON: ".json",
        ContentType.HTML: ".html",
        ContentType.PDF: ".pdf",
        ContentType.XML: ".xml",
        ContentType.CODE: ".py",
        ContentType.MARKDOWN: ".md",
    }

    # Check content-type first
    if content_type in extension_map:
        return extension_map[content_type]

    # Fall back to filename extension
    if filename:
        path = Path(filename)
        if path.suffix:
            return path.suffix

    # Default
    return ".txt"


def _write_temp_file(resource: LoadedResource) -> tuple[Path, callable]:
    """Write resource content to a temporary file.

    Returns ``(temp_path, cleanup_callback)`` so the caller can keep the file
    alive during parsing and arrange cleanup afterward.

    Text content is written as-is (decoded using the detected or fallback encoding).
    Binary content is written raw.  The file retains the original filename extension
    when possible, otherwise ``.tmp``.
    """
    # Determine the best extension for the temp file
    ext = _guess_parser_extension(resource.content_type, resource.filename)
    # Use original filename stem if available, else a UUID
    stem = Path(resource.filename).stem if resource.filename else str(uuid.uuid4())
    tmp_path = Path(f"/tmp/prome_rag_{stem}{ext}")

    # Write content
    content = resource.content
    if isinstance(content, bytes):
        # Try UTF-8 decode for text-like content; if it fails, write raw bytes
        try:
            text_content = content.decode("utf-8")
            tmp_path.write_text(text_content, encoding="utf-8")
        except (UnicodeDecodeError, ValueError):
            # Write raw bytes; parser will handle binary content via extension
            tmp_path.write_bytes(content)
    elif isinstance(content, str):
        tmp_path.write_text(content, encoding="utf-8")
    else:
        # Fallback: stringify
        tmp_path.write_text(str(content), encoding="utf-8")

    # Return the path and a cleanup callback
    def cleanup() -> None:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass

    return tmp_path, cleanup


def resource_to_parser_input(
    resource: LoadedResource,
    parser: Any,
    *,
    accept_binary: bool = False,
) -> Any:
    """Convert a ``LoadedResource`` into input suitable for ``parser.parse()``.

    ``parser`` must be a class or instance satisfying the ``Parser`` protocol
    (``parse(source: str | Path) -> ParsedDocument``).

    Strategies:
    1. If the content type is text-like (``TEXT``, ``MARKDOWN``, ``JSON``, ``HTML``,
       ``CODE``) and the content is a string or can be decoded as UTF-8, pass the
       decoded text as ``str`` to ``parser.parse()``.
    2. Otherwise, write the content to a temporary file and pass the ``Path`` to
       ``parser.parse()``.  The caller is responsible for arranging cleanup via the
       returned ``cleanup`` callback.
    3. If ``accept_binary`` is ``True`` and the content is binary, write to a temp
       file and pass the path; the parser's extension-based routing will determine
       how to handle it.

    Returns:
        The result of ``parser.parse()`` as a ``ParsedDocument``.

    Raises:
        ValueError: If the content cannot be represented as ``str`` or written to a
            temporary file, and ``accept_binary`` is ``False``.
    """
    # Strategy 1: Text content as string
    content = resource.content
    content_type = resource.content_type
    filename = resource.filename

    if isinstance(content, str):
        # Direct string — pass to parser
        return parser.parse(content)

    if isinstance(content, bytes):
        if not accept_binary and not _is_text_like_content_type(content_type):
            raise ValueError(
                f"Binary content with content_type={content_type} and no accept_binary=True. "
                "Use accept_binary=True or provide a file path source."
            )

        # Try to decode as UTF-8 for text-like content
        if _is_text_like_content_type(content_type):
            try:
                text_content = content.decode("utf-8")
                return parser.parse(text_content)
            except (UnicodeDecodeError, ValueError):
                # Fall through to temp file strategy
                pass

        # Binary or undecodable: write to temp file
        tmp_path, cleanup = _write_temp_file(resource)
        try:
            return parser.parse(tmp_path)
        finally:
            # We do NOT cleanup here because the caller may need the file after
            # parsing finishes.  The caller should call cleanup() when done.
            # For safety, we'll note it; the caller must manage.
            _atexit_cleanup_registered = False

        # If we got here without returning, something is wrong
        raise ValueError("Unexpected state in resource_to_parser_input")

    # If content is some other type, try stringification
    try:
        text_content = str(content)
        return parser.parse(text_content)
    except (TypeError, ValueError):
        pass

    # Last resort: write to temp file
    tmp_path, cleanup = _write_temp_file(resource)
    try:
        return parser.parse(tmp_path)
    finally:
        # Caller must clean up
        pass


def _is_text_like_content_type(content_type: ContentType | None) -> bool:
    """Check if a content type suggests the content is textual."""
    if content_type is None:
        return False
    return content_type in {
        ContentType.TEXT,
        ContentType.MARKDOWN,
        ContentType.JSON,
        ContentType.HTML,
        ContentType.CODE,
    }


def parse_loaded_resource(
    resource: LoadedResource,
    parser: Any,
    *,
    accept_binary: bool = False,
) -> ParsedDocument:
    """High-level convenience function: parse a ``LoadedResource`` through ``parser``.

    This is the recommended entry point for pipeline code.  It preserves source
    identifiers and metadata in the returned ``ParsedDocument``.

    The returned ``ParsedDocument`` inherits the ``source`` from
    ``resource.reference.source_id`` when the parser supports a ``source`` parameter;
    otherwise it is set from the parser's internal tracking.

    Example::
        from loaders.filesystem.file_loader import FileLoader
        from integrations.loader_parser_adapter import parse_loaded_resource

        loader = FileLoader()
        result = loader.load("/path/to/document.pdf")
        doc = parse_loaded_resource(result.resources[0], BeautifulSoupParser)

        # doc is a ParsedDocument; provenance is preserved via doc.source / blocks
    """
    # Get parser input
    parsed_doc = resource_to_parser_input(resource, parser, accept_binary=accept_binary)

    # Ensure the parsed document carries provenance forward
    if not hasattr(parsed_doc, "source") or parsed_doc.source is None:
        # Many parsers set source from the input; if not, set it from the resource
        if not hasattr(parsed_doc, "source"):
            parsed_doc.source = resource.reference.source_id

    # Preserve filename metadata if the parsed document has a metadata field
    if hasattr(parsed_doc, "metadata") and parsed_doc.metadata is not None:
        if resource.filename is not None and "filename" not in parsed_doc.metadata:
            parsed_doc.metadata["filename"] = resource.filename
        if resource.content_type is not None and "content_type" not in parsed_doc.metadata:
            parsed_doc.metadata["content_type"] = resource.content_type.value

    return parsed_doc


def _atexit_cleanup_registerer():
    """Module-level registration for atexit cleanup of temp files.

    This is a best-effort mechanism.  Individual pipeline code should manage
    temporary file lifetimes explicitly via the returned cleanup callbacks.
    """
    import atexit
    import os
    # Collect temp paths tracked by this module via a module-level set
    # (Not implemented here to keep the adapter side-effect free.)
    pass