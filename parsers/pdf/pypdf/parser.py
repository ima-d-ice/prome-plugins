"""PyPDF-based PDF parser: lightweight text extraction alternative."""

from __future__ import annotations

import logging
from pathlib import Path

from parsers.contracts import (
    BlockType,
    CorruptDocumentError,
    EmptyDocumentError,
    EncryptedDocumentError,
    ParsedBlock,
    ParsedDocument,
    ParsedPage,
    ParserDependencyError,
)
from parsers.contracts.validation import validate_source

logger = logging.getLogger(__name__)


def _load_pypdf():
    try:
        import pypdf

        return pypdf
    except ImportError as exc:
        raise ParserDependencyError(
            "pypdf is not installed. Install it with:\n"
            "  pip install 'prome-rag-plugins[pypdf]'\n"
            "or:\n"
            "  pip install pypdf"
        ) from exc


def _page_dimensions(page) -> tuple[float | None, float | None]:
    try:
        mediabox = page.mediabox
        return float(mediabox.width), float(mediabox.height)
    except Exception:
        return None, None


class PyPDFParser:
    name = "pypdf"

    def parse(self, source: str | Path) -> ParsedDocument:
        pypdf = _load_pypdf()
        path = validate_source(source)

        try:
            reader = pypdf.PdfReader(str(path))
        except Exception as exc:
            raise CorruptDocumentError(f"Could not open '{path}' as a PDF: {exc}") from exc

        try:
            if reader.is_encrypted:
                try:
                    result = reader.decrypt("")
                except Exception as exc:
                    raise EncryptedDocumentError(
                        f"PDF '{path}' is password-protected and could not be decrypted: {exc}"
                    ) from exc
                if result == 0:
                    raise EncryptedDocumentError(
                        f"PDF '{path}' is password-protected; encrypted PDFs are not supported."
                    )

            num_pages = len(reader.pages)
            if num_pages == 0:
                raise EmptyDocumentError(f"PDF '{path}' contains no pages.")

            raw_metadata = reader.metadata or {}
            metadata = {}
            for key, value in raw_metadata.items():
                if value in (None, ""):
                    continue
                metadata[str(key).lstrip("/")] = str(value)

            pages: list[ParsedPage] = []
            warnings: list[str] = []
            for index in range(num_pages):
                page_number = index + 1
                page = None
                try:
                    page = reader.pages[index]
                    text = (page.extract_text() or "").strip()
                except Exception as exc:
                    logger.warning("Failed to extract page %s of '%s': %s", page_number, path, exc)
                    text = ""
                    warnings.append(f"Page {page_number} could not be extracted: {exc}")

                width, height = _page_dimensions(page) if page is not None else (None, None)
                blocks = []
                if text:
                    blocks.append(
                        ParsedBlock(
                            block_type=BlockType.TEXT,
                            content=text,
                            page_number=page_number,
                            bbox=None,
                            metadata={},
                        )
                    )
                else:
                    warnings.append(f"Page {page_number} has no extractable text.")

                pages.append(ParsedPage(page_number=page_number, width=width, height=height, blocks=blocks))

            return ParsedDocument(
                source=str(path),
                parser_name=self.name,
                parser_version=str(getattr(pypdf, "__version__", "unknown")),
                metadata=metadata,
                pages=pages,
                warnings=warnings,
            )
        finally:
            try:
                reader.stream.close()
            except Exception:
                pass
