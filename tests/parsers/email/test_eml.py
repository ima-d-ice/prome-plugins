from pathlib import Path
import pytest
from parsers.contracts import (
    BlockType,
    CorruptDocumentError,
    EmptyDocumentError,
    ParsedDocument,
)
from parsers.email.eml import EmlParser

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "parsers" / "email" / "fixtures"


def test_eml_simple_plain():
    parser = EmlParser()
    doc = parser.parse(FIXTURES_DIR / "simple_plain.eml")
    assert isinstance(doc, ParsedDocument)
    assert doc.parser_name == "eml_stdlib"
    assert doc.metadata["subject"] == "Quarterly Sync"
    assert "alice@example.com" in doc.metadata["from"]
    assert doc.metadata["is_thread_reply"] is False
    assert len(doc.blocks) >= 1
    assert "meet tomorrow at 10 AM" in doc.text


def test_eml_html():
    parser = EmlParser()
    doc = parser.parse(FIXTURES_DIR / "html_email.eml")
    assert doc.metadata["subject"] == "Welcome to Prome"
    assert any("Welcome!" in b.content for b in doc.blocks)


def test_eml_multipart_attachment():
    parser = EmlParser()
    doc = parser.parse(FIXTURES_DIR / "multipart_attachment.eml")
    assert doc.metadata["attachment_count"] == 1
    attach_blocks = [b for b in doc.blocks if b.block_type == BlockType.ATTACHMENT]
    assert len(attach_blocks) == 1
    assert attach_blocks[0].metadata["filename"] == "statement_nov.pdf"
    assert "application/pdf" in attach_blocks[0].metadata["content_type"]


def test_eml_thread_reply():
    parser = EmlParser()
    doc = parser.parse(FIXTURES_DIR / "thread_reply.eml")
    assert doc.metadata["is_thread_reply"] is True
    assert doc.metadata["in_reply_to"] == "<msg-001@example.com>"
    assert "<msg-001@example.com>" in doc.metadata["references_list"]


def test_eml_unicode():
    parser = EmlParser()
    doc = parser.parse(FIXTURES_DIR / "unicode.eml")
    assert "Café et résumé" in doc.metadata["subject"]
    assert "こんにちは" in doc.text


def test_eml_empty():
    parser = EmlParser()
    with pytest.raises(EmptyDocumentError):
        parser.parse(FIXTURES_DIR / "empty.eml")
