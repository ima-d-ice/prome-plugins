"""Tests for upload validation: filenames, MIME, sizes, hashes, duplicates."""

from __future__ import annotations

import pytest

from loaders.contracts.exceptions import (
    EmptyContentError,
    FileValidationError,
    MaxFileSizeExceededError,
    UnsupportedFormatError,
)
from loaders.uploads import FileValidator, UploadScope
from loaders.uploads.validation import (
    detect_magic,
    sniff_content_kind,
)
from tests.loaders.conftest import make_uploaded_file


def test_sanitize_filename_removes_path_separators():
    v = FileValidator()
    assert v.sanitize_filename("../../etc/passwd") == "etc_passwd"
    assert v.sanitize_filename("..\\..\\windows\\system32\\cmd.exe") == "windows_system32_cmd.exe"
    assert v.sanitize_filename("a/b\\c\x00d.txt") == "a_b_cd.txt"


def test_sanitize_filename_collapses_and_defaults():
    v = FileValidator()
    assert v.sanitize_filename("   ") == "upload"
    assert v.sanitize_filename("") == "upload"
    assert ".." not in v.sanitize_filename("....//....//evil.txt")


def test_sanitize_filename_strips_control_chars():
    v = FileValidator()
    assert v.sanitize_filename("a\x01b\x7f.txt") == "ab.txt"


def test_validate_sanitized_filename_blocks_traversal():
    v = FileValidator()
    ok, sanitized, _warnings = v.validate_sanitized_filename("../../etc/passwd")
    assert ok  # sanitized to a safe name rather than executed as traversal
    assert ".." not in sanitized


def test_extract_extension():
    assert FileValidator.extract_extension("a.PDF") == ".pdf"
    assert FileValidator.extract_extension("noext") == ""
    assert FileValidator.extract_extension("archive.tar.gz") == ".gz"


def test_max_file_size_rejected():
    v = FileValidator(max_file_size=10)
    file = make_uploaded_file("big.txt", data=b"x" * 100)
    with pytest.raises(MaxFileSizeExceededError):
        v.validate_file(file)


def test_empty_file_rejected():
    v = FileValidator()
    file = make_uploaded_file("empty.txt", data=b"")
    file.file_size = 0
    with pytest.raises(EmptyContentError):
        v.validate_file(file)


def test_unsupported_extension_rejected():
    v = FileValidator()
    file = make_uploaded_file("evil.exe", data=b"MZ-bytes")
    with pytest.raises(UnsupportedFormatError):
        v.validate_file(file)


def test_magic_bytes_mismatch_rejected():
    v = FileValidator()
    # .pdf extension but not a PDF payload
    file = make_uploaded_file("fake.pdf", data=b"this is not a pdf")
    with pytest.raises(FileValidationError):
        v.validate_file(file)


def test_magic_bytes_ok_for_pdf():
    v = FileValidator()
    file = make_uploaded_file("doc.pdf", data=b"%PDF-1.4\nbody")
    ok, errors = v.validate_file(file)
    assert ok
    assert errors == []


def test_magic_bytes_ok_for_png():
    v = FileValidator()
    file = make_uploaded_file("img.png", data=b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    ok, _ = v.validate_file(file)
    assert ok


def test_detect_magic_text_formats_pass_through():
    assert detect_magic(".txt", b"any text")
    assert detect_magic(".md", b"# heading")
    assert detect_magic(".json", b"{}")
    assert not detect_magic(".pdf", b"not a pdf")
    assert sniff_content_kind(b"%PDF-1.4\nx") == "pdf"
    assert sniff_content_kind(b"{\"a\": 1}") == "json"


def test_validate_request_requires_conversation_id():
    v = FileValidator()
    from loaders.uploads import UploadRequest

    req = UploadRequest(
        request_id="r1",
        scope=UploadScope.CONVERSATION,
        conversation_id=None,
        files=[make_uploaded_file("a.txt")],
    )
    ok, errors = v.validate_request(req)
    assert not ok
    assert any("Conversation ID" in e for e in errors)


def test_validate_request_supports_knowledge_base_scope():
    v = FileValidator()
    from loaders.uploads import UploadRequest

    req = UploadRequest(
        request_id="r1",
        scope=UploadScope.KNOWLEDGE_BASE,
        knowledge_base_id="kb-1",
        files=[make_uploaded_file("a.txt")],
    )
    ok, _errors = v.validate_request(req)
    assert ok


def test_validate_request_batch_limits():
    v = FileValidator(max_files_per_batch=2)
    from loaders.uploads import UploadRequest

    files = [make_uploaded_file(f"f{i}.txt", file_id=f"fid-{i}") for i in range(3)]
    req = UploadRequest(request_id="r1", conversation_id="c1", files=files)
    ok, errors = v.validate_request(req)
    assert not ok
    assert any("Files per batch" in e for e in errors)


def test_duplicate_detection_same_scope():
    v = FileValidator()
    a = make_uploaded_file("a.txt", data=b"hello", file_id="a")
    b = make_uploaded_file("b.txt", data=b"hello", file_id="b")
    existing = {"a": a}
    is_dup, existing_id = v.check_duplicate_content(b, existing)
    assert is_dup
    assert existing_id == "a"


def test_validate_mime_type_allowlist():
    v = FileValidator()
    assert v.validate_mime_type("text/plain; charset=utf-8")
    assert not v.validate_mime_type("application/x-msdownload")
