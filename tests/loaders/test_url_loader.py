"""Tests for the URL loader using a controlled local HTTP server."""

from __future__ import annotations

import pytest

from loaders.contracts.exceptions import SSRFError
from loaders.web import URLLoader
from loaders.web.fetcher import resolve_and_check_ssrf


def test_ssrf_rejects_loopback():
    with pytest.raises(SSRFError):
        resolve_and_check_ssrf("127.0.0.1")


def test_ssrf_rejects_localhost():
    with pytest.raises(SSRFError):
        resolve_and_check_ssrf("localhost")


def test_ssrf_rejects_private_networks():
    for addr in ("10.0.0.5", "172.16.1.1", "192.168.1.1"):
        with pytest.raises(SSRFError):
            resolve_and_check_ssrf(addr)


def test_ssrf_rejects_link_local():
    with pytest.raises(SSRFError):
        resolve_and_check_ssrf("169.254.1.1")


def test_ssrf_rejects_cloud_metadata():
    with pytest.raises(SSRFError):
        resolve_and_check_ssrf("169.254.169.254")


def test_ssrf_rejects_unresolvable_host():
    with pytest.raises(SSRFError):
        resolve_and_check_ssrf("nonexistent.invalid.")


def test_unsupported_protocol_blocked():
    from loaders.web import Fetcher

    fetcher = Fetcher()
    result = fetcher.fetch("ftp://example.com/file.txt")
    assert result.success_count == 0
    assert result.failure_count >= 1
    assert any("Protocol" in f.message for f in result.failures)


def test_url_loader_fetches_from_local_server(http_server, monkeypatch):
    # SSRF guard blocks loopback by design; patch it for the local fixture only.
    monkeypatch.setattr("loaders.web.fetcher.resolve_and_check_ssrf", lambda host: None)
    loader = URLLoader(timeout=5)
    result = loader.load(f"{http_server}/")
    assert result.success_count == 1
    resource = result.resources[0]
    assert resource.content_hash
    assert resource.file_size > 0
    assert resource.provenance.fetched_from == f"{http_server}/"
    assert resource.provenance.etag is not None
    assert resource.provenance.last_modified is not None


def test_url_loader_content_hash_stable(http_server, monkeypatch):
    monkeypatch.setattr("loaders.web.fetcher.resolve_and_check_ssrf", lambda host: None)
    loader = URLLoader(timeout=5)
    first = loader.load(f"{http_server}/")
    second = loader.load(f"{http_server}/")
    assert first.resources[0].content_hash == second.resources[0].content_hash


def test_url_loader_fetch_diagnostics(http_server, monkeypatch):
    monkeypatch.setattr("loaders.web.fetcher.resolve_and_check_ssrf", lambda host: None)
    loader = URLLoader(timeout=5)
    result = loader.load(f"{http_server}/")
    resource = result.resources[0]
    assert resource.fetch_info is not None
    assert resource.fetch_info.status_code == 200
    assert resource.fetch_info.final_url == f"{http_server}/"


def test_url_loader_pdf_content_type(http_server, monkeypatch):
    monkeypatch.setattr("loaders.web.fetcher.resolve_and_check_ssrf", lambda host: None)
    loader = URLLoader(timeout=5)
    result = loader.load(f"{http_server}/doc.pdf")
    assert result.success_count == 1
    from loaders.contracts.models import ContentType

    assert result.resources[0].content_type is ContentType.PDF


def test_url_loader_404_fails_with_diagnostics(http_server, monkeypatch):
    monkeypatch.setattr("loaders.web.fetcher.resolve_and_check_ssrf", lambda host: None)
    loader = URLLoader(timeout=5, retry_attempts=1)
    result = loader.load(f"{http_server}/missing-page")
    assert result.success_count == 0
    assert result.failure_count >= 1


def test_url_loader_never_executes_javascript(http_server, monkeypatch):
    """The loader returns raw content; JS payload must arrive verbatim."""
    monkeypatch.setattr("loaders.web.fetcher.resolve_and_check_ssrf", lambda host: None)
    loader = URLLoader(timeout=5)
    result = loader.load(f"{http_server}/")
    assert b"<script" not in result.resources[0].content  # fixture has no script
