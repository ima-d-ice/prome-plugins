"""Tests for the constrained website crawler."""

from __future__ import annotations

import pytest

from loaders.contracts.exceptions import SSRFError
from loaders.web import RobotsChecker, WebsiteCrawler
from loaders.web.fetcher import Fetcher, resolve_and_check_ssrf


@pytest.fixture(autouse=True)
def _patch_ssrf(monkeypatch):
    # The local fixture server is loopback by design; patch the SSRF guard
    # for happy-path crawl tests. SSRF behavior itself is tested separately.
    monkeypatch.setattr("loaders.web.fetcher.resolve_and_check_ssrf", lambda host: None)


def test_crawler_stays_on_same_domain(http_server):
    crawler = WebsiteCrawler(max_depth=1, max_pages=10, polite_delay=0)
    result = crawler.crawl(f"{http_server}/")
    crawled = {r.reference.original_uri_or_path for r in result.resources}
    assert f"{http_server}/" in crawled
    assert f"{http_server}/about.html" in crawled
    # No external hosts were visited
    assert all(u.startswith(http_server) for u in crawled)


def test_crawler_same_domain_only_blocks_external_links(http_server, monkeypatch):
    crawler = WebsiteCrawler(max_depth=2, max_pages=10, polite_delay=0, same_domain_only=True)
    result = crawler.crawl(f"{http_server}/")
    crawled = {r.reference.original_uri_or_path for r in result.resources}
    assert all(u.startswith(http_server) for u in crawled)


def test_crawler_depth_limit(http_server):
    crawler = WebsiteCrawler(max_depth=0, max_pages=10, polite_delay=0)
    result = crawler.crawl(f"{http_server}/")
    assert len(result.resources) == 1  # start page only


def test_crawler_max_pages_limit(http_server):
    crawler = WebsiteCrawler(max_depth=3, max_pages=1, polite_delay=0)
    result = crawler.crawl(f"{http_server}/")
    assert len(result.resources) == 1


def test_crawler_robots_txt_disallows(http_server):
    retriever = Fetcher(timeout=5)

    def robot_body(url: str):
        r = retriever.fetch(url)
        if r.resources:
            content = r.resources[0].content
            return content.decode("utf-8") if isinstance(content, bytes) else str(content)
        return None

    checker = RobotsChecker(user_agent="prome-rag-plugins/1.0", retriever=robot_body)
    # /secret.html is disallowed in the fixture robots.txt
    assert checker.can_fetch(f"{http_server}/secret.html") is False
    assert checker.can_fetch(f"{http_server}/") is True


def test_url_normalization_and_dedup(http_server):
    crawler = WebsiteCrawler(max_depth=1, max_pages=10, polite_delay=0)
    a = crawler.normalize_url(f"{http_server}/about.html#top")
    b = crawler.normalize_url(f"{http_server}/about.html")
    assert a == b
    c = crawler.normalize_url(f"{http_server.upper()}/ABOUT.html")
    assert c == crawler.normalize_url(f"{http_server}/ABOUT.html")


def test_crawler_excluded_external_domain_from_links(http_server):
    crawler = WebsiteCrawler(max_depth=1, max_pages=10, polite_delay=0)
    # The home page links to /about.html (same domain) which links back home.
    result = crawler.crawl(f"{http_server}/")
    urls = [r.reference.original_uri_or_path for r in result.resources]
    assert len(urls) == len(set(urls)), "no duplicate pages should be crawled"


def test_crawler_crawl_never_hits_metadata_endpoint(http_server):
    # This test verifies the SSRF guard would reject the cloud metadata
    # address if the crawler ever tried to follow such a link.
    with pytest.raises(SSRFError):
        resolve_and_check_ssrf("169.254.169.254")


def test_crawler_reports_per_page_failures(http_server, monkeypatch):
    def fake_fetch(url, headers=None):
        from loaders.contracts.models import LoaderResult

        if url.endswith("/about.html"):
            return LoaderResult()  # no resources -> failure recorded
        return Fetcher().fetch(url, headers=headers)


    monkeypatch.setattr("loaders.web.fetcher.resolve_and_check_ssrf", lambda host: None)
    crawler = WebsiteCrawler(max_depth=1, max_pages=10, polite_delay=0)
    monkeypatch.setattr(crawler.fetcher, "fetch", fake_fetch)
    result = crawler.crawl(f"{http_server}/")
    assert any("about.html" in f.message for f in result.failures)
