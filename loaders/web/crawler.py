"""Constrained website crawler for Prome RAG Plugins.

SSRF protection is mandatory:
- Starting and discovered URLs are validated against loopback, private,
  link-local, reserved and multicast destinations, plus known cloud
  metadata endpoints.
- Redirects are revalidated after resolution.
- DNS results are validated before fetching.
- Unsupported protocols are blocked.
"""

from __future__ import annotations

import re
import time
import uuid
from collections import deque
from urllib.parse import urljoin, urlparse, urlunparse

from loaders.contracts.exceptions import (
    LoaderValidationError,
    SSRFError,
)
from loaders.contracts.models import (
    ContentType,
    LoaderResult,
    SourceReference,
    SourceType,
)
from loaders.web.fetcher import Fetcher
from loaders.web.robots import RobotsChecker

DEFAULT_MAX_DEPTH = 2
DEFAULT_MAX_PAGES = 50
DEFAULT_TIMEOUT = 30.0
DEFAULT_POLITE_DELAY = 1.0


class WebsiteCrawler:
    """Constrained same-domain website crawler.

    Supports starting URL, domain allowlist, max depth, max pages,
    max response size, request timeout, polite delay, robots.txt, URL
    normalization, duplicate detection, exclusion rules and per-page
    failure reporting.
    """

    def __init__(
        self,
        *,
        fetcher: Fetcher | None = None,
        robots_checker: RobotsChecker | None = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_pages: int = DEFAULT_MAX_PAGES,
        max_response_size: int = 50 * 1024 * 1024,
        same_domain_only: bool = True,
        allowed_domains: set[str] | None = None,
        polite_delay: float = DEFAULT_POLITE_DELAY,
        timeout: float = DEFAULT_TIMEOUT,
        user_agent: str = "prome-rag-plugins/1.0",
    ):
        self.fetcher = fetcher or Fetcher(
            timeout=timeout,
            max_response_size=max_response_size,
            user_agent=user_agent,
        )
        self.robots_checker = robots_checker or RobotsChecker(user_agent=user_agent)
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.max_response_size = max_response_size
        self.same_domain_only = same_domain_only
        self.allowed_domains = {d.lower() for d in (allowed_domains or set())}
        self.polite_delay = polite_delay

        # Crawl state (reset per crawl)
        self.visited: set[str] = set()
        self.queued: set[str] = set()
        self.pages_crawled: int = 0
        self.failures: list[str] = []

    def crawl(self, start_url: str) -> LoaderResult:
        """Crawl from a starting URL."""
        self.visited = set()
        self.queued = set()
        self.pages_crawled = 0
        self.failures = []

        try:
            self.fetcher.validate_url(start_url)
        except SSRFError:
            raise
        except LoaderValidationError as e:
            return LoaderResult().add_failure(str(e), None)

        normalized_start = self.normalize_url(start_url)
        self.queued.add(normalized_start)

        result = LoaderResult()
        queue = deque([(normalized_start, 0)])
        root_host = urlparse(start_url).hostname.lower() if urlparse(start_url).hostname else ""

        while queue and self.pages_crawled < self.max_pages:
            url, depth = queue.popleft()

            if url in self.visited:
                continue
            if depth > self.max_depth:
                continue

            # SSRF revalidation per URL
            if not self._is_ssrf_safe(url):
                self.failures.append(f"SSRF blocked: {url}")
                continue

            # Robots.txt
            if not self.robots_checker.can_fetch(url, user_agent=self.fetcher.user_agent):
                self.failures.append(f"Robots.txt disallows: {url}")
                continue

            # Polite delay (skip before first request)
            if self.pages_crawled > 0:
                time.sleep(self.polite_delay)

            page_result = self.fetcher.fetch(url)
            if not page_result.resources:
                self.failures.append(f"Failed to fetch: {url}")
                self.visited.add(url)
                continue

            resource = page_result.resources[0]
            self.visited.add(url)
            self.pages_crawled += 1

            # Extract outgoing links for HTML pages
            links: list[str] = []
            if resource.content_type == ContentType.HTML and isinstance(resource.content, bytes):
                links = self._extract_links(url, resource.content)

            resource.reference = SourceReference(
                source_id=str(uuid.uuid5(uuid.NAMESPACE_URL, url)),
                source_type=SourceType.CRAWLED,
                original_uri_or_path=url,
                metadata={"depth": depth, "title": self._extract_title(resource.content)},
            )
            result.resources.append(resource)

            # Enqueue links if depth allows
            if depth < self.max_depth:
                for link in links:
                    normalized = self.normalize_url(link)
                    if normalized in self.visited or normalized in self.queued:
                        continue
                    if not self._domain_allowed(normalized, root_host):
                        continue
                    if not self._is_ssrf_safe(normalized):
                        continue
                    self.queued.add(normalized)
                    queue.append((normalized, depth + 1))

        result = LoaderResult(resources=result.resources)
        for failure in self.failures:
            result.add_failure(failure)
        return result

    def normalize_url(self, url: str) -> str:
        """Normalize a URL for deduplication."""
        parsed = urlparse(url)
        scheme = parsed.scheme.lower() or "http"
        host = (parsed.hostname or "").lower()
        port = parsed.port
        netloc = host
        if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
            netloc = f"{host}:{port}"

        path = parsed.path or "/"
        # Remove trailing slash for dedup except root
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")

        # Sort query parameters for stable comparison
        query = ""
        if parsed.query:
            pairs = sorted(parsed.query.split("&"))
            query = "&".join(pairs)

        return urlunparse((scheme, netloc, path, "", query, ""))

    def _domain_allowed(self, url: str, root_host: str) -> bool:
        """Check whether a URL's domain is allowed."""
        host = (urlparse(url).hostname or "").lower()
        if self.allowed_domains:
            return host in self.allowed_domains
        if self.same_domain_only:
            return host == root_host
        return True

    def _is_ssrf_safe(self, url: str) -> bool:
        """Check if a URL resolves only to safe destinations."""
        try:
            self.fetcher.validate_url(url)
            return True
        except (SSRFError, LoaderValidationError, ValueError, OSError):
            return False

    @staticmethod
    def _extract_links(base_url: str, content: bytes) -> list[str]:
        """Extract href/src links from HTML content."""
        text = content.decode("utf-8", errors="ignore")
        found: list[str] = []
        for match in re.finditer(r'(?:href|src)\s*=\s*["\']([^"\']+)["\']', text, re.IGNORECASE):
            link = match.group(1).strip()
            if not link or link.startswith(("#", "mailto:", "javascript:", "tel:", "data:")):
                continue
            absolute = urljoin(base_url, link)
            parsed = urlparse(absolute)
            if parsed.scheme.lower() in ("http", "https"):
                found.append(absolute)
        return found

    @staticmethod
    def _extract_title(content) -> str | None:
        """Extract <title> from HTML content."""
        if not isinstance(content, bytes):
            return None
        match = re.search(rb"<title[^>]*>(.*?)</title>", content, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip().decode("utf-8", errors="ignore")
        return None
