"""URL loader for Prome RAG Plugins."""

from __future__ import annotations

from pathlib import Path

from loaders.contracts.models import LoaderResult
from loaders.web.fetcher import Fetcher


class URLLoader:
    """Loader for HTTP and HTTPS URLs.

    Downloads raw content; existing HTML/PDF parsers handle extraction.
    Does NOT execute JavaScript or introduce browser automation.
    """

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        max_response_size: int = 50 * 1024 * 1024,
        max_redirects: int = 10,
        retry_attempts: int = 3,
        retry_backoff: float = 1.5,
        ssl_verify: bool = True,
    ):
        self.fetcher = Fetcher(
            timeout=timeout,
            max_response_size=max_response_size,
            max_redirects=max_redirects,
            retry_attempts=retry_attempts,
            retry_backoff=retry_backoff,
            ssl_verify=ssl_verify,
        )

    def load(self, source: str | Path) -> LoaderResult:
        """Load content from a URL."""
        url = str(source)
        return self.fetcher.fetch(url)

    def load_with_etag(
        self,
        url: str,
        if_none_match: str | None = None,
        if_modified_since: str | None = None,
    ) -> tuple[LoaderResult, bool]:
        """Load content with conditional fetch headers.

        Returns (result, not_modified). not_modified is True when the
        server responds that content was unchanged.
        """
        headers = {}
        if if_none_match:
            headers["If-None-Match"] = if_none_match
        if if_modified_since:
            headers["If-Modified-Since"] = if_modified_since

        result = self.fetcher.fetch(url, headers=headers)
        not_modified = bool(result.warnings and "304" in " ".join(result.warnings))
        return result, not_modified
