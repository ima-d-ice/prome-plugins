"""Robots.txt checker for constrained web crawling.

The robots.txt body is fetched through an injected retriever callable
(typically a Fetcher). Results are cached per origin. When no retriever
is provided, or robots.txt cannot be fetched/parsed, the checker fails
open (allows fetching), matching the de-facto robots.txt convention for
inaccessible files.
"""

from __future__ import annotations

import time
import urllib.parse
from collections.abc import Callable

from loaders.contracts.exceptions import SSRFError


class RobotsChecker:
    """Checks robots.txt compliance for a given origin and user agent.

    Supports:
    - Path allow/deny rules (prefix match, longest-match wins).
    - User-agent-specific blocks with a wildcard fallback.
    - Per-origin caching with TTL.
    """

    def __init__(
        self,
        user_agent: str = "prome-rag-plugins/1.0",
        retriever: Callable[[str], str | None] | None = None,
        cache_ttl: float = 3600.0,
    ):
        self.user_agent = user_agent
        self.retriever = retriever
        self.cache_ttl = cache_ttl
        self._cache: dict[str, tuple[float, list[str]]] = {}

    def can_fetch(self, url: str, user_agent: str | None = None) -> bool:
        """Return True if the URL is allowed by robots.txt."""
        parsed = urllib.parse.urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        path = parsed.path or "/"

        rules = self._get_rules(origin)
        if rules is None:
            # robots.txt unavailable -> allow (standard practice)
            return True

        ua = user_agent or self.user_agent
        return self._is_allowed(rules, path, ua)

    def _get_rules(self, origin: str) -> list[str] | None:
        """Get cached robots.txt rules for an origin (None if unavailable)."""
        cached = self._cache.get(origin)
        if cached is not None:
            fetched_at, rules = cached
            if time.time() - fetched_at < self.cache_ttl:
                return rules

        body: str | None = None
        if self.retriever is not None:
            try:
                body = self.retriever(f"{origin}/robots.txt")
            except (OSError, ValueError, SSRFError):
                # robots.txt unavailable/refused -> fail open (allow)
                body = None

        rules = self._parse_rules(body or "")
        self._cache[origin] = (time.time(), rules)
        return rules

    @staticmethod
    def _parse_rules(body: str) -> list[str]:
        """Parse robots.txt into lines (kept raw for _is_allowed)."""
        if not body:
            return []
        return [line.rstrip() for line in body.splitlines() if line.strip()]

    def _is_allowed(self, rules: list[str], path: str, user_agent: str) -> bool:
        """Evaluate parsed rules for a path (longest disallow wins)."""
        if not rules:
            return True

        agent_key = user_agent.lower().split("/")[0]

        # Group rules by user-agent block
        current_agent = "*"
        blocks: dict[str, list[tuple[str, str]]] = {}
        for raw in rules:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip()
            if key == "user-agent":
                current_agent = value.lower()
            elif key in ("disallow", "allow"):
                blocks.setdefault(current_agent, []).append((key, value))

        # Pick exact agent block, else wildcard block
        selected = blocks.get(agent_key) or blocks.get("*") or []

        allow_len = -1
        disallow_len = -1
        for key, pattern in selected:
            if not pattern:
                continue
            if self._matches(pattern, path):
                if key == "allow":
                    allow_len = max(allow_len, len(pattern))
                else:
                    disallow_len = max(disallow_len, len(pattern))

        if disallow_len == -1:
            return True
        # Longest match wins; ties go to allow (standard practice)
        return allow_len >= disallow_len

    @staticmethod
    def _matches(pattern: str, path: str) -> bool:
        """Prefix match with '*' wildcard support."""
        if pattern == "/":
            return True
        # Simple wildcard support: split on '*'
        if "*" in pattern:
            import re

            regex = "^" + re.escape(pattern).replace(r"\*", ".*") + ".*"
            return re.match(regex, path) is not None
        return path == pattern.rstrip("/") or path.startswith(pattern.rstrip("/") + "/")
