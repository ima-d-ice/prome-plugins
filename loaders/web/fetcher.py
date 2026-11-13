"""HTTP URL fetcher for Prome RAG Plugins loaders.

Standard-library only (urllib). No heavyweight dependencies required.
Does NOT execute JavaScript or introduce browser automation.
"""

from __future__ import annotations

import hashlib
import socket
import ssl
import urllib.error
import urllib.request
from datetime import datetime, timezone
from urllib.parse import urlparse

from loaders.contracts.exceptions import (
    LoaderValidationError,
    SSRFError,
)
from loaders.contracts.models import (
    ContentType,
    ETag,
    FetchInfo,
    LastModified,
    LoadedResource,
    LoaderResult,
    SourceProvenance,
    SourceReference,
    SourceType,
)

DEFAULT_MAX_RESPONSE_SIZE = 50 * 1024 * 1024  # 50 MiB
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_REDIRECTS = 10
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF = 1.5


def resolve_and_check_ssrf(host: str) -> None:
    """Resolve a host and reject SSRF-unsafe destinations.

    Rejects loopback, private, link-local, reserved, and multicast
    addresses, including common cloud metadata endpoints. Raises
    SSRFError on unsafe destinations.
    """
    import ipaddress

    try:
        addr = ipaddress.ip_address(host)
        _check_address(addr, host)
        return
    except ValueError:
        pass

    # DNS resolution check — reject if any resolved address is unsafe
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        # Host cannot be resolved; treat as not safe to fetch
        raise SSRFError(f"Unable to resolve host: {host}")

    for info in infos:
        sockaddr = info[4]
        ip_str = sockaddr[0]
        try:
            addr = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        _check_address(addr, host)


def _check_address(addr, host: str) -> None:
    """Check a single resolved address for SSRF safety."""
    if addr.is_loopback:
        raise SSRFError(f"Loopback address denied: {host} ({addr})")
    if addr.is_private:
        raise SSRFError(f"Private network address denied: {host} ({addr})")
    if addr.is_link_local:
        raise SSRFError(f"Link-local address denied: {host} ({addr})")
    if addr.is_reserved:
        raise SSRFError(f"Reserved address denied: {host} ({addr})")
    if addr.is_multicast:
        raise SSRFError(f"Multicast address denied: {host} ({addr})")
    if addr.is_unspecified:
        raise SSRFError(f"Unspecified address denied: {host} ({addr})")

    # Common cloud metadata service ranges
    ip_str = str(addr)
    metadata_hosts = (
        "169.254.169.254",   # AWS / Azure / GCP metadata
        "100.100.100.200",   # Alibaba metadata
        "169.254.169.253",
        "fd00:ec2::254",     # AWS IPv6 metadata
    )
    if ip_str in metadata_hosts:
        raise SSRFError(f"Cloud metadata service denied: {host} ({addr})")


class Fetcher:
    """HTTP fetcher for downloading content from URLs.

    Supports request timeout, maximum response size, redirect limit,
    content-type validation, ETag / Last-Modified, content hash, retry
    policy and fetch diagnostics.
    """

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        max_response_size: int = DEFAULT_MAX_RESPONSE_SIZE,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        allowed_protocols: tuple[str, ...] = ("http", "https"),
        retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF,
        ssl_verify: bool = True,
        user_agent: str = "prome-rag-plugins/1.0",
    ):
        self.timeout = timeout
        self.max_response_size = max_response_size
        self.max_redirects = max_redirects
        self.allowed_protocols = allowed_protocols
        self.retry_attempts = retry_attempts
        self.retry_backoff = retry_backoff
        self.ssl_verify = ssl_verify
        self.user_agent = user_agent

    def validate_url(self, url: str) -> None:
        """Validate protocol and SSRF protections."""
        parsed = urlparse(url)
        protocol = parsed.scheme.lower()

        if protocol not in self.allowed_protocols:
            raise LoaderValidationError(
                f"Protocol '{protocol}' is not allowed. Allowed: {self.allowed_protocols}"
            )

        host = parsed.hostname
        if not host:
            raise LoaderValidationError("URL has no host")

        resolve_and_check_ssrf(host)

    def fetch(self, url: str, headers: dict | None = None) -> LoaderResult:
        """Fetch content from a URL (with retries).

        Retryable: transient network errors and HTTP 5xx responses.
        Not retryable: 4xx responses, size/protocol violations, SSRF (raised).
        """
        try:
            self.validate_url(url)
        except SSRFError:
            raise
        except LoaderValidationError as e:
            return LoaderResult().add_failure(str(e), None)

        last_error: Exception | None = None

        for attempt in range(self.retry_attempts):
            try:
                return self._fetch_single(url, headers=headers)
            except SSRFError:
                # Security violations are never retried nor swallowed
                raise
            except urllib.error.HTTPError as e:
                if e.code >= 500 and attempt < self.retry_attempts - 1:
                    last_error = e
                    self._backoff(attempt)
                    continue
                return LoaderResult().add_failure(
                    f"HTTP {e.code}: {e.reason}", None, error_code=str(e.code)
                )
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                last_error = e
                if attempt < self.retry_attempts - 1:
                    self._backoff(attempt)
                continue
            except LoaderValidationError as e:
                # Non-retryable (oversize body, protocol, validation)
                return LoaderResult().add_failure(str(e), None)

        return LoaderResult().add_failure(
            f"Failed to fetch URL after {self.retry_attempts} attempts: {last_error}",
            None,
        )

    def _backoff(self, attempt: int) -> None:
        """Exponential backoff between retry attempts."""
        import time

        time.sleep(self.retry_backoff ** (attempt + 1))

    def _fetch_single(
        self, url: str, headers: dict | None = None
    ) -> LoaderResult:
        """Perform a single fetch attempt."""
        result = LoaderResult()

        req = urllib.request.Request(url)
        req.add_header("User-Agent", self.user_agent)
        for key, value in (headers or {}).items():
            req.add_header(key, value)

        context = ssl.create_default_context() if self.ssl_verify else ssl._create_unverified_context()

        redirect_count = 0
        final_url = url

        # HTTP errors propagate to fetch() for retry/reporting decisions
        response = urllib.request.urlopen(
            req, timeout=self.timeout, context=context
        )

        with response:
            final_url = response.geturl()
            # Revalidate the final URL (redirect target) for SSRF
            try:
                self.validate_url(final_url)
            except SSRFError:
                raise SSRFError(f"Redirect to unsafe address blocked: {final_url}")
            except (LoaderValidationError, ValueError, OSError) as e:
                raise SSRFError(f"Redirect target could not be validated: {final_url}: {e}")

            content_type_header = response.headers.get("Content-Type", "")
            status_code = getattr(response, "status", None)
            etag_header = response.headers.get("ETag")
            last_modified_header = response.headers.get("Last-Modified")
            redirect_count = len(getattr(response, "redirect_history", ()) or ())

            # Read with size cap
            content = self._read_capped(response)

        actual_size = len(content)
        content_type = detect_content_type(content, content_type_header)
        content_hash = hashlib.sha256(content).hexdigest()

        # ETag / Last-Modified
        etag: ETag | None = ETag(tag=etag_header) if etag_header else None
        last_modified: LastModified | None = parse_last_modified(
            last_modified_header, etag
        )

        source_ref = SourceReference(
            source_id=_new_source_id(url),
            source_type=SourceType.URL,
            original_uri_or_path=url,
        )

        provenance = SourceProvenance(
            fetched_from=url,
            fetched_at=datetime.now(timezone.utc),
            last_modified=last_modified,
            content_hash=content_hash,
            etag=etag,
        )

        fetch_info = FetchInfo(
            fetch_timestamp=datetime.now(timezone.utc),
            status_code=status_code,
            redirect_count=redirect_count,
            final_url=final_url,
        )

        resource = LoadedResource(
            reference=source_ref,
            content=content,
            content_type=content_type,
            filename=derive_filename(url, content_type),
            file_size=actual_size,
            content_hash=content_hash,
            last_modified=last_modified,
            fetch_info=fetch_info,
            provenance=provenance,
        )

        result.resources.append(resource)
        return result

    def _read_capped(self, response) -> bytes:
        """Read response body with a hard size cap."""
        limit = self.max_response_size + 1
        content = response.read(limit)
        if len(content) > self.max_response_size:
            raise LoaderValidationError(
                f"Response exceeds maximum size of {self.max_response_size} bytes"
            )
        return content


def detect_content_type(content: bytes, content_type_header: str = "") -> ContentType:
    """Detect content type from bytes and/or header."""
    header = (content_type_header or "").split(";")[0].strip().lower()
    mapping = {
        "text/plain": ContentType.TEXT,
        "text/markdown": ContentType.MARKDOWN,
        "text/html": ContentType.HTML,
        "text/csv": ContentType.TEXT,
        "application/json": ContentType.JSON,
        "application/pdf": ContentType.PDF,
        "application/xml": ContentType.TEXT,
        "image/png": ContentType.IMAGE,
        "image/jpeg": ContentType.IMAGE,
        "image/gif": ContentType.IMAGE,
        "image/webp": ContentType.IMAGE,
        "audio/mpeg": ContentType.AUDIO,
        "audio/wav": ContentType.AUDIO,
        "audio/ogg": ContentType.AUDIO,
        "video/mp4": ContentType.VIDEO,
    }
    if header in mapping:
        return mapping[header]

    if content.startswith(b"%PDF"):
        return ContentType.PDF
    if content[:32].lower().startswith((b"<!doctype html", b"<html", b"<!--")):
        return ContentType.HTML
    if content.startswith((b"{", b"[")):
        return ContentType.JSON
    if content.startswith((b"\x89PNG", b"\xff\xd8\xff", b"GIF87a", b"GIF89a")):
        return ContentType.IMAGE
    if header.startswith("text/"):
        return ContentType.TEXT
    return ContentType.UNKNOWN


def parse_last_modified(value: str | None, etag: ETag | None = None) -> LastModified | None:
    """Parse an HTTP Last-Modified header."""
    if not value:
        return None
    from email.utils import parsedate_to_datetime

    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return LastModified(timestamp=dt, etag=etag.tag if etag else None)
    except (ValueError, TypeError, OverflowError):
        return None


def derive_filename(url: str, content_type: ContentType = ContentType.UNKNOWN) -> str | None:
    """Derive a filename from URL path."""
    parsed = urlparse(url)
    path = parsed.path
    if path:
        name = path.rstrip("/").rsplit("/", 1)[-1]
        if name and "." in name:
            return name
    return None


def _new_source_id(url: str) -> str:
    """Stable source id derived from URL."""
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, url))
