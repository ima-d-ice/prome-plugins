"""The labeled evaluation corpus of the hybrid retrieval engine.

The corpus deliberately mixes the two kinds of queries a production system
faces:

* **exact technical queries** — error codes, API names, type signatures,
  environment variables.  These are the bread and butter of a lexical
  channel: a dense model tends to scatter them across semantically nearby
  passages, while BM25 lands on the exact term.
* **natural-language questions** — the cases where lexical matching
  underperforms and the dense channel earns its keep.

The ground truth is expressed on the *text* of the relevant passage, so the
labels stay valid no matter how the corpus is chunked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    source: str
    title: str
    sections: tuple[tuple[str, tuple[str, ...]], ...] = field(default_factory=tuple)


# ----------------------------------------------------------------------
# The corpus.
# ----------------------------------------------------------------------
_DOCS: tuple[Document, ...] = (
    Document(
        source="platform-runbook.md",
        title="Platform Runbook",
        sections=(
            ("Service Restarts", (
                "The platform gracefully restarts every service at 02:00 UTC "
                "for the nightly maintenance window. A service that fails to "
                "drain its in-flight requests is terminated after ten minutes.",
                "The grace period is tracked by the lifecycle controller. "
                "SIGINT is sent first; if the process does not exit, SIGTERM "
                "follows thirty seconds later.",
                "Operators can trigger a restart of a single instance with "
                "the command: service control --restart --instance <id>.",
            )),
            ("Key Rotation", (
                "Signing keys are rotated every thirty days by the key "
                "rotation scheduler, which runs at 02:00 UTC. A key that is "
                "not rotated within the grace period of seven days is revoked "
                "automatically.",
                "The rotation writes the new key version to the key store and "
                "publishes the change to the discovery endpoint, so that the "
                "collectors pick it up without downtime.",
                "Operators can verify the current key version with: "
                "keyctl current --service auth.",
            )),
            ("Cache Eviction Policy", (
                "The cache uses an allkeys-lru eviction policy. When the cache "
                "is full, the least recently used key is evicted to make room "
                "for the incoming key.",
                "The TTL of a key is set from the X-Cache-TTL header. A key "
                "without a TTL never expires and only falls to eviction under "
                "memory pressure.",
                "The cache treats the header X-Cache-Bypass=true as a request "
                "to read the origin directly, skipping the cache lookup.",
            )),
            ("Error Codes", (
                "The API returns error code 4291 when the caller exceeds the "
                "token bucket limit. Clients must wait the number of seconds "
                "in the Retry-After header before retrying.",
                "Error code 5030 is returned when the upstream shard is in a "
                "resharding window, which lasts at most ninety seconds.",
                "Error code 4120 indicates the request header "
                "X-Request-Signature did not verify against the rotating key.",
                "The error payload follows the shape {\"code\": 4291, "
                "\"message\": \"...\", \"retry_after\": 12}.",
            )),
            ("Metrics And Alerts", (
                "The platform exports a Prometheus metric called "
                "platform_cache_hit_ratio_total, which is scraped every "
                "fifteen seconds.",
                "Alerting rules page the on-call engineer when the cache hit "
                "ratio drops below 0.75 for ten consecutive minutes.",
                "The metric platform_key_rotation_current_age_seconds reports "
                "the age of the current signing key.",
            )),
        ),
    ),
    Document(
        source="api-reference.md",
        title="API Reference",
        sections=(
            ("Authentication Endpoint", (
                "POST /v2/auth/sessions creates a session and returns a "
                "session token in the response body. The endpoint accepts the "
                "client credentials grant.",
                "The request header X-Client-Id must carry the public client "
                "identifier issued by the developer console.",
                "A session token expires after 3600 seconds and must be "
                "refreshed with the refresh token endpoint.",
            )),
            ("Rate Limits", (
                "The rate limiter is a token bucket with a capacity of 100 "
                "tokens and a refill rate of 10 tokens per second. The limit "
                "applies per authenticated client.",
                "When the bucket is empty the API returns HTTP 429 with the "
                "Retry-After header set to the number of seconds until the "
                "next token becomes available.",
                "The rate limit headers are X-RateLimit-Remaining, "
                "X-RateLimit-Reset and X-RateLimit-Limit.",
            )),
            ("Pagination", (
                "List endpoints accept the page_size query parameter, which "
                "defaults to 20 and is capped at 100. The cursor for the next "
                "page is returned in the X-Next-Cursor header.",
                "Clients must treat the cursor as opaque and pass it back "
                "unchanged; the format is not stable between releases.",
            )),
            ("Webhooks", (
                "Webhooks are delivered with a signed payload. The signature "
                "is in the X-Webhook-Signature header and is computed as an "
                "HMAC-SHA256 of the raw request body with the endpoint secret.",
                "A delivery is retried up to five times with exponential "
                "backoff. After five failures the endpoint is paused.",
            )),
        ),
    ),
)

# ----------------------------------------------------------------------
# The queries and their ground truth.
# ----------------------------------------------------------------------
_QUERIES: tuple[dict[str, Any], ...] = (
    # --- exact technical queries (BM25 favours them) ---
    {
        "id": "q01",
        "text": "4291",
        "evidence": "The API returns error code 4291 when the caller exceeds the token bucket limit.",
        "notes": "exact error code",
    },
    {
        "id": "q02",
        "text": "X-RateLimit-Remaining",
        "evidence": "The rate limit headers are X-RateLimit-Remaining, X-RateLimit-Reset and X-RateLimit-Limit.",
        "notes": "exact header name",
    },
    {
        "id": "q03",
        "text": "platform_key_rotation_current_age_seconds",
        "evidence": "The metric platform_key_rotation_current_age_seconds reports the age of the current signing key.",
        "notes": "exact metric name",
    },
    {
        "id": "q04",
        "text": "service control --restart --instance",
        "evidence": "Operators can trigger a restart of a single instance with the command: service control --restart --instance <id>.",
        "notes": "exact command",
    },
    {
        "id": "q05",
        "text": "HMAC-SHA256",
        "evidence": "The signature is in the X-Webhook-Signature header and is computed as an HMAC-SHA256 of the raw request body with the endpoint secret.",
        "notes": "exact algorithm name",
    },
    {
        "id": "q06",
        "text": "4291 message retry_after",
        "evidence": "The error payload follows the shape {\"code\": 4291, \"message\": \"...\", \"retry_after\": 12}.",
        "notes": "json keys of the error payload",
    },
    {
        "id": "q07",
        "text": "--restart --instance",
        "evidence": "Operators can trigger a restart of a single instance with the command: service control --restart --instance <id>.",
        "notes": "flag pair",
    },
    {
        "id": "q08",
        "text": "service control --restart",
        "evidence": "Operators can trigger a restart of a single instance with the command: service control --restart --instance <id>.",
        "notes": "partial command",
    },
    # --- natural-language questions (dense favours them) ---
    {
        "id": "q09",
        "text": "How do I prevent the cache from growing forever?",
        "evidence": "The cache uses an allkeys-lru eviction policy. When the cache is full, the least recently used key is evicted to make room for the incoming key.",
        "notes": "paraphrase of eviction",
    },
    {
        "id": "q10",
        "text": "Why do my requests start failing once the rate limit is exceeded?",
        "evidence": "When the bucket is empty the API returns HTTP 429 with the Retry-After header set to the number of seconds until the next token becomes available.",
        "notes": "paraphrase of 429 + Retry-After",
    },
    {
        "id": "q11",
        "text": "What happens to a session token after it expires?",
        "evidence": "A session token expires after 3600 seconds and must be refreshed with the refresh token endpoint.",
        "notes": "paraphrase of token expiry",
    },
    {
        "id": "q12",
        "text": "How are webhooks retried when the delivery fails?",
        "evidence": "A delivery is retried up to five times with exponential backoff. After five failures the endpoint is paused.",
        "notes": "paraphrase of retry",
    },
    {
        "id": "q13",
        "text": "Can a session be refreshed before it expires?",
        "evidence": "A session token expires after 3600 seconds and must be refreshed with the refresh token endpoint.",
        "notes": "paraphrase of refresh",
    },
    {
        "id": "q14",
        "text": "Why was the whole endpoint paused for a customer?",
        "evidence": "A delivery is retried up to five times with exponential backoff. After five failures the endpoint is paused.",
        "notes": "paraphrase of pause after retries",
    },
    # --- mixed ---
    {
        "id": "q15",
        "text": "What does error code 4120 mean and how do I fix it?",
        "evidence": "Error code 4120 indicates the request header X-Request-Signature did not verify against the rotating key.",
        "notes": "error code + paraphrase",
    },
    {
        "id": "q16",
        "text": "Which header tells me how many requests remain before I get rate limited?",
        "evidence": "The rate limit headers are X-RateLimit-Remaining, X-RateLimit-Reset and X-RateLimit-Limit.",
        "notes": "paraphrase asking for a header name",
    },
    {
        "id": "q17",
        "text": "How do I paginate through list endpoints?",
        "evidence": "Clients must treat the cursor as opaque and pass it back unchanged; the format is not stable between releases.",
        "notes": "paraphrase asking for the cursor mechanism",
    },
    {
        "id": "q18",
        "text": "Why is the cache sometimes skipping the cache lookup?",
        "evidence": "The cache treats the header X-Cache-Bypass=true as a request to read the origin directly, skipping the cache lookup.",
        "notes": "paraphrase of bypass header",
    },
)

CORPUS: tuple[Document, ...] = _DOCS
QUERIES: tuple[dict[str, Any], ...] = _QUERIES
