# Prome RAG Plugins — Loaders

A modular **Loader** ecosystem for Prome RAG. Loaders fetch, receive and manage
source content. They stop at providing a **validated, retrievable source
resource** — they do **not** perform parsing, chunking, embedding or vector
indexing.

```text
Uploaded File / Source
        |
        v
      Loader
        |
        v
      Parser
        |
        v
    Chunker
        |
        v
Qwen Embedding
        |
        v
 Indexing Layer
```

- `loaders/contracts` — shared data contracts (`SourceReference`,
  `LoadedResource`, `LoaderResult`) and the `Loader` protocol.
- `loaders/uploads` — the **ChatGPT-style file attachment backend**
  (conversation-scoped and knowledge-base-scoped uploads).
- `loaders/filesystem` — local file and directory loaders.
- `loaders/web` — URL loader and a constrained, SSRF-protected website crawler.
- `loaders/sync` — incremental source synchronization (SQLite-backed).

## Installation

Loaders are importable with the standard library only:

```python
import loaders
```

No HTTP client, database server or queue is required. The URL fetcher and
crawler use `urllib` from the standard library. Parsing is handled by the
existing Prome RAG parsers, which retain their own optional-dependency model.

## Quick start — file uploads

```python
from datetime import datetime, timezone

from loaders.uploads import UploadManager, UploadScope, UploadedFile, UploadedFileStatus

manager = UploadManager(
    storage_root="./data/uploads",       # configured storage root (required)
    session_root="./data/upload-sessions",
    max_file_size=100 * 1024 * 1024,     # 100 MiB default
    max_files_per_batch=20,              # per request default
    max_total_batch_size=500 * 1024 * 1024,
    default_expiry_hours=168,            # 7 days
)

result = manager.upload_files(
    conversation_id="conversation-123",  # scope required for CONVERSATION
    files=[UploadedFile(
        file_id="unique-id",
        original_filename="report.pdf",
        sanitized_filename="",
        content_type="application/pdf",
        file_size=len(data),
        content_hash="",
        upload_timestamp=datetime.now(timezone.utc),
        storage_reference="",
        status=UploadedFileStatus.RECEIVED,
        content=data,                    # bytes or str
    )],
)
```

### Upload scopes

Two explicit scopes are supported:

**`CONVERSATION`** — files belong to a specific conversation and are available
as evidence only within that conversation's permitted scope. They are **never**
promoted to persistent knowledge-base ingestion silently.

**`KNOWLEDGE_BASE`** — files are accepted into a persistent knowledge base and
can be indexed for reuse, subject to access control.

```python
manager.upload_files(
    knowledge_base_id="kb-company-wiki",
    scope=UploadScope.KNOWLEDGE_BASE,
    files=[...],
)
```

Validation requires `conversation_id` for `CONVERSATION` and
`knowledge_base_id` for `KNOWLEDGE_BASE`.

### Partial success and atomicity

Multiple files per request are supported. Each file is validated independently.
By default one invalid file **does not** invalidate unrelated valid files
(partial success). Pass `atomic=True` to select an all-or-nothing policy, in
which case **no** file is stored if any file fails validation.

Per-file errors are reported in `UploadResult.per_file_errors` (filename →
message); rejected files appear in `UploadResult.rejected`. Successful files
appear in `UploadResult.successful`. `UploadResult.duplicate_status` reports
`duplicate` when any file was rejected because its content hash already exists.

### Upload limits (safe defaults)

| Setting | Default | Purpose |
|---|---|---|
| `max_file_size` | 100 MiB | Per-file size cap |
| `max_files_per_batch` | 20 | Files per request |
| `max_total_batch_size` | 500 MiB | Combined batch size |
| `max_conversation_attachments` | 50 | Retention per conversation |
| `default_expiry_hours` | 168 h (7 days) | Default attachment retention |
| `allowed_extensions` | curated allowlist | `.pdf`, `.txt`, `.md`, … |
| `allowed_mime_types` | curated allowlist | MIME allowlist |

All limits are configurable via the constructor. Uploaded files are **never
silently truncated** — oversize files are rejected.

### Validation performed

- Filename sanitization (path separators, null bytes, control characters,
  traversal segments).
- Extension allowlist.
- MIME-type allowlist (client-provided MIME alone is **never** trusted).
- Magic-byte sniffing for binary containers (`PDF`, `PNG`, `JPEG`, `GIF`,
  `WebP`, `ZIP`).
- Content hash (SHA-256) computation and duplicate-content detection.
  Knowledge-base uploads deduplicate globally; conversation uploads
  deduplicate only within the same conversation (hashes never leak across
  conversations).
- Empty-content rejection.
- Batch-size and per-batch file-count limits.

### File storage

A persistent **local file-storage backend** (`StorageBackend`) is provided.
It uses:

- Unique internal IDs (never the uploaded filename) as storage paths.
- Atomic writes (write to `.tmp` then rename).
- SHA-256 content hashing and integrity checks.
- Explicit `storage_root` configuration.
- Safe cleanup via `delete()`.

Uploaded filenames are **never** used as storage paths; path traversal,
directory traversal and symlink escapes are sanitized at the validation layer.
Storage is separate from vector-store storage.

### Attachment lifecycle

Attachments follow an explicit pipeline:

```text
RECEIVED → VALIDATING → STORED → READY_FOR_INGESTION
        → PROCESSING → AVAILABLE → EXPIRED / DELETED
```

An upload succeeding does **not** make the file available for RAG retrieval;
it must pass downstream processing and indexing first. `AVAILABLE` is set
only after downstream processing completes.

Operations equivalent to a chat application's upload button:

```python
manager.upload_files(conversation_id, files)      # upload_files()
manager.list_attachments(conversation_id)          # list_attachments()
manager.get_attachment(conversation_id, file_id)   # get_attachment()
manager.delete_attachment(conversation_id, file_id)# delete_attachment()
manager.delete_conversation_attachments(conversation_id)  # bulk delete
```

Conversation isolation is enforced: `get_attachment`, `delete_attachment` and
`list_attachments` never expose another conversation's files.

`AttachmentLifecycle` supports: listing attachments, fetching metadata,
deleting single attachments, deleting all attachments for a conversation,
expiry configuration (`expiry_time` or `default_expiry_hours`), cleanup of
expired files (`expire_old_attachments`), and marking failed uploads with
preserved diagnostics (`mark_failed`).

Source files are not deleted immediately after parsing; lifecycle and
retention are configurable.

## Local filesystem loaders

```python
from pathlib import Path
from loaders.filesystem import FileLoader, DirectoryLoader

loader = FileLoader(permitted_roots=[Path("./corpus")], max_file_size=20 * 1024 * 1024)
result = loader.load("./corpus/report.pdf")

dloader = DirectoryLoader(
    permitted_roots=[Path("./corpus")],
    recursive=True,
    max_files=500,
    max_total_size=200 * 1024 * 1024,
    hidden_file_policy=HiddenFilePolicy.EXCLUDE,
    symlink_policy=SymlinkPolicy.REJECT,
)
result = dloader.load("./corpus")
```

- **Single-file loader**: safe path resolution against configurable
  `permitted_roots`, content-type detection, size limits, SHA-256 hashing,
  clear failure diagnostics. Paths outside permitted roots raise
  `PathTraversalError`.
- **Directory loader**: recursive and non-recursive discovery, extension
  allowlists, exclusion patterns, hidden-file policy, symlink policy,
  maximum files/size limits, deterministic (sorted) ordering, per-file
  failure reporting, and it never scans outside configured roots.

Excluded by default: `.env` and other secret files, `.git`, virtual
environments, Python caches, build outputs, model caches and sensitive
application configuration. Symlink loops and traversal outside permitted
roots are blocked. Source files are never modified or deleted.

## URL loader

```python
from loaders.web import URLLoader

loader = URLLoader(timeout=30.0, max_response_size=50 * 1024 * 1024)
result = loader.load("https://example.com/report.pdf")
```

Supports request timeout, maximum response size, redirect limit,
content-type detection, `ETag`/`Last-Modified`, SHA-256 content hashing,
retry policy with backoff, and fetch diagnostics (status code, redirect
count, final URL). The URL loader downloads **raw** content; existing
HTML/PDF parsers handle extraction. JavaScript is never executed and no
browser automation is introduced.

## Constrained website crawler

```python
from loaders.web import WebsiteCrawler

crawler = WebsiteCrawler(
    max_depth=2,
    max_pages=50,
    same_domain_only=True,
    polite_delay=1.0,
)
result = crawler.crawl("https://example.com/")
```

Supports a starting URL, same-domain crawling by default (explicit domain
allowlist via `allowed_domains`), maximum depth, maximum pages, maximum
response size, timeout, polite delay, `robots.txt` checking (longest-match
Allow/Disallow), URL normalization, duplicate detection, exclusion rules and
per-page failure reporting.

### SSRF security (mandatory)

All destinations — initial, discovered, and **redirect targets** — are
validated before fetching. Loopback, private, link-local, reserved,
multicast and unspecified addresses are rejected, as are known cloud
metadata endpoints (e.g. `169.254.169.254`) and internal administration
services. DNS results are resolved and validated before any connection, and
unresolvable hosts are rejected. Supported protocols are `http` and
`https` only.

Security tests use controlled local fixtures and do **not** probe real
internal services.

## Incremental source synchronization

```python
from loaders.sync import SyncStatePersister, Reconciler

persister = SyncStatePersister("./data/sync-state.db")   # SQLite
reconciler = Reconciler(persister)

result = reconciler.reconcile(
    source_id="kb-corpus",
    source_type=SourceType.LOCAL_DIRECTORY,
    current_resources=resources,
)
print(result["change_report"])   # new / modified / unchanged / deleted / changed_urls
```

Uses stable source IDs, SHA-256 content hashes, `ETag` and `Last-Modified`
plus previously persisted state. Detects new, modified, unchanged, deleted
and changed-URL sources. State is persisted in SQLite with primary-key
upserts, atomic multi-statement transactions and safe reopening after
interrupted reconciliation (`reconcile()` detects missing state and performs
an initial reconciliation). Explicit deletion events are recorded via
`handle_deletion()` — downstream vector-store/lexical deletion is owned by
the indexing coordinator, **not** by the sync layer.

## Integration boundaries

- **Parsers**: `LoadedResource.content` is handed verbatim to existing
  parsers (`.parse(path)`); loaders never reimplement parsing. The
  `test_parser_compat` suite verifies this boundary.
- **Chunkers / Embeddings / Vector stores**: out of scope by design; the
  loader stops at retrieval.
- **Guardrails**: when an upload-validation guardrail exposes a public
  interface it can be layered ahead of `FileValidator`; until then, the
  loader's own validation is authoritative. No guardrails packages are
  imported by this module.
- **Observability**: diagnostics are exposed via `LoaderResult.warnings`,
  `LoaderResult.failures` and `UploadResult.diagnostics`. Wrap loader calls
  in your existing tracing/metrics instrumentation at the call site.

## Frontend / chat API integration requirements

A future chat frontend should:

1. Use a hidden `<input type="file" multiple>` or an equivalent file picker;
   send selected files as multipart to the app's upload endpoint (or call
   `UploadManager.upload_files` directly if embedding Python).
2. Associate returned `file_id`s with the current conversation ID stored in
   the client session.
3. Show per-file progress; on completion, display validation failures using
   `UploadResult.per_file_errors` keyed by filename.
4. Render attachment status from the attachment list: `received`,
   `validating`, `stored`, `ready_for_ingestion`, `processing`,
   `available`, `failed`, `expired`, `deleted`.
5. Removal maps to `delete_attachment`; bulk cleanup maps to
   `delete_conversation_attachments`.
6. Upload progress should read lifecycle status; do **not** present an
   `available` badge until downstream processing marks the file `available`.

**No HTTP API exists in this repository today**, so no framework endpoint is
added here. Recommended endpoint (once an API exists):

```text
POST   /api/conversations/{conversation_id}/attachments
GET    /api/conversations/{conversation_id}/attachments
GET    /api/conversations/{conversation_id}/attachments/{attachment_id}
DELETE /api/conversations/{conversation_id}/attachments/{attachment_id}
DELETE /api/conversations/{conversation_id}/attachments
```

**No frontend exists in this repository**, so no graphical upload button is
implemented here.

## Unsupported functionality

- No JavaScript execution or browser automation for URL loading/crawling.
- No cloud object storage backend (local persistent backend only).
- No virus/malware scanning (content hash + magic-byte + extension checks only).
- No OCR, audio transcription or video processing (handled by parsers if installed).
- No vector-store indexing (owned downstream by the indexing coordinator).
- No remote upload service (in-process, framework-independent only).
- `robots.txt` is honored with the crawler's configured fetcher; it is not
  independently fetched when no retriever is provided (fail-open, standard
  practice).
- Atomic upload aborts the whole batch when any file fails validation;
  partial success (default) never stores invalid files either — it simply
  stores the valid ones.

## Package layout

```text
loaders/
    __init__.py
    contracts/    models.py protocols.py capabilities.py exceptions.py
    uploads/      manager.py validation.py storage.py sessions.py lifecycle.py models.py
    filesystem/   file_loader.py directory_loader.py filters.py
    web/          fetcher.py url_loader.py crawler.py robots.py
    sync/         state.py change_detector.py reconciliation.py
```

## Testing

```bash
python -m pytest tests/loaders -q
```

108 tests cover contracts, upload validation/storage/lifecycle, conversation
isolation, atomicity, filesystem loaders, the URL loader (local fixture
server), the crawler (depth/domain/robots/SSRF), SQLite sync state, and the
chat upload workflow, including the loader→parser boundary.

## Benchmarks

```bash
python benchmarks/loaders/benchmark.py --repeats 5
```
