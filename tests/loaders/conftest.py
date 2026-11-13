"""Shared fixtures for loader tests."""

from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

PAGES = {
    "/": "<html><head><title>Home</title></head><body><a href='/about.html'>About</a><a href='/secret.html'>Secret</a></body></html>",
    "/about.html": "<html><head><title>About Us</title></head><body><h1>About</h1><a href='/'>Home</a></body></html>",
    "/secret.html": "<html><head><title>Secret</title></head><body>secret</body></html>",
    "/robots.txt": "User-agent: *\nDisallow: /secret.html\nAllow: /\n",
    "/doc.pdf": "%PDF-1.4\n%%fake-pdf-body\n",
}


@pytest.fixture()
def http_server():
    """Run a controlled local HTTP server for loader tests."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = PAGES.get(self.path)
            if body is None:
                self.send_error(404)
                return
            content = body.encode("utf-8")
            self.send_response(200)
            ctype = "application/pdf" if self.path.endswith(".pdf") else "text/html"
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("ETag", f'"etag-{abs(hash(self.path))}"')
            self.send_header("Last-Modified", "Fri, 13 Nov 2026 00:00:00 GMT")
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=2)


@pytest.fixture()
def local_files(tmp_path):
    """Create a small fixture tree of local files."""
    root = tmp_path / "source"
    root.mkdir()
    (root / "alpha.txt").write_text("alpha content")
    (root / "beta.md").write_text("# Beta\n\nbeta content")
    sub = root / "nested"
    sub.mkdir()
    (sub / "gamma.txt").write_text("gamma content")
    (root / ".hidden").write_text("hidden content")
    (root / ".env").write_text("SECRET=do-not-load")
    return root


def make_uploaded_file(name, data=b"sample content", file_id=None):
    from loaders.uploads import UploadedFile, UploadedFileStatus

    data = data if isinstance(data, bytes) else data.encode("utf-8")
    return UploadedFile(
        file_id=file_id or f"file-{abs(hash((name, data)))}",
        original_filename=name,
        sanitized_filename="",
        content_type="application/octet-stream",
        file_size=len(data),
        content_hash=hashlib.sha256(data).hexdigest(),
        upload_timestamp=datetime.now(timezone.utc),
        storage_reference="",
        status=UploadedFileStatus.RECEIVED,
        content=data,
    )
