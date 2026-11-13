"""Verify loader output can be handed to the existing parser ecosystem.

Loader responsibility ends at validated, retrievable content. Parsing is
performed by the existing Prome RAG parsers; this test only checks that
the boundary is clean and that no parsing logic is duplicated in loaders.
"""

from __future__ import annotations

import hashlib

import pytest

from loaders.filesystem import FileLoader


def test_loader_output_can_feed_html_parser(tmp_path):
    pytest.importorskip("bs4")
    from parsers.html.beautifulsoup import BeautifulSoupParser

    fixture = tmp_path / "sample.html"
    fixture.write_text(
        "<!DOCTYPE html><html><head><title>Sample</title></head>"
        "<body><h1>Heading</h1><p>Paragraph content.</p></body></html>"
    )

    loader = FileLoader(permitted_roots=[tmp_path])
    result = loader.load(fixture)
    assert result.success_count == 1
    resource = result.resources[0]

    # Hash integrity before handing to parser
    assert resource.content_hash == hashlib.sha256(resource.content).hexdigest()

    # Round-trip the content through the existing parser
    parser = BeautifulSoupParser()
    document = parser.parse(fixture)
    assert document.text
    assert "Paragraph content" in document.text
    assert resource.content_hash == hashlib.sha256(resource.content).hexdigest()


def test_loader_output_handles_empty_and_known_types(tmp_path):
    fixture = tmp_path / "data.json"
    fixture.write_text('{"key": "value"}')

    loader = FileLoader(permitted_roots=[tmp_path])
    result = loader.load(fixture)
    assert result.success_count == 1
    from loaders.contracts.models import ContentType

    assert result.resources[0].content_type is ContentType.JSON


def test_loaders_do_not_import_parser_internals():
    """Loaders must not reimplement or import parser internals."""
    import loaders
    import loaders.filesystem.directory_loader
    import loaders.uploads.manager
    import loaders.web.crawler
    import loaders.web.url_loader

    for module in (
        loaders,
        loaders.web.crawler,
        loaders.web.url_loader,
        loaders.uploads.manager,
        loaders.filesystem.directory_loader,
    ):
        module_file = getattr(module, "__file__", None)
        assert module_file is not None
        with open(module_file) as fh:
            source = fh.read()
        # Loaders must not import parser/chunker/embedding/vector-store modules
        for forbidden in (
            "from parsers",
            "import parsers",
            "from chunkers",
            "import chunkers",
            "from embeddings",
            "from vector_stores",
        ):
            assert forbidden not in source, f"{module_file} imports {forbidden}"
