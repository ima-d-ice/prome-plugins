import pytest

from parsers.contracts import BlockType, ParserDependencyError

from tests.parsers.html.conftest import make_parser, parser_importable

beautifulsoup_available = parser_importable("beautifulsoup")
lxml_available = False
try:
    import lxml  # noqa: F401

    lxml_available = True
except ImportError:
    pass


@pytest.mark.skipif(not beautifulsoup_available, reason="beautifulsoup4 not installed")
class TestBeautifulSoupParsing:
    def test_heading_levels_preserved(self, fixtures_dir):
        parser = make_parser("beautifulsoup")
        doc = parser.parse(fixtures_dir / "documentation_headings.html")
        headings = [(b.content, b.metadata.get("heading_level")) for b in doc.blocks if b.block_type is BlockType.HEADING]
        levels = dict(headings)
        assert levels["Core Concepts"] == 1
        assert levels["The Storage Layer"] == 3
        assert levels["Inline Elements"] == 4
        assert len(headings) == 10

    def test_nested_list_structure_and_order(self, fixtures_dir):
        parser = make_parser("beautifulsoup")
        doc = parser.parse(fixtures_dir / "nested_lists.html")
        lists = [b for b in doc.blocks if b.block_type is BlockType.LIST]
        assert len(lists) == 2
        unordered = lists[0]
        assert unordered.metadata["list_type"] == "ul"
        assert unordered.metadata["items"][0] == "Flour"
        assert "Dark brown" in unordered.metadata["items"]
        assert unordered.metadata["depth"] >= 2
        assert "- Dark brown" in unordered.content or "Dark brown" in unordered.content

        ordered = lists[1]
        assert ordered.metadata["ordered"] is True
        assert ordered.metadata["list_type"] == "ol"
        assert ordered.content.splitlines()[0].startswith("1. ")

    def test_table_rows_headers_and_cells_preserved(self, fixtures_dir):
        parser = make_parser("beautifulsoup")
        doc = parser.parse(fixtures_dir / "complex_table.html")
        tables = [b for b in doc.blocks if b.block_type is BlockType.TABLE]
        assert len(tables) == 3
        first = tables[0]
        assert first.metadata["headers"] == ["Item", "Quantity", "Unit Price", "Total"]
        assert first.metadata["has_header"] is True
        assert first.metadata["rows"] == 5  # header + 4 data rows
        assert ["Widget Pro", "12", "19.99", "239.88"] in first.metadata["row_cells"]
        # colspan expanded in the combined-header table
        combined = tables[2]
        assert combined.metadata["headers"] == ["Combined Segment", "Combined Segment"]
        assert combined.metadata["row_cells"][1] == ["Alpha", "Beta"]

    def test_links_preserved_with_href(self, fixtures_dir):
        parser = make_parser("beautifulsoup")
        doc = parser.parse(fixtures_dir / "links_images.html")
        links = [b for b in doc.blocks if b.block_type is BlockType.LINK]
        hrefs = [b.metadata["href"] for b in links]
        assert "https://reference.example.com/overview" in hrefs
        assert "https://tracker.example.net/issues" in hrefs
        assert len(links) == 7
        # link text is preserved
        texts = [b.content for b in links]
        assert "overview guide" in texts

    def test_images_preserved_with_alt(self, fixtures_dir):
        parser = make_parser("beautifulsoup")
        doc = parser.parse(fixtures_dir / "links_images.html")
        images = [b for b in doc.blocks if b.block_type is BlockType.IMAGE]
        assert len(images) == 2
        alts = {b.metadata["alt"] for b in images}
        assert alts == {"Pipeline architecture diagram", "Campus office atrium with plants"}
        srcs = {b.metadata["src"] for b in images}
        assert "images/pipeline-diagram.png" in srcs

    def test_code_block_language(self, fixtures_dir):
        parser = make_parser("beautifulsoup")
        doc = parser.parse(fixtures_dir / "scripts_styles.html")
        code = [b for b in doc.blocks if b.block_type is BlockType.CODE]
        assert len(code) == 1
        assert code[0].metadata.get("language") == "python"
        assert 'return " ".join(text.split())' in code[0].content

    def test_scripts_and_styles_excluded(self, fixtures_dir):
        parser = make_parser("beautifulsoup")
        doc = parser.parse(fixtures_dir / "scripts_styles.html")
        assert "SECRET_ANALYTICS_TOKEN" not in doc.text
        assert "tok_should_never_appear_in_output" not in doc.text
        assert ".inline-style-block" not in doc.text
        assert "@media print" not in doc.text

    def test_boilerplate_retained_by_default(self, fixtures_dir):
        parser = make_parser("beautifulsoup")
        doc = parser.parse(fixtures_dir / "navigation_layout.html")
        assert "MUSTER ANALYTICS" in doc.text
        assert "Copyright 2026 Muster Analytics" in doc.text
        assert "SPONSORED: Upgrade your plan today" in doc.text

    def test_boilerplate_can_be_excluded(self, fixtures_dir):
        parser = make_parser("beautifulsoup", include_boilerplate=False)
        doc = parser.parse(fixtures_dir / "navigation_layout.html")
        assert "How Navigation Layouts Work" in doc.text
        assert "landmark roles" in doc.text
        assert "MUSTER ANALYTICS" not in doc.text
        assert "Copyright 2026 Muster Analytics" not in doc.text
        assert "Download the whitepaper" not in doc.text

    def test_metadata_extraction(self, fixtures_dir):
        parser = make_parser("beautifulsoup")
        doc = parser.parse(fixtures_dir / "metadata_og_jsonld.html")
        metadata = doc.metadata
        assert metadata["title"] == "Metadata Showcase Article"
        assert "Every metadata flavor" in metadata["description"]
        assert metadata["author"] == "Jordan Example"
        assert metadata["canonical"] == "https://example.com/metadata-showcase"
        assert metadata["source_url"] == "https://example.com/metadata-showcase"
        assert metadata["og:title"] == "Metadata Showcase Article"
        assert metadata["og:image"] == "https://example.com/og.png"
        assert metadata["twitter:card"] == "summary_large_image"
        json_ld = metadata["json_ld"]
        assert isinstance(json_ld, list) and json_ld[0]["@type"] == "Article"
        assert json_ld[0]["author"]["name"] == "Jordan Example"

    def test_source_url_overrides_canonical(self, fixtures_dir):
        parser = make_parser("beautifulsoup")
        doc = parser.parse(fixtures_dir / "metadata_og_jsonld.html", source_url="https://override.example.com/x")
        assert doc.metadata["source_url"] == "https://override.example.com/x"

    def test_dom_ordering_is_document_order(self, fixtures_dir):
        parser = make_parser("beautifulsoup")
        doc = parser.parse(fixtures_dir / "links_images.html")
        orders = [b.metadata["dom_order"] for b in doc.blocks]
        assert orders == sorted(orders)
        tags = [b.metadata["tag"] for b in doc.blocks]
        assert tags[0] == "h1"
        assert "img" in tags

    def test_malformed_html_parses(self, fixtures_dir):
        parser = make_parser("beautifulsoup")
        doc = parser.parse(fixtures_dir / "malformed.html")
        assert "This paragraph is never closed" in doc.text

    @pytest.mark.skipif(not lxml_available, reason="lxml not installed")
    def test_lxml_backend_handles_malformed_html_better(self, fixtures_dir):
        parser = make_parser("beautifulsoup", backend="lxml")
        doc = parser.parse(fixtures_dir / "malformed.html")
        types = [b.block_type for b in doc.blocks]
        assert BlockType.TABLE in types
        assert BlockType.LIST in types
        heading = next(b for b in doc.blocks if b.block_type is BlockType.HEADING)
        assert heading.content == "Malformed Page Example"

    def test_unsupported_backend_raises(self):
        parser = make_parser("beautifulsoup", backend="not-a-backend")
        with pytest.raises(ValueError):
            parser.parse("<html><body><p>x</p></body></html>")

    @pytest.mark.skipif(lxml_available, reason="lxml installed")
    def test_missing_backend_dependency(self, monkeypatch, fixtures_dir):
        from tests.parsers.html.conftest import parser_importable

        monkeypatch.setitem(__import__("sys").modules, "lxml", None)
        parser = make_parser("beautifulsoup", backend="lxml")
        with pytest.raises(ParserDependencyError):
            parser.parse(fixtures_dir / "simple_article.html")

    def test_missing_beautifulsoup_dependency(self, monkeypatch, fixtures_dir):
        import sys

        monkeypatch.setitem(sys.modules, "bs4", None)
        parser = make_parser("beautifulsoup")
        with pytest.raises(ParserDependencyError) as excinfo:
            parser.parse(fixtures_dir / "simple_article.html")
        assert "beautifulsoup4" in str(excinfo.value)
        assert "pip install" in str(excinfo.value)

    def test_parser_stays_valid_protocol(self, fixtures_dir):
        from parsers.contracts import Parser

        parser = make_parser("beautifulsoup")
        assert isinstance(parser, Parser)
