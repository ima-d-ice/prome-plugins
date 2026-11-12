import pytest

from parsers.contracts import BlockType, ParserDependencyError

from tests.parsers.html.conftest import make_parser, parser_importable

trafilatura_available = parser_importable("trafilatura")


@pytest.mark.skipif(not trafilatura_available, reason="trafilatura not installed")
class TestTrafilaturaParsing:
    def test_main_content_extracted_boilerplate_removed(self, fixtures_dir):
        parser = make_parser("trafilatura")
        doc = parser.parse(fixtures_dir / "navigation_layout.html")
        assert "How Navigation Layouts Work" in doc.text
        assert "landmark roles" in doc.text
        assert "MUSTER ANALYTICS" not in doc.text
        assert "SPONSORED: Upgrade your plan today" not in doc.text
        assert "Copyright 2026 Muster Analytics" not in doc.text
        assert "Download the whitepaper" not in doc.text

    def test_heading_levels_preserved(self, fixtures_dir):
        parser = make_parser("trafilatura")
        doc = parser.parse(fixtures_dir / "documentation_headings.html")
        headings = {b.content: b.metadata.get("heading_level") for b in doc.blocks if b.block_type is BlockType.HEADING}
        assert headings["Core Concepts"] == 1
        assert headings["The Storage Layer"] == 3
        assert headings["Inline Elements"] == 4
        assert len(headings) == 10

    def test_nested_list_structure(self, fixtures_dir):
        parser = make_parser("trafilatura")
        doc = parser.parse(fixtures_dir / "nested_lists.html")
        lists = [b for b in doc.blocks if b.block_type is BlockType.LIST]
        assert len(lists) == 2
        unordered = lists[0]
        assert unordered.metadata["list_type"] == "ul"
        assert unordered.metadata["items"][0] == "Flour"
        assert "Dark brown" in unordered.metadata["items"]
        ordered = lists[1]
        assert ordered.metadata["ordered"] is True
        assert ordered.content.splitlines()[0].startswith("1. ")

    def test_tables_preserved(self, fixtures_dir):
        parser = make_parser("trafilatura")
        doc = parser.parse(fixtures_dir / "complex_table.html")
        tables = [b for b in doc.blocks if b.block_type is BlockType.TABLE]
        assert len(tables) == 3
        assert tables[0].metadata["headers"] == ["Item", "Quantity", "Unit Price", "Total"]
        assert tables[0].metadata["has_header"] is True
        assert ["Widget Pro", "12", "19.99", "239.88"] in tables[0].metadata["row_cells"]

    def test_tables_can_be_disabled(self, fixtures_dir):
        parser = make_parser("trafilatura", include_tables=False)
        doc = parser.parse(fixtures_dir / "complex_table.html")
        assert not any(b.block_type is BlockType.TABLE for b in doc.blocks)
        assert "Quarterly Invoice Summary" in doc.text  # prose still extracted

    def test_links_extracted_by_default(self, fixtures_dir):
        parser = make_parser("trafilatura")
        doc = parser.parse(fixtures_dir / "links_images.html")
        links = [b for b in doc.blocks if b.block_type is BlockType.LINK]
        hrefs = [b.metadata["href"] for b in links]
        assert "https://reference.example.com/overview" in hrefs
        assert "https://tracker.example.net/issues" in hrefs
        assert "https://partners.example.com/three" in hrefs
        assert len(links) == 7

    def test_links_can_be_disabled(self, fixtures_dir):
        parser = make_parser("trafilatura", include_links=False)
        doc = parser.parse(fixtures_dir / "links_images.html")
        assert not any(b.block_type is BlockType.LINK for b in doc.blocks)
        assert "overview guide" in doc.text  # link text remains inline

    def test_code_block_preserved_with_language(self, fixtures_dir):
        parser = make_parser("trafilatura")
        doc = parser.parse(fixtures_dir / "scripts_styles.html")
        code = [b for b in doc.blocks if b.block_type is BlockType.CODE]
        assert len(code) == 1
        assert code[0].metadata.get("language") == "python"

    def test_scripts_and_styles_excluded(self, fixtures_dir):
        parser = make_parser("trafilatura")
        doc = parser.parse(fixtures_dir / "scripts_styles.html")
        assert "SECRET_ANALYTICS_TOKEN" not in doc.text
        assert ".inline-style-block" not in doc.text
        assert "tok_should_never_appear_in_output" not in doc.text

    def test_metadata_extraction(self, fixtures_dir):
        parser = make_parser("trafilatura")
        doc = parser.parse(fixtures_dir / "metadata_og_jsonld.html")
        metadata = doc.metadata
        assert metadata["title"] == "Metadata Showcase Article"
        assert metadata["author"] == "Jordan Example"
        assert metadata["description"]
        assert metadata["source_url"] == "https://example.com/metadata-showcase"
        assert metadata["date"] == "2026-09-14"

    def test_images_not_surfaced(self, fixtures_dir):
        # Documented capability gap: Trafilatura does not emit image blocks.
        parser = make_parser("trafilatura")
        doc = parser.parse(fixtures_dir / "links_images.html")
        assert not any(b.block_type is BlockType.IMAGE for b in doc.blocks)

    def test_no_main_content_identifiable(self, fixtures_dir):
        parser = make_parser("trafilatura")
        doc = parser.parse(fixtures_dir / "minimal_no_content.html")
        assert doc.blocks == []
        assert any("no main content" in w.lower() for w in doc.warnings)
        # title-only metadata still surfaced via the metadata fallback
        assert doc.metadata.get("title") == "Under Construction"

    def test_favor_precision_and_recall_are_exclusive(self):
        with pytest.raises(ValueError):
            make_parser("trafilatura", favor_precision=True, favor_recall=True)

    def test_favor_precision_still_works(self, fixtures_dir):
        parser = make_parser("trafilatura", favor_precision=True)
        doc = parser.parse(fixtures_dir / "simple_article.html")
        assert "Getting Started with Widgets" in doc.text

    def test_favor_recall_still_works(self, fixtures_dir):
        parser = make_parser("trafilatura", favor_recall=True)
        doc = parser.parse(fixtures_dir / "simple_article.html")
        assert "Getting Started with Widgets" in doc.text

    def test_target_language_filter(self, fixtures_dir):
        parser = make_parser("trafilatura", target_language="en")
        doc = parser.parse(fixtures_dir / "simple_article.html")
        assert doc.blocks  # English content kept

    def test_missing_trafilatura_dependency(self, monkeypatch, fixtures_dir):
        import sys

        monkeypatch.setitem(sys.modules, "trafilatura", None)
        parser = make_parser("trafilatura")
        with pytest.raises(ParserDependencyError) as excinfo:
            parser.parse(fixtures_dir / "simple_article.html")
        assert "trafilatura" in str(excinfo.value).lower()
        assert "pip install" in str(excinfo.value)

    def test_parser_stays_valid_protocol(self, fixtures_dir):
        from parsers.contracts import Parser

        parser = make_parser("trafilatura")
        assert isinstance(parser, Parser)


class TestPublicationDateNotFabricated:
    def test_free_text_year_does_not_become_date(self, fixtures_dir):
        parser = make_parser("trafilatura")
        doc = parser.parse(fixtures_dir / "long_document.html")
        assert "date" not in doc.metadata

    def test_incidental_year_in_prose_does_not_become_date(self, fixtures_dir):
        parser = make_parser("trafilatura")
        doc = parser.parse(fixtures_dir / "links_images.html")
        assert "date" not in doc.metadata

    def test_copyright_year_does_not_become_date(self, fixtures_dir):
        parser = make_parser("trafilatura")
        doc = parser.parse(fixtures_dir / "navigation_layout.html")
        assert "date" not in doc.metadata

    def test_real_meta_date_is_kept(self, fixtures_dir):
        parser = make_parser("trafilatura")
        doc = parser.parse(fixtures_dir / "metadata_og_jsonld.html")
        assert doc.metadata["date"] == "2026-09-14"

    def test_filedate_not_reported_as_ingestion(self, fixtures_dir):
        parser = make_parser("trafilatura")
        doc = parser.parse(fixtures_dir / "metadata_og_jsonld.html")
        assert "filedate" not in doc.metadata
        assert "ingested_at" not in doc.metadata
