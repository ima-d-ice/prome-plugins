import pytest

pytest.importorskip("docling", reason="docling not installed")

from parsers.contracts import BlockType
from parsers.pdf.docling import DoclingParser


@pytest.fixture(scope="module")
def parser():
    return DoclingParser()


def test_headings_detected(parser, fixtures_dir):
    doc = parser.parse(fixtures_dir / "multipage_structured.pdf")
    headings = [
        b.content for page in doc.pages for b in page.blocks if b.block_type == BlockType.HEADING
    ]
    assert any("Introduction" in h for h in headings)
    assert any("Methods" in h for h in headings)
    assert any("Results" in h for h in headings)


def test_table_detected(parser, fixtures_dir):
    doc = parser.parse(fixtures_dir / "tables.pdf")
    tables = [
        b for page in doc.pages for b in page.blocks if b.block_type == BlockType.TABLE
    ]
    assert tables, "expected a table block"
    assert "Widget" in tables[0].content


def test_bbox_normalized_top_left(parser, fixtures_dir):
    doc = parser.parse(fixtures_dir / "simple_digital.pdf")
    for page in doc.pages:
        for block in page.blocks:
            if block.bbox is None:
                continue
            assert block.bbox.y1 >= block.bbox.y0
            assert block.bbox.y0 >= 0


def test_page_dimensions(parser, fixtures_dir):
    doc = parser.parse(fixtures_dir / "simple_digital.pdf")
    assert doc.pages[0].width == pytest.approx(612, abs=1)
    assert doc.pages[0].height == pytest.approx(792, abs=1)


def test_converter_reused_across_parses(fixtures_dir):
    parser = DoclingParser()
    assert parser._converter is None
    parser.parse(fixtures_dir / "simple_digital.pdf")
    first = parser._converter
    assert first is not None
    assert parser.converter_init_count == 1
    assert parser.converter_init_seconds is not None
    parser.parse(fixtures_dir / "tables.pdf")
    assert parser._converter is first, "converter must be reused, not rebuilt per parse"
    assert parser.converter_init_count == 1


def test_parser_instances_do_not_share_converters(fixtures_dir):
    first = DoclingParser()
    second = DoclingParser()
    first.parse(fixtures_dir / "simple_digital.pdf")
    second.parse(fixtures_dir / "simple_digital.pdf")
    assert first._converter is not None
    assert second._converter is not None
    assert first._converter is not second._converter


def test_reuse_does_not_change_parsing_results(fixtures_dir):
    parser = DoclingParser()
    cold = parser.parse(fixtures_dir / "tables.pdf")
    warm = parser.parse(fixtures_dir / "tables.pdf")
    assert cold.text == warm.text
    assert [
        (b.block_type, b.content) for page in cold.pages for b in page.blocks
    ] == [
        (b.block_type, b.content) for page in warm.pages for b in page.blocks
    ]
    fresh = DoclingParser().parse(fixtures_dir / "tables.pdf")
    assert cold.text == fresh.text


def test_converter_init_failure_raises_and_recovers(fixtures_dir, monkeypatch):
    import docling.document_converter as docling_document_converter

    class Boom:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("simulated converter init failure")

    parser = DoclingParser()
    monkeypatch.setattr(docling_document_converter, "DocumentConverter", Boom)
    from parsers.contracts import ParserError

    with pytest.raises(ParserError, match="converter initialization failed"):
        parser.parse(fixtures_dir / "simple_digital.pdf")
    monkeypatch.undo()
    doc = parser.parse(fixtures_dir / "simple_digital.pdf")  # retried, not sticky-broken
    assert "Quarterly Report Overview" in doc.text
