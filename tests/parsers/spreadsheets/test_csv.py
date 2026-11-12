import pytest
from parsers.contracts import (
    BlockType,
    CorruptDocumentError,
    EmptyDocumentError,
    ParsedDocument,
)
from parsers.spreadsheets.csv import CsvParser


def test_csv_standard(spreadsheet_fixtures_dir):
    parser = CsvParser()
    doc = parser.parse(spreadsheet_fixtures_dir / "standard.csv")
    assert isinstance(doc, ParsedDocument)
    assert doc.parser_name == "csv_stdlib"
    assert len(doc.blocks) == 1
    b = doc.blocks[0]
    assert b.block_type == BlockType.TABLE
    assert b.metadata["rows"] == 3
    assert b.metadata["cols"] == 5
    assert "Alice Smith" in doc.text


def test_csv_delimiters(spreadsheet_fixtures_dir):
    parser = CsvParser()
    doc_semi = parser.parse(spreadsheet_fixtures_dir / "semicolon.csv")
    assert doc_semi.blocks[0].metadata["delimiter"] == ";"
    assert "Hammer" in doc_semi.text

    doc_pipe = parser.parse(spreadsheet_fixtures_dir / "pipe_delimited.csv")
    assert doc_pipe.blocks[0].metadata["delimiter"] == "|"
    assert "London" in doc_pipe.text


def test_csv_quoted_multiline(spreadsheet_fixtures_dir):
    parser = CsvParser()
    doc = parser.parse(spreadsheet_fixtures_dir / "quoted_multiline.csv")
    assert "Task A" in doc.text
    assert "second line with comma" in doc.text


def test_csv_unicode(spreadsheet_fixtures_dir):
    parser = CsvParser()
    doc = parser.parse(spreadsheet_fixtures_dir / "unicode.csv")
    assert "bonjour, café!" in doc.text.lower()
    assert "こんにちは" in doc.text
    assert "مرحبا" in doc.text


def test_csv_large_streaming(spreadsheet_fixtures_dir):
    parser = CsvParser(max_rows=50)
    doc = parser.parse(spreadsheet_fixtures_dir / "large.csv")
    assert doc.blocks[0].metadata["rows"] == 49  # 50 rows total including header


def test_csv_empty(spreadsheet_fixtures_dir):
    parser = CsvParser()
    with pytest.raises(EmptyDocumentError):
        parser.parse(spreadsheet_fixtures_dir / "empty.csv")


def test_csv_malformed(spreadsheet_fixtures_dir):
    parser_strict = CsvParser(encoding="ascii", fallback_encodings=False)
    with pytest.raises(CorruptDocumentError):
        parser_strict.parse(spreadsheet_fixtures_dir / "unicode.csv")

    parser_utf8 = CsvParser(encoding="utf-8", fallback_encodings=False)
    with pytest.raises(CorruptDocumentError):
        parser_utf8.parse(spreadsheet_fixtures_dir / "malformed.csv")
