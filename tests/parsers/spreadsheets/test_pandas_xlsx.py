import pytest
from parsers.contracts import (
    BlockType,
    CorruptDocumentError,
    EmptyDocumentError,
    ParsedDocument,
)
from parsers.spreadsheets.xlsx.pandas import PandasXlsxParser


def test_pandas_xlsx_multi_sheet(spreadsheet_fixtures_dir):
    parser = PandasXlsxParser()
    doc = parser.parse(spreadsheet_fixtures_dir / "multi_sheet.xlsx")
    assert isinstance(doc, ParsedDocument)
    assert doc.parser_name == "pandas_xlsx"
    assert doc.page_count == 2
    assert "Sales" in doc.pages[0].blocks[0].metadata["sheet_name"]
    assert "Expenses" in doc.pages[1].blocks[0].metadata["sheet_name"]
    assert "Alice" in doc.text


def test_pandas_xlsx_dtypes(spreadsheet_fixtures_dir):
    parser = PandasXlsxParser()
    doc = parser.parse(spreadsheet_fixtures_dir / "data_types.xlsx")
    b = doc.pages[0].blocks[0]
    assert b.block_type == BlockType.TABLE
    dtypes = b.metadata.get("dtypes", {})
    assert "Integer" in dtypes
    assert "Boolean" in dtypes


def test_pandas_xlsx_empty(spreadsheet_fixtures_dir):
    parser = PandasXlsxParser()
    with pytest.raises(EmptyDocumentError):
        parser.parse(spreadsheet_fixtures_dir / "empty.xlsx")


def test_pandas_xlsx_malformed(spreadsheet_fixtures_dir):
    parser = PandasXlsxParser()
    with pytest.raises(CorruptDocumentError):
        parser.parse(spreadsheet_fixtures_dir / "malformed.xlsx")
