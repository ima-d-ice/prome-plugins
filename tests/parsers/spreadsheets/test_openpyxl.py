import pytest
from parsers.contracts import (
    BlockType,
    CorruptDocumentError,
    EmptyDocumentError,
    ParsedDocument,
)
from parsers.spreadsheets.xlsx.openpyxl import OpenpyxlParser


def test_openpyxl_multi_sheet(spreadsheet_fixtures_dir):
    parser = OpenpyxlParser()
    doc = parser.parse(spreadsheet_fixtures_dir / "multi_sheet.xlsx")
    assert isinstance(doc, ParsedDocument)
    assert doc.parser_name == "openpyxl"
    assert doc.page_count == 2
    sheet1 = doc.pages[0]
    assert sheet1.page_number == 1
    assert sheet1.blocks[0].metadata["sheet_name"] == "Sales"
    assert "Alice" in sheet1.text
    sheet2 = doc.pages[1]
    assert sheet2.blocks[0].metadata["sheet_name"] == "Expenses"
    assert "Marketing" in sheet2.text


def test_openpyxl_data_types_and_merged_cells(spreadsheet_fixtures_dir):
    parser = OpenpyxlParser()
    doc = parser.parse(spreadsheet_fixtures_dir / "data_types.xlsx")
    assert doc.page_count == 1
    b = doc.pages[0].blocks[0]
    assert b.block_type == BlockType.TABLE
    assert "A4:B4" in b.metadata.get("merged_cells", [])
    assert any("=B2*2" in f for f in b.metadata.get("formulas", {}).values())


def test_openpyxl_unicode(spreadsheet_fixtures_dir):
    parser = OpenpyxlParser()
    doc = parser.parse(spreadsheet_fixtures_dir / "unicode.xlsx")
    assert "café" in doc.text.lower()
    assert "こんにちは世界" in doc.text
    assert "привет" in doc.text.lower()


def test_openpyxl_empty(spreadsheet_fixtures_dir):
    parser = OpenpyxlParser()
    with pytest.raises(EmptyDocumentError):
        parser.parse(spreadsheet_fixtures_dir / "empty.xlsx")


def test_openpyxl_malformed(spreadsheet_fixtures_dir):
    parser = OpenpyxlParser()
    with pytest.raises(CorruptDocumentError):
        parser.parse(spreadsheet_fixtures_dir / "malformed.xlsx")
