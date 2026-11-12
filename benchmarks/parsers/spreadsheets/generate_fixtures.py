"""Generate deterministic synthetic XLSX/CSV fixtures for spreadsheet benchmarks."""

from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path
import openpyxl
from openpyxl.styles import Font

FIXTURES = Path(__file__).parent / "fixtures"


def make_multi_sheet() -> None:
    wb = openpyxl.Workbook()
    # Sheet 1: Sales
    ws1 = wb.active
    ws1.title = "Sales"
    ws1.append(["Region", "Rep", "Units", "Revenue", "Date"])
    ws1.append(["North", "Alice", 150, 2450.50, date(2025, 1, 15)])
    ws1.append(["South", "Bob", 200, 3100.00, date(2025, 1, 16)])
    ws1.append(["East", "Charlie", 75, 1200.25, date(2025, 1, 17)])

    # Sheet 2: Expenses
    ws2 = wb.create_sheet(title="Expenses")
    ws2.append(["Category", "Budget", "Actual"])
    ws2.append(["Marketing", 5000, 4850])
    ws2.append(["R&D", 12000, 11500])
    ws2.append(["Operations", 3000, 3200])

    wb.save(FIXTURES / "multi_sheet.xlsx")


def make_data_types() -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Types"
    ws.append(["String", "Integer", "Float", "Boolean", "Date", "Formula", "Empty"])
    ws.append(["Alpha", 42, 3.14159, True, date(2025, 5, 20), "=B2*2", None])
    ws.append(["Beta", 100, 0.001, False, date(2025, 6, 1), "=B3+C3", None])
    # Add a merged cell
    ws.merge_cells("A4:B4")
    ws["A4"] = "Merged Cell Value"
    wb.save(FIXTURES / "data_types.xlsx")


def make_merged_cells() -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "MergedReport"
    ws.merge_cells("A1:C1")
    ws["A1"] = "Executive Summary 2025"
    ws["A1"].font = Font(bold=True)
    ws.append([])
    ws.append(["Department", "Q1 Actual", "Q2 Actual"])
    ws.append(["Sales", 10000, 12000])
    ws.append(["Support", 4000, 4500])
    ws.append(["Total", "=SUM(B4:B5)", "=SUM(C4:C5)"])
    wb.save(FIXTURES / "merged_cells.xlsx")


def make_unicode_xlsx() -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Unicode"
    ws.append(["Language", "Phrase", "Notes"])
    ws.append(["French", "Café au lait et croissants", "Accents"])
    ws.append(["Japanese", "こんにちは世界", "Kanji & Hiragana"])
    ws.append(["Greek", "Καλημέρα κόσμε", "Greek characters"])
    ws.append(["Russian", "Привет мир", "Cyrillic"])
    ws.append(["Arabic", "مرحبا بالعالم", "RTL"])
    wb.save(FIXTURES / "unicode.xlsx")


def make_empty_xlsx() -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "EmptySheet"
    wb.save(FIXTURES / "empty.xlsx")


def make_malformed_xlsx() -> None:
    (FIXTURES / "malformed.xlsx").write_bytes(b"NOT AN XLSX ZIP ARCHIVE")


def make_standard_csv() -> None:
    with (FIXTURES / "standard.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Name", "Department", "Salary", "HireDate"])
        writer.writerow([101, "Alice Smith", "Engineering", 125000, "2021-03-15"])
        writer.writerow([102, "Bob Jones", "Design", 95000, "2022-06-01"])
        writer.writerow([103, "Carol White", "Product", 110000, "2020-11-20"])


def make_semicolon_csv() -> None:
    with (FIXTURES / "semicolon.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["Product", "Price", "VatRate", "InStock"])
        writer.writerow(["Hammer", "15.99", "0.20", "true"])
        writer.writerow(["Nails (box)", "4.50", "0.20", "true"])
        writer.writerow(["Screwdriver", "8.75", "0.20", "false"])


def make_pipe_csv() -> None:
    with (FIXTURES / "pipe_delimited.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["Code", "City", "Country"])
        writer.writerow(["NYC", "New York", "USA"])
        writer.writerow(["LON", "London", "UK"])
        writer.writerow(["TYO", "Tokyo", "Japan"])


def make_quoted_csv() -> None:
    with (FIXTURES / "quoted_multiline.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["ID", "Title", "Description"])
        writer.writerow([1, "Task A", "First line,\nsecond line with comma."])
        writer.writerow([2, "Task B", 'Contains "quotes" inside'])


def make_unicode_csv() -> None:
    with (FIXTURES / "unicode.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Lang", "Greeting"])
        writer.writerow(["French", "Bonjour, café!"])
        writer.writerow(["Japanese", "こんにちは"])
        writer.writerow(["Arabic", "مرحبا"])


def make_empty_csv() -> None:
    (FIXTURES / "empty.csv").write_bytes(b"")


def make_malformed_csv() -> None:
    (FIXTURES / "malformed.csv").write_bytes(b"\xff\xfe\x00\x00InvalidEncoding")


def make_large_csv() -> None:
    with (FIXTURES / "large.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Sensor", "Value", "Status"])
        for i in range(1, 2001):
            writer.writerow([i, f"sensor_{i % 10}", round(20.0 + (i % 50) * 0.5, 2), "OK"])


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    make_multi_sheet()
    make_data_types()
    make_merged_cells()
    make_unicode_xlsx()
    make_empty_xlsx()
    make_malformed_xlsx()
    make_standard_csv()
    make_semicolon_csv()
    make_pipe_csv()
    make_quoted_csv()
    make_unicode_csv()
    make_empty_csv()
    make_malformed_csv()
    make_large_csv()
    print(f"Generated spreadsheet fixtures in {FIXTURES}")


if __name__ == "__main__":
    main()
