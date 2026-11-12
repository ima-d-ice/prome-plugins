# pandas XLSX Parser

Tabular and DataFrame-oriented Excel (.xlsx) extraction using `pandas`.

## Capabilities

- **Tabular extraction**: Converts each worksheet into a clean DataFrame structure.
- **Column schema & types**: Detects column names and pandas data types (int64, float64, datetime64, object).
- **Missing value analysis**: Computes null/missing value counts per column.
- **Multi-sheet support**: Extracts all worksheets, mapping each to a `ParsedPage`.

## Installation

```bash
pip install -e '.[spreadsheets]'
# or directly
pip install pandas openpyxl
```

## Usage

```python
from parsers.spreadsheets.xlsx.pandas import PandasXlsxParser

parser = PandasXlsxParser()
doc = parser.parse("data.xlsx")

for page in doc.pages:
    print(f"Sheet: {page.blocks[0].metadata['sheet_name']}")
    print(f"Columns: {page.blocks[0].metadata['headers']}")
```

## Capabilities Not Preserved (vs openpyxl)

- **Raw formulas**: pandas evaluates or reads cached formula values; raw formula definitions are not preserved.
- **Merged cells**: Merged cell geometries are flattened; cell spans are not preserved.
- **Visual styling**: Fonts, colors, border styles, and sheet tab colors are omitted.
