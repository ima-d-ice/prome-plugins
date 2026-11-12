# openpyxl XLSX Parser

Structure-preserving Excel (.xlsx) extraction using `openpyxl`.

## Capabilities

- **Structure preservation**: Maps sheets into separate `ParsedPage`s with ordered indices.
- **Cell coordinates & types**: Preserves coordinate relationships ("A1", "B2") and openpyxl data types (numeric, string, date, boolean, formula).
- **Formulas**: Keeps raw formula strings intact without silently converting them to incorrect values.
- **Merged cells**: Records merged cell spans in block metadata.
- **Sheet visibility**: Extracts hidden vs visible status and sheet dimensions.
- **Table grids**: Exports clean Markdown tables and retains structured cell matrices under `metadata["row_cells"]`.

## Installation

```bash
pip install -e '.[spreadsheets]'
# or directly
pip install openpyxl
```

## Usage

```python
from parsers.spreadsheets.xlsx.openpyxl import OpenpyxlParser

parser = OpenpyxlParser()
doc = parser.parse("data.xlsx")

for page in doc.pages:
    print(f"Sheet: {page.blocks[0].metadata['sheet_name']}")
    print(page.blocks[0].content)
```

## Known Limitations

- Cell formatting (colors, fonts, borders) and charts are not parsed.
- Cached formula values are only accessible if saved with a formula evaluation engine (Excel/LibreOffice).
