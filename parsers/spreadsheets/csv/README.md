# CSV Parser

Standard-library based CSV and delimiter-separated values (DSV) extraction.

## Capabilities

- **Zero dependencies**: Uses Python's built-in `csv` module.
- **Dialect detection**: Automatically detects delimiters (`,`, `;`, `\t`, `|`) and header presence via `csv.Sniffer`.
- **Encoding fallback**: Tries UTF-8, UTF-8-SIG, Latin-1, and CP1252 to gracefully decode diverse data files.
- **Quoting & multiline**: Correctly parses quoted cells with embedded delimiters and linebreaks.
- **Streaming & limits**: Supports `max_rows` for controlled partial reading of huge files.

## Usage

```python
from parsers.spreadsheets.csv import CsvParser

parser = CsvParser()
doc = parser.parse("data.csv")

table_block = doc.blocks[0]
print(f"Delimiter: {table_block.metadata['delimiter']}")
print(f"Headers: {table_block.metadata['headers']}")
print(f"Rows: {table_block.metadata['rows']}")
```
