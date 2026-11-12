# python-docx Parser

Native Word document (.docx) extraction using `python-docx`.

## Capabilities

- **Structure preservation**: Extracts headings (with `heading_level`), paragraphs, ordered and unordered lists (`list_type`, `ordered`, `list_level`).
- **Tables**: Extracts tabular data as structured blocks with dimensions (`rows`, `cols`), headers, and row/cell values.
- **Hyperlinks**: Resolves internal and external hyperlinks with URL and anchor metadata.
- **Headers & Footers**: Extracts section headers and footers with region identification.
- **Metadata**: Core Word metadata (title, author, subject, revision, timestamps) and section properties.
- **Provenance**: Preserves original document element order via `document_order` block metadata.
- **Pageless representation**: DOCX files are paginated at render time; extracted blocks reside in `doc.blocks` with `doc.pages == []`.

## Installation

```bash
pip install -e '.[docx]'
# or directly
pip install python-docx
```

## Usage

```python
from parsers.office.docx.python_docx import PythonDocxParser

parser = PythonDocxParser()
doc = parser.parse("path/to/document.docx")

print(f"Title: {doc.metadata.get('title')}")
for block in doc.blocks:
    print(f"[{block.block_type.value}] {block.content[:60]}")
```

## Known Limitations

- Embedded macros, complex drawings (DrawingML shapes), and chart data are not fully extracted.
- Exact page numbers are not available without a Word rendering engine.
