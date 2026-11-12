# Docling DOCX Parser

Structure-aware DOCX parsing using IBM's Docling document understanding engine.

## Capabilities

- **Unified document model**: Parses DOCX files using Docling's AST with hierarchical levels.
- **Headings & structure**: Preserves heading hierarchy, sections, and paragraphs.
- **Tables**: Exports table data to clean Markdown representations and extracts row/column cell matrices.
- **Lists**: Preserves list items with hierarchical nesting.
- **Pageless representation**: DOCX blocks are stored in `doc.blocks` with `doc.pages == []`.

## Installation

```bash
pip install -e '.[docling]'
# or directly
pip install docling
```

## Usage

```python
from parsers.office.docx.docling import DoclingDocxParser

parser = DoclingDocxParser()
doc = parser.parse("path/to/document.docx")

for block in doc.blocks:
    print(f"[{block.block_type.value}] {block.content[:60]}")
```

## Known Limitations

- Requires `docling` package.
- Slower cold-start than native `python-docx` parser.
