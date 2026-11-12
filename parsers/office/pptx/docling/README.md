# Docling PPTX Parser

Layout-aware PowerPoint presentation (.pptx) extraction using IBM's Docling engine.

## Capabilities

- **Slide preservation**: Maps presentation slides to `ParsedPage` instances with spatial dimensions and boundaries.
- **Hierarchical elements**: Identifies titles, section headers, text blocks, and list items.
- **Tables**: Parses table elements into Markdown and structured cell grids.
- **Images**: Extracts image references and spatial coordinates.
- **Bounding boxes**: Provides normalized coordinates for rendered elements.

## Installation

```bash
pip install -e '.[docling]'
# or directly
pip install docling
```

## Usage

```python
from parsers.office.pptx.docling import DoclingPptxParser

parser = DoclingPptxParser()
doc = parser.parse("presentation.pptx")

for page in doc.pages:
    print(f"Slide {page.page_number} ({len(page.blocks)} blocks):")
    for block in page.blocks:
        print(f"  [{block.block_type.value}] {block.content[:60]}")
```

## Known Limitations

- Requires `docling` package.
- Heavier initial startup time compared to `python-pptx`.
