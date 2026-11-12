# python-pptx Parser

Structure-aware PowerPoint presentation (.pptx) extraction using `python-pptx`.

## Capabilities

- **Slide boundaries**: Maps each presentation slide to a `ParsedPage` preserving temporal order and dimensions.
- **Titles & hierarchy**: Distinguishes slide title placeholders as `BlockType.HEADING`.
- **Text & bullet lists**: Extracts text frames, retaining paragraph nesting and bullet levels (`level > 0` mapped to `BlockType.LIST`).
- **Tables**: Parses table shapes into row/column grids with headers and Markdown representations.
- **Images & shapes**: Captures images, shape names, types, and spatial bounding boxes (in points).
- **Speaker notes**: Extracts speaker notes associated with each slide under `is_speaker_note=True`.

## Installation

```bash
pip install -e '.[pptx]'
# or directly
pip install python-pptx
```

## Usage

```python
from parsers.office.pptx.python_pptx import PythonPptxParser

parser = PythonPptxParser()
doc = parser.parse("presentation.pptx")

for page in doc.pages:
    print(f"--- Slide {page.page_number} ---")
    for block in page.blocks:
        print(f"[{block.block_type.value}] {block.content[:60]}")
```

## Known Limitations

- Embedded SmartArt and charts are parsed as generic shapes or text elements.
- Animations and slide transition timings are not captured.
