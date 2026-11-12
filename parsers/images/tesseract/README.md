# Tesseract OCR Parser

Image OCR text extraction with spatial bounding boxes and word confidence using Tesseract.

## Capabilities

- **Formats**: PNG, JPEG, TIFF, BMP, WEBP.
- **Bounding boxes**: Provides exact word and line-level bounding boxes (`BoundingBox(x0, y0, x1, y1)`).
- **Confidence scores**: Per-word and line-averaged OCR confidence percentages.
- **Page structure**: Maps single-frame images to `ParsedPage(page_number=1)` with image pixel dimensions.

## Requirements

Requires system Tesseract OCR executable:
- macOS: `brew install tesseract`
- Ubuntu/Debian: `sudo apt-get install tesseract-ocr`

## Usage

```python
from parsers.images.tesseract import TesseractParser

parser = TesseractParser()
doc = parser.parse("document_scan.png")

for block in doc.pages[0].blocks:
    print(f"[{block.metadata['confidence']}%] {block.content} at {block.bbox}")
```

## Known Limitations

- OCR is text extraction only; complex chart structures and vector diagrams are not semantically parsed.
