# PaddleOCR Parser

Deep learning based optical character recognition using `PaddleOCR`.

## Capabilities

- **Deep learning recognition**: Robust against rotated text, diverse fonts, and varied layouts.
- **Bounding polygons**: Retains 4-point quadrilateral coordinates for rotated bounding regions.
- **Confidence scores**: Per-text-region model probabilities.

## Installation

```bash
pip install -e '.[paddleocr]'
# or directly
pip install paddlepaddle paddleocr
```

## Known Limitations

- Requires `paddlepaddle` and `paddleocr`. Model weights (~20-50MB) are downloaded on first run.
- Significantly heavier CPU/GPU requirements than Tesseract.
