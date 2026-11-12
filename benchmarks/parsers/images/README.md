# Image OCR Benchmark Suite

Benchmarking suite for Tesseract and PaddleOCR image parsers in Prome RAG Plugins.

## Covered Parsers

- `tesseract_ocr`: Tesseract OCR with bounding box and word confidence
- `paddleocr`: PaddleOCR deep-learning text recognition (skipped when dependencies are uninstalled)

## Fixtures

- `clean_text.png`: High-contrast printed text
- `receipt.png`: Structured multi-line invoice/receipt text
- `blank.png` / `corrupt.png`: Negative test fixtures

## Running the Benchmark

```bash
python benchmarks/parsers/images/benchmark.py
```
