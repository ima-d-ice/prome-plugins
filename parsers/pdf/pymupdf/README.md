# PyMuPDF PDF Parser

Fast text extraction from digitally generated PDFs using PyMuPDF.

- Page-wise text with bounding boxes and page dimensions
- Heading detection heuristic (font size >= 13pt and >= 1.2x the 10th-percentile body size)
- Table detection and extraction via `page.find_tables()`
- PDF metadata extraction
- Uses PyMuPDF's natural block order (rough reading order for digital PDFs)

## Install

```bash
pip install 'prome-rag-plugins[pymupdf]'
# or
pip install pymupdf
```

## Usage

```python
from parsers.pdf.pymupdf import PyMuPDFParser

doc = PyMuPDFParser().parse("document.pdf")
for page in doc.pages:
    for block in page.blocks:
        print(page.page_number, block.block_type.value, block.content[:80])
```

Raises `ParserDependencyError` when PyMuPDF is missing, `EncryptedDocumentError`
for password-protected PDFs, `CorruptDocumentError` for unreadable files and
`EmptyDocumentError` for zero-page PDFs.
