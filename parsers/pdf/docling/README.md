# Docling PDF Parser

Layout-aware PDF extraction using Docling. Best structure fidelity of the
three implementations (headings, tables as markdown, reading order) at the
cost of speed and dependency weight.

- Headings, paragraphs, lists, tables and other document elements
- Table content exported as markdown
- Page references and bounding boxes (normalized to top-left origin)
- Provenance via Docling's item provenance (`page_no`, `bbox`)
- OCR is **disabled** (`PdfPipelineOptions(do_ocr=False)`); OCR belongs to the
  future `ocr/` directory

## Install

```bash
pip install 'prome-rag-plugins[docling]'
# or
pip install docling
```

Note: Docling is large and downloads model weights on first use.

## Usage

```python
from parsers.pdf.docling import DoclingParser

doc = DoclingParser().parse("document.pdf")
for page in doc.pages:
    for block in page.blocks:
        print(page.page_number, block.block_type.value, block.content[:80])
```

Raises `ParserDependencyError` when Docling is missing, `EncryptedDocumentError`
for password-protected PDFs (message-dependent, detected via Docling's error),
`CorruptDocumentError` for unreadable files and `EmptyDocumentError` for
zero-page PDFs.
