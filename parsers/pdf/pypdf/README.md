# PyPDF PDF Parser

Lightweight text extraction using pypdf. The simplest alternative for
comparison and for very small footprints.

- Page-wise text in page order
- One text block per page (no layout/structure analysis)
- Basic PDF metadata (keys normalized, leading `/` stripped)
- Page dimensions from the media box

## Install

```bash
pip install 'prome-rag-plugins[pypdf]'
# or
pip install 'pypdf[crypto]'   # crypto extra enables AES handling for encrypted PDFs
```

## Usage

```python
from parsers.pdf.pypdf import PyPDFParser

doc = PyPDFParser().parse("document.pdf")
print(doc.text)
```

Raises `ParserDependencyError` when pypdf is missing, `EncryptedDocumentError`
for password-protected PDFs, `CorruptDocumentError` for unreadable files and
`EmptyDocumentError` for zero-page PDFs.
