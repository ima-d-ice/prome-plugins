# MBOX Parser

Multi-message mailbox archive (.mbox) extraction using Python's standard `mailbox` module.

## Capabilities

- **Message boundaries**: Maps each individual email message in the mailbox archive to a `ParsedPage` with an ordered 1-based index.
- **Header & thread tracking**: Preserves all message headers, `In-Reply-To` and `References` thread graphs for each email.
- **Attachment metadata**: Identifies attachment names, MIME types, and sizes as `BlockType.ATTACHMENT` blocks per message.
- **MIME hierarchy**: Unpacks plain text, HTML, and alternative body formats.

## Usage

```python
from parsers.email.mbox import MboxParser

parser = MboxParser()
doc = parser.parse("archive.mbox")

print(f"Total messages: {doc.page_count}")
for page in doc.pages:
    print(f"Message {page.page_number} has {len(page.blocks)} blocks")
```
