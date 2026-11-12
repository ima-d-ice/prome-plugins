# EML Parser

RFC 822 and MIME `.eml` email message extraction using Python's standard library.

## Capabilities

- **Header extraction**: Extracts `From`, `To`, `Cc`, `Bcc`, `Subject`, `Date`, `Message-ID`, `In-Reply-To`, and `References`.
- **Thread relationships**: Automatically identifies reply/thread relationships from `In-Reply-To` and `References` headers without guesswork.
- **Multipart MIME handling**: Unpacks nested MIME parts into plain text, HTML, and attachment blocks.
- **Attachment provenance**: Extracts attachment metadata (filename, content type, byte size, content disposition) as `BlockType.ATTACHMENT` blocks without executing arbitrary parsers.
- **Header decoding**: Safely decodes RFC 2047 encoded words (`=?UTF-8?...?=`).

## Usage

```python
from parsers.email.eml import EmlParser

parser = EmlParser()
doc = parser.parse("message.eml")

print(f"Subject: {doc.metadata['subject']}")
print(f"From: {doc.metadata['from']}")
print(f"Is Reply: {doc.metadata['is_thread_reply']}")
for block in doc.blocks:
    print(f"[{block.block_type.value}] {block.content[:60]}")
```
