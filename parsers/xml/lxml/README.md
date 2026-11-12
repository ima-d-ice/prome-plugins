# lxml XML Parser

High-performance, XPath-rich XML extraction using `lxml`.

## Capabilities

- **Native XPath generation**: Accurate element locations via `tree.getpath()`.
- **XXE and DTD security**: Fully configured with `resolve_entities=False`, `load_dtd=False`, and `no_network=True`.
- **Namespaces**: Preserves namespace mappings (`xmlns`) and attributes.
- **CDATA support**: Automatically decodes embedded CDATA text sections.

## Installation

```bash
pip install -e '.[xml]'
# or directly
pip install lxml
```

## Usage

```python
from parsers.xml.lxml import LxmlXmlParser

parser = LxmlXmlParser()
doc = parser.parse("feed.xml")

for block in doc.blocks:
    print(f"[{block.block_type.value}] {block.content}")
```
