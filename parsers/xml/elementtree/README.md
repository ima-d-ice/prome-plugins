# ElementTree XML Parser

Safe, zero-dependency XML extraction using Python's standard `xml.etree.ElementTree`.

## Capabilities

- **Zero dependencies**: Standard-library implementation.
- **Security hardened**: Disallows external entity resolution (XXE prevention) and prohibits external DTD fetching.
- **Hierarchy & XPath**: Automatically computes indexed XPath locations for each element (`/catalog/book[2]/title`).
- **Namespaces & attributes**: Separates local element tags from XML namespace URIs and extracts all element attributes.

## Usage

```python
from parsers.xml.elementtree import ElementTreeXmlParser

parser = ElementTreeXmlParser()
doc = parser.parse("data.xml")

for block in doc.blocks:
    print(f"{block.metadata['xpath']}: {block.content}")
```
