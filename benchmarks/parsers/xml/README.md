# XML Benchmark Suite

Benchmarking suite for ElementTree and lxml XML parsers in Prome RAG Plugins.

## Covered Parsers

- `elementtree_xml`: Safe standard-library XML parser with XPath tracking
- `lxml_xml`: Fast, XPath-native XML parser with XXE protection

## Fixtures

- `catalog.xml`: Nested book items with attributes and multiple fields
- `namespaces.xml`: XML using distinct namespace declarations (`xmlns:dc`, `xmlns:meta`)
- `cdata.xml`: Elements enclosing raw CDATA blocks
- `unicode.xml`: Multilingual text in XML nodes
- `empty.xml` / `malformed.xml`: Negative test fixtures

## Running the Benchmark

```bash
python benchmarks/parsers/xml/benchmark.py
```
