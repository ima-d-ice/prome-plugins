# JSON Parser

Structure-preserving JSON and NDJSON/JSONL parser using Python's standard `json` module.

## Capabilities

- **Zero dependencies**: Uses Python's built-in `json` module.
- **Hierarchy preservation**: Preserves deep nesting without lossy flattening.
- **JSON paths**: Emits `BlockType.KEY_VALUE` blocks for each leaf node with exact JSON paths (`$.store.books[0].title`).
- **Data type preservation**: Preserves string, number (int/float), boolean, null, object, and array types in metadata.
- **NDJSON / JSONL**: Automatically recognizes newline-delimited JSON with per-line record mapping.

## Usage

```python
from parsers.json import JsonParser

parser = JsonParser()
doc = parser.parse("data.json")

print(f"Root type: {doc.metadata['root_type']}")
print(f"Total leaves: {doc.metadata['leaf_count']}")
for block in doc.blocks:
    if block.block_type.value == "key_value":
        print(f"{block.metadata['json_path']} = {block.metadata['value']} ({block.metadata['data_type']})")
```
