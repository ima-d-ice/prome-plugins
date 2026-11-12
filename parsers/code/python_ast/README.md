# Python AST Parser

Native Python source code and repository extraction using Python's standard `ast` module.

## Capabilities

- **Zero dependencies**: Standard-library AST parsing.
- **Symbol definitions**: Extracts classes, functions, and methods with docstrings, decorators, and signatures.
- **Relationships**: Maps methods to their enclosing parent class.
- **Imports**: Extracts `import` and `from ... import` statements with module names and aliases.
- **Line provenance**: Preserves accurate 1-based `start_line` and `end_line` ranges and raw source slices.

## Usage

```python
from parsers.code.python_ast import PythonAstParser

parser = PythonAstParser()
doc = parser.parse("main.py")

print(f"Classes: {doc.metadata['classes_count']}, Functions: {doc.metadata['functions_count']}")
for block in doc.blocks:
    if block.block_type.value == "definition":
        print(f"[{block.metadata['kind']}] {block.metadata['name']} (Lines {block.metadata['start_line']}-{block.metadata['end_line']})")
```
