# Tree-sitter Code Parser

Multi-language AST-based source code extraction using `tree-sitter`.

## Capabilities

- **Multi-language support**: Python, JavaScript, TypeScript, Java, C, C++, and Go.
- **Symbol extraction**: Classes, interfaces, functions, methods, and import statements.
- **Precise provenance**: Exact 1-based start and end line ranges, column positions, and source code slices.
- **Syntax types**: Retains underlying tree-sitter AST node types in block metadata.

## Installation

```bash
pip install -e '.[code]'
# or directly
pip install tree-sitter tree-sitter-python tree-sitter-javascript tree-sitter-typescript tree-sitter-java tree-sitter-c tree-sitter-cpp tree-sitter-go
```

## Usage

```python
from parsers.code.tree_sitter import TreeSitterParser

parser = TreeSitterParser()
doc = parser.parse("handler.go")

print(f"Language: {doc.metadata['language']}")
for block in doc.blocks:
    if block.block_type.value == "definition":
        print(f"[{block.metadata['kind']}] {block.metadata['name']} ({block.metadata['start_line']}-{block.metadata['end_line']})")
```
