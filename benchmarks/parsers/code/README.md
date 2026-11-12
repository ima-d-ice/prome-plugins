# Code Benchmark Suite

Benchmarking suite for Python AST and Tree-sitter code parsers in Prome RAG Plugins.

## Covered Parsers

- `python_ast`: Standard-library Python AST parser
- `tree_sitter`: Tree-sitter multi-language AST parser (Python, JS, TS, Java, C, C++, Go)

## Fixtures

- `sample.py`: Python module with classes, methods, docstrings, and imports
- `sample.js`: JavaScript class, methods, and functions
- `sample.ts`: TypeScript interface, class, and typed function
- `sample.java`: Java package, imports, and class methods
- `sample.c`: C header includes, helper function, and `main()`
- `sample.cpp`: C++ includes, class with methods
- `sample.go`: Go package, imports, struct, and methods
- `empty.py` / `malformed.py`: Negative test fixtures

## Running the Benchmark

```bash
python benchmarks/parsers/code/benchmark.py
```
