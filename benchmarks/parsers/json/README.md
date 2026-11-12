# JSON Benchmark Suite

Benchmarking suite for JSON and NDJSON parsers in Prome RAG Plugins.

## Covered Parsers

- `json_stdlib`: Standard-library JSON/NDJSON parser with JSON-path hierarchy tracking

## Fixtures

- `nested_object.json`: Multi-level nested dictionary and array structures
- `primitives.json`: Scalar values (strings, integers, floats, booleans, null)
- `stream.ndjson`: Newline-delimited JSON log records
- `empty.json` / `malformed.json`: Negative test fixtures

## Running the Benchmark

```bash
python benchmarks/parsers/json/benchmark.py
```
