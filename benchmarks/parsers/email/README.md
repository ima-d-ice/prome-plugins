# Email Parsers Benchmark Suite

Benchmarking suite for EML and MBOX parsers in Prome RAG Plugins.

## Covered Parsers

- `eml_stdlib`: Standard-library RFC 822 / MIME parser
- `mbox_stdlib`: Standard-library multi-message mailbox archive parser

## Fixtures

- `simple_plain.eml`: Standard headers and plain text body
- `html_email.eml`: HTML body message
- `multipart_attachment.eml`: Mixed multipart message with PDF attachment
- `thread_reply.eml`: Reply email with `In-Reply-To` and `References`
- `unicode.eml`: Encoded international headers and UTF-8 body
- `archive.mbox`: Multi-message mailbox archive
- `empty.eml` / `malformed.eml`: Negative test fixtures

## Running the Benchmark

```bash
python benchmarks/parsers/email/benchmark.py
```
