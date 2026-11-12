# BeautifulSoup HTML Parser (`parsers.html.beautifulsoup`)

General-purpose, structure-first HTML extraction. Preserves the original
document structure as much as reasonably possible: headings with levels,
nested lists, tables with headers/rows/cols (incl. `colspan` expansion),
links with `href`, images with `alt`, code blocks, and full DOM ordering.

This parser does **not** download URLs, execute scripts, evaluate CSS, or do
any network access. It is not a crawler.

## Install

```bash
pip install 'prome-rag-plugins[beautifulsoup]'
# or: pip install beautifulsoup4
```

Optional parsing backends (default works out of the box with the stdlib):

```bash
pip install lxml       # more lenient HTML5-ish parsing for malformed input
pip install html5lib   # spec-compliant parsing (slowest)
```

## Usage

```python
from parsers.html.beautifulsoup import BeautifulSoupParser

parser = BeautifulSoupParser()                      # backend="html.parser"
doc = parser.parse("page.html")                     # file path
doc = parser.parse("<html>...</html>")              # raw HTML string
doc = parser.parse(html_bytes)                      # raw bytes
doc = parser.parse("page.html", source_url="https://example.com/page")

for block in doc.blocks:
    print(block.block_type.value, block.metadata.get("dom_order"), block.content[:60])
```

HTML has no pages: blocks live on `doc.blocks` with `page_number=None`
(`doc.pages == []`, `doc.page_count == 0`). Block order is DOM order; every
block carries `metadata["dom_order"]` (0-based, strictly increasing) and
`metadata["tag"]` (source element name).

## Options

| Argument | Default | Effect |
|---|---|---|
| `backend` | `"html.parser"` | `"html.parser"` (stdlib, lightweight), `"lxml"` or `"html5lib"` (both optional installs) |
| `include_boilerplate` | `True` | Keep `<nav>`, `<header>`, `<footer>`, `<aside>` and `role=navigation/banner/contentinfo/complementary` subtrees. Set `False` to drop them. |

## Extraction behavior

- Headings: `h1`–`h6` → `HEADING` with `metadata["heading_level"]`.
- Paragraphs, blockquotes (`metadata["quote"]`), figcaptions → `TEXT`.
- `<pre>` → `CODE` (language from `language-*`/`lang-*` class).
- `<ul>`/`<ol>` → `LIST` with indented rendering, `metadata["items"]`,
  `metadata["ordered"]`, `metadata["depth"]`. Nested lists are preserved.
- `<dl>` → `LIST` with `metadata["list_type"] == "dl"`.
- `<table>` → `TABLE`; `<thead>`/`<th>` become `metadata["headers"]`,
  `metadata["row_cells"]` holds the full 2-D grid, `metadata["has_header"]`,
  `metadata["rows"]`, `metadata["cols"]`, optional `metadata["caption"]`.
- `<a href>` → `LINK` with `metadata["href"]`; `<img>` → `IMAGE` with
  `content = alt`, plus `src`/`alt`/`title`/`width`/`height` in metadata.
- Metadata: `<title>`, `description`, `author`, `keywords`, `generator`,
  `canonical`, all `og:*`/`twitter:*`/`article:*` properties (flat keys),
  `lang`, and parsed `application/ld+json` as `metadata["json_ld"]`
  (list of decoded objects; `{"raw": ..., "parse_error": true}` on failure).
- `source_url`: constructor/`parse()` argument wins; otherwise falls back to
  canonical link, then `og:url`. Never fabricated.
- `<script>`, `<style>`, `<noscript>`, `<template>`, `<iframe>`, `<svg>` are
  ignored and never leak into output.
- Generic containers (`div`, `section`, `main`, ...) emit their own residual
  inline text without duplicating text captured by nested blocks.
- Links inside list/table/code subtrees are kept in the surrounding text but
  not emitted as separate `LINK` blocks.
- Links/images inside `<table>` cells are not emitted as blocks (their text is
  preserved in the table's cell content).

## Extraction scope

This parser performs **full-DOM** extraction: navigation/header/footer are
retained by default because they are part of the authored document and may
carry useful provenance. Use `include_boilerplate=False` for a
content-first extraction (note: Trafilatura is the stronger tool for
pure main-content isolation).

## Error handling

Same contract as all Prome parsers: `FileNotFoundError` (missing path),
`EmptyDocumentError` (0 bytes / empty input), `CorruptDocumentError` (backend
failure), `ParserDependencyError` (missing `beautifulsoup4` or requested
backend) with install instructions. Malformed HTML never raises — the parser
is lenient by design (backend-dependent).

## Backend notes

The default `html.parser` backend is fast and dependency-free but nests
unclosed elements conservatively (e.g. an unclosed `<h1>` will swallow
following siblings). For malformed production pages, prefer
`backend="lxml"`.
