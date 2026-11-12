# Trafilatura HTML Parser (`parsers.html.trafilatura`)

Main-content extraction for HTML: isolate the meaningful article/document
text and drop boilerplate (navigation, headers, sidebars, footers). Where
Trafilatura retains structure (headings, paragraphs, lists, tables, code,
inline links), it is emitted as blocks. Where it does not (images, arbitrary
inline markup), absence is reported honestly instead of being fabricated.

This parser does **not** download URLs, execute scripts, or do any network
access. It is not a crawler.

## Install

```bash
pip install 'prome-rag-plugins[trafilatura]'
# or: pip install trafilatura
```

## Usage

```python
from parsers.html.trafilatura import TrafilaturaParser

parser = TrafilaturaParser()                        # main-content defaults
doc = parser.parse("page.html")                     # file path
doc = parser.parse("<html>...</html>")              # raw HTML string
doc = parser.parse(html_bytes)                      # raw bytes
doc = parser.parse("page.html", source_url="https://example.com/page")

for block in doc.blocks:
    print(block.block_type.value, block.metadata.get("dom_order"), block.content[:60])
```

HTML has no pages: blocks live on `doc.blocks` with `page_number=None`
(`doc.pages == []`, `doc.page_count == 0`). Block order is extraction
(document) order; every block carries `metadata["dom_order"]` and
`metadata["tag"]`.

## Options

| Argument | Default | Effect |
|---|---|---|
| `favor_precision` | `False` | Prefer less text with higher accuracy |
| `favor_recall` | `False` | Prefer more text even when unsure (mutually exclusive with `favor_precision`; passing both raises `ValueError`) |
| `include_tables` | `True` | Keep `<table>` elements (disable to drop tables) |
| `include_formatting` | `True` | Keep formatting-related structure in the extracted body |
| `include_links` | `True` | Keep `<ref>` links; emitted as `LINK` blocks with `metadata["href"]` and kept inline in text |
| `include_images` | `False` | Trafilatura's image support is minimal; images are not surfaced as blocks regardless — reported as unsupported, never as 0 |
| `include_comments` | `False` | Extract reader comments alongside the main text |
| `target_language` | `None` | ISO 639-1 code; documents of other languages are discarded |

## Extraction behavior

- Main content: via `trafilatura.bare_extraction(..., output_format="xml", with_metadata=True)`.
- Headings: `<head rend="h1">` … → `HEADING` with `metadata["heading_level"]`.
- Paragraphs → `TEXT`; `<quote>` → `TEXT` (children recursed, `metadata["quote"]`).
- Lists: `<list rend="ul|ol">` → `LIST` with the same rendering as the
  BeautifulSoup plugin; nested lists preserved with `metadata["depth"]`.
- Tables: `<table>/<row>/<cell role="head">` → `TABLE` with `metadata["headers"]`,
  `metadata["row_cells"]`, `metadata["has_header"]`, `metadata["rows"]`, `metadata["cols"]`.
- Code: `<code>` (possibly nested) → `CODE`; language from `language-*` class.
- Links: inline `<ref target="...">` → `LINK` with `metadata["href"]`.
- Metadata: `title`, `author`, `date`, `description`, `sitename`, `hostname`,
  `language`, `pagetype`, `license`, `image`, `categories`, `tags`, plus
  `source_url` (from the `source_url` argument or the page's canonical/og:url).
- `date` is the **publication date** discovered from the page itself
  (meta tags, JSON-LD, `<time>` elements, URLs). htmldate's extensive
  free-text search is disabled by default, so unrelated text such as
  "2000 docs/min" or a copyright year is never reported as a publication
  date. Set `extensive_date_search=True` to re-enable the lenient heuristic.
  Pages without a discoverable publication date yield **no** `date` key.
  Trafilatura's `filedate` field (the search upper bound, i.e. today) is
  never reported as a date; ingestion timestamps are the ingestion layer's
  responsibility, not the parser's.

## No-content pages

If Trafilatura cannot identify a main content region (e.g. an empty
`<body>`), the parser returns a `ParsedDocument` with **no blocks**, a
warning `"No main content could be identified in the document."`, and
whatever metadata it can still recover (e.g. `<title>` via a metadata-only
fallback). No exception is raised.

## Capability gaps (reported, not fabricated)

- Images: no `IMAGE` blocks are produced (`include_images=True` still only
  yields a metadata URL).
- Boilerplate regions are dropped by design — that is the point of this
  parser. Use the BeautifulSoup plugin for full-DOM retention.
- Author/date come from whatever the page exposes (meta tags, JSON-LD);
  pages without them simply omit the keys.

## Error handling

Same contract as all Prome parsers: `FileNotFoundError` (missing path),
`EmptyDocumentError` (0 bytes / empty input), `DocumentParseError`
(backend failure), `ParserDependencyError` (missing `trafilatura`) with
install instructions.
