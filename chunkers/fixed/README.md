# Fixed-size chunker

Splits the flattened document text into uniform windows of at most
`chunk_size` units with `overlap` units of repeated context between
consecutive chunks.

## Behavior

- Character mode (default) computes direct character offsets. With
  `respect_word_boundaries=True` (default) a window edge is snapped back to
  the last whitespace inside the window when one exists, and overlap rewinds
  that land inside a word are snapped forward to the next word boundary.
  Chunks are exact slices of the flattened document text.
- Tokenizer mode (`tokenizer=` set; anything non-character) packs word-like
  atoms (`\S+\s*`) until the counted limit is reached; overlap is re-measured
  in the same units by carrying trailing atoms.
- Unbreakable runs longer than `chunk_size` are kept whole in tokenizer
  mode and hard-cut in character mode (limits are best-effort).
- Output is deterministic and every chunk is non-empty.

## Configuration

```python
from chunkers.fixed import FixedSizeChunker, FixedSizeConfig

config = FixedSizeConfig(
    chunk_size=1000,              # characters by default
    overlap=200,                  # must be >= 0 and < chunk_size
    tokenizer=None,               # e.g. tiktoken encoder, ApproxTokenCounter(), or None
    respect_word_boundaries=True,
)
chunks = FixedSizeChunker(config).chunk(parsed_document)
```

Invalid values raise `ChunkerConfigError` (a `ValueError`):

```python
FixedSizeConfig(chunk_size=0)            # ChunkerConfigError
FixedSizeConfig(chunk_size=100, overlap=100)  # ChunkerConfigError: overlap >= chunk_size
```

## Usage with a parsed document

```python
from parsers.html.beautifulsoup import BeautifulSoupParser
from chunkers.fixed import FixedSizeChunker, FixedSizeConfig

doc = BeautifulSoupParser().parse("article.html")
chunks = FixedSizeChunker(FixedSizeConfig(chunk_size=500, overlap=50)).chunk(doc)

for chunk in chunks:
    print(chunk.index, chunk.count, chunk.length, chunk.length_unit)
    for ref in chunk.block_refs:
        print("   ", ref.page_number, ref.block_index, ref.block_type, ref.char_start, ref.char_end)
```

## Provenance

Every chunk carries `block_refs` into the original document (page number,
block index, block type, character range within the block) plus
`metadata["char_start"]`/`metadata["char_end"]` offsets into the flattened
document text, the covered `pages`, `source`, `parser_name` and
`parser_version`. `heading` is always `None` (this strategy is
structure-agnostic).

## Known limitations

- Ignores block types: tables, lists and code are split mid-block like any
  other text. Use the structure-aware chunker when block integrity matters.
- Overlap is a literal repeat of earlier text; the duplicated context is
  reported honestly as `duplicate_char_rate` in benchmarks, not hidden.
- Word-boundary snapping breaks windows mid-word only when no whitespace
  exists in the window.
