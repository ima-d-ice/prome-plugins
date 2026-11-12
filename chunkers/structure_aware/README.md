# Structure-aware chunker

Takes advantage of the structure the Prome parsers already extract (heading
hierarchy, tables, lists, reading order, pages) and chunks along it.
Paragraphs stay attached to their headings; lists and tables are kept whole
whenever they fit; new major sections open a new chunk.

## Behavior

Blocks are flattened in reading order (pages in order, then blocks in order).
The chunker maintains a *heading stack* from `BlockType.HEADING` blocks:

- **Headings are pending.** A heading is never emitted as the tail of a
  chunk; it is held and glued to the content that follows it. (An all-heading
  stretch larger than `chunk_size` is flushed as its own chunk.)
- **Major section boundaries close the chunk.** When a heading's level is the
  same as or higher than the current section's level (i.e. it opens a new
  major section) and `close_on_heading` is true (default), the accumulated
  chunk is closed before the heading arrives. Deeper headings are glued into
  the current chunk.
- **Blocks stay atomic where possible.** Tables, lists and paragraphs are
  added whole; the chunk is closed when the *next* block would overflow —
  the block is never split mid-way to fit.
- **Oversized blocks are split** by descending separators (`\n\n`, `\n`,
  `. `, …, hard cut) and the pieces are packed up to `chunk_size`; all pieces
  keep references into the original block (including sub-block `char_start`/
  `char_end` offsets).
- **Heading context is recorded** in every chunk: `heading` (innermost
  section heading text, or None), `metadata["heading_path"]` (list of heading
  texts) and `metadata["heading_level"]` (level of the innermost open
  section). `metadata["block_types"]` and `metadata["block_count"]` summarize
  the chunk's content.
- Page and block provenance is preserved as with the other strategies; the
  flattened order is the document's page/reading order.

Headings missing a parseable `heading_level` in their block metadata are
treated as **level 1** (a major section boundary). This matches PyMuPDF,
which emits headings without levels, and is a deliberate conservative
default. `heading_level` values of `2` and `"2"` are both accepted.

This is **not** hierarchical (parent/child) chunking: output is a flat list
of chunks.

## Configuration

```python
from chunkers.structure_aware import StructureAwareChunker, StructureAwareConfig

config = StructureAwareConfig(
    chunk_size=1000,      # measured with tokenizer (default: characters)
    tokenizer=None,       # any str -> int callable or TokenCounter
    close_on_heading=True,
)
chunks = StructureAwareChunker(config).chunk(parsed_document)
```

## Usage with a parsed document

```python
from parsers.office.docx.python_docx import PythonDocxParser
from chunkers.structure_aware import StructureAwareChunker, StructureAwareConfig

doc = PythonDocxParser().parse("handbook.docx")
chunks = StructureAwareChunker(StructureAwareConfig(chunk_size=800)).chunk(doc)

for chunk in chunks:
    print(chunk.heading, "->", chunk.metadata["heading_path"])
```

```python
from parsers.spreadsheets.csv import CsvParser

doc = CsvParser().parse("rows.csv")     # one TABLE block
chunks = StructureAwareChunker(StructureAwareConfig(chunk_size=400)).chunk(doc)
# The table is split row-wise across chunks (never mid-row), every row kept,
# every chunk referencing the TABLE block with sub-block offsets.
```

## Provenance

`block_refs` per chunk (page number, block index, block type, char range
within the block), `metadata["char_start"]`/`["char_end"]` offsets into the
flattened document, `pages` covered, `source`/`parser_name`/`parser_version`,
`strategy="structure_aware"`, plus `heading_path`, `heading_level`,
`block_types`, `block_count`.

## Known limitations

- Effectiveness depends on parser structure quality. Parsers that emit flat
  blocks with no headings (e.g. PyPDF) degrade gracefully to size-limited
  packing, but you gain nothing over the recursive chunker except table/list
  atomicity.
- Unleveled headings are all treated as level 1, which can over-split
  level-less documents into many small chunks.
- `heading_path` is bound when a chunk opens (the innermost section stack at
  that moment); headings appearing *inside* a chunk are part of the chunk
  text but are not appended to its `heading_path`.
- Tables/lists larger than `chunk_size` are split (content preserved,
  provenance preserved); nothing keeps a single row together only if even a
  row exceeds the limit — then the row itself may be split.
- Overlap is not supported by design (sections are the natural unit of
  repetition); duplicate coverage is reported as `duplicate_char_rate` (0 by
  construction) in benchmarks.
