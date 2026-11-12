# Recursive chunker

Splits the flattened document text with an ordered list of separators, then
packs the pieces greedily into chunks. Splitting prefers the coarsest
structure available — paragraphs and blank lines first, then lines, then
sentences, then words — and only falls back to a hard character cut when no
separator exists.

This is **not** hierarchical (parent/child) chunking: the output is a flat
list of chunks. Parent/child relationships belong to the later hierarchical
milestone and are intentionally absent.

## Behavior

- `recursive_split` descends the separator list level by level; a too-large
  span is divided by the first separator present in it, and each piece that
  still exceeds `chunk_size` is split again with the next separators.
- `merge_spans` packs the atomic pieces left-to-right into chunks of at most
  `chunk_size` units; after a chunk is emitted, the longest trailing run of
  atoms whose counted size fits `overlap` starts the next chunk.
- The final separator `""` is a hard cut. For character counters the window
  is `chunk_size` characters; for tokenizer counters it is
  `4 * chunk_size` characters (a documented heuristic, since token width
  varies).
- Every chunk is an exact slice of the flattened document text; output is
  deterministic.

Default separators: `\n\n`, `\n`, `. `, `? `, `! `, `; `, `, `, ` `, `""`.

## Configuration

```python
from chunkers.recursive import RecursiveChunker, RecursiveConfig

config = RecursiveConfig(
    chunk_size=1000,       # characters by default
    overlap=200,           # >= 0 and < chunk_size
    tokenizer=None,        # any str -> int callable or TokenCounter
    separators=("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""),
)
chunks = RecursiveChunker(config).chunk(parsed_document)
```

Custom separators (e.g. spreadsheet rows, pipe-delimited data):

```python
RecursiveConfig(chunk_size=800, overlap=0, separators=("\n", "|", " ", ""))
```

Invalid values raise `ChunkerConfigError`.

## Usage with a parsed document

```python
from parsers.pdf.pypdf import PyPDFParser
from chunkers.recursive import RecursiveChunker, RecursiveConfig

doc = PyPDFParser().parse("scan.pdf")   # works even without heading metadata
chunks = RecursiveChunker(RecursiveConfig(chunk_size=500, overlap=50)).chunk(doc)
```

The recursive chunker is a good default when the parser produced little or
unreliable structure metadata but the text itself has natural boundaries.

## Provenance

Same as the fixed chunker: `block_refs` (page number, block index, block
type, char range within the block), `metadata["char_start"]`/`["char_end"]`,
covered `pages`, `source`/`parser_name`/`parser_version`. `heading` is `None`
(this strategy does not consult block types).

## Known limitations

- Ignores block types: a table row set or a code block can be split across
  chunks (it respects their line/sentence separators first, which usually
  lands on row boundaries, but nothing guarantees atomicity).
- Overlap depends on atomic piece boundaries — a chunk carried from the
  tail may be smaller than the configured `overlap` if the trailing pieces
  are large.
- The tokenizer-mode hard cut uses a 4-characters-per-token heuristic window,
  so a pathological unbreakable token may exceed `chunk_size` slightly.
- Sentence splitting is punctuation-driven and will misfire on inputs like
  decimal numbers and abbreviations ("3.14", "e.g.").
